"""
Benchy-Graph

This script generates a performance dashboard from llama-benchy CSV benchmark data,
visualizing throughput, latency, and other metrics for different phases
and concurrency levels.

Usage:
    python test.py <input.csv> <output.png>

Dependencies:
    - pandas
    - matplotlib
    - seaborn
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

COLORS = {
    'pp': '#2E86AB',
    'tg': '#A23B72',
    'pp_tsr': '#7FB3D5',
    'tg_tsr': '#D98880',
    'pp_light': '#A7C7E7',
    'tg_light': '#E7B8D6'
}

PLOT_STYLE = {
    'figure.figsize': [14, 10],
    'axes.linewidth': 1.2,
    'font_scale': 1.1
}

DASHBOARD_TITLE = ('Benchy-Graph | ')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate Benchy-Graph from CSV data.'
    )
    parser.add_argument('csv_path', help='Path to input CSV file')
    parser.add_argument('image_path', help='Path to output image file')
    return parser.parse_args()


def load_and_process_data(csv_path):
    """
    Load CSV data and extract phase, param, and concurrency information.

    Args:
        csv_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Processed dataframe with extracted columns

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    model_name = None
    if 'model' in df.columns and not df['model'].isna().all():
        model_name = str(df['model'].iloc[0]).strip()

    required_columns = ['test_name']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Extract phase, param, concurrency from test_name
    pattern = r'(pp|tg)\s*(\d+)\s*\(\s*c(\d+)\s*\)'
    extracted = df['test_name'].str.extract(pattern)
    df = df.assign(
        phase=extracted[0],
        param=pd.to_numeric(extracted[1], errors='coerce'),
        concurrency=pd.to_numeric(extracted[2], errors='coerce')
    )

    # Drop rows with missing phase or concurrency
    df = df.dropna(subset=['phase', 'concurrency']).copy()
    return df, model_name


def setup_plot_style():
    """Configure matplotlib and seaborn plot styles."""
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=PLOT_STYLE['font_scale'])
    rcparams = {k: v for k, v in PLOT_STYLE.items() if k != 'font_scale'}
    plt.rcParams.update(rcparams)


def plot_throughput_subplot(ax, df):
    """Plot average throughput and request throughput on the given axis."""
    unique_concurrency = sorted(df['concurrency'].dropna().unique())
    phase_offsets = {'pp': -0.08, 'tg': 0.08} if len(unique_concurrency) <= 1 else {'pp': 0.0, 'tg': 0.0}
    for phase, marker, color in [('pp', 'o', COLORS['pp']), ('tg', 's', COLORS['tg'])]:
        # Average throughput
        subset = df[(df['phase'] == phase) & (df['t_s_mean'].notna())]
        if not subset.empty:
            agg = subset.groupby('concurrency')['t_s_mean'].agg(['mean', 'std']).reset_index()
            x = agg['concurrency'] + phase_offsets[phase]
            ax.errorbar(x, agg['mean'], yerr=agg['std'],
                       marker=marker, label=f'{phase.upper()} avg speed',
                       color=color, capsize=4, linewidth=2, markersize=8)

        # Request throughput
        subset_req = df[(df['phase'] == phase) & (df['t_s_req_mean'].notna())]
        if not subset_req.empty:
            agg_req = subset_req.groupby('concurrency')['t_s_req_mean'].agg(['mean', 'std']).reset_index()
            x_req = agg_req['concurrency'] + phase_offsets[phase]
            ax.errorbar(x_req, agg_req['mean'], yerr=agg_req['std'],
                       marker='^', label=f'{phase.upper()} t/s/r',
                       color=COLORS[f'{phase}_tsr'], linestyle='--', capsize=4, linewidth=2, markersize=8)

    ax.set_xlabel('Concurrency (number of requests)', fontsize=12)
    ax.set_ylabel('Tokens/sec', fontsize=12)
    ax.set_title('Avg throughput and request throughput', fontsize=13, fontweight='bold')
    if unique_concurrency:
        ax.set_xticks(unique_concurrency)
        if len(unique_concurrency) == 1:
            x = unique_concurrency[0]
            ax.set_xlim(x - 0.5, x + 0.5)
        else:
            ax.set_xlim(min(unique_concurrency) - 0.5, max(unique_concurrency) + 0.5)
    ax.legend(frameon=True, fancybox=True)
    ax.grid(axis='y', alpha=0.4)


def plot_latency_subplot(ax, df):
    """Plot start latency (TTFR) for prefill phase on the given axis."""
    df_pp = df[df['phase'] == 'pp']

    if df_pp['ttfr_mean'].notna().any():
        agg_ttfr = df_pp.dropna(subset=['ttfr_mean']).groupby('concurrency')['ttfr_mean'].agg(['mean', 'std']).reset_index()
        ax.errorbar(agg_ttfr['concurrency'], agg_ttfr['mean'], yerr=agg_ttfr['std'],
                   marker='o', label='TTFR (Time To First Response)', color=COLORS['pp'], capsize=4)

    ax.set_xlabel('Concurrency', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Start Latency (Prefill phase)', fontsize=13, fontweight='bold')
    unique_concurrency = sorted(df_pp['concurrency'].dropna().unique())
    if unique_concurrency:
        ax.set_xticks(unique_concurrency)
        if len(unique_concurrency) == 1:
            x = unique_concurrency[0]
            ax.set_xlim(x - 0.5, x + 0.5)
        else:
            ax.set_xlim(min(unique_concurrency) - 0.5, max(unique_concurrency) + 0.5)
    ax.legend(frameon=True, fancybox=True)
    ax.grid(axis='y', alpha=0.4)


def plot_heatmap_subplot(ax, df):
    """Plot throughput heatmap on the given axis."""
    pivot_throughput = df.dropna(subset=['t_s_mean']).pivot_table(
        values='t_s_mean', index='phase', columns='concurrency', aggfunc='mean'
    )
    if not pivot_throughput.empty:
        sns.heatmap(pivot_throughput, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                   cbar_kws={'label': 'tokens/sec'}, linewidths=0.5)
        ax.set_title('Avg Throughput (tokens/sec)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Concurrency')
        ax.set_ylabel('Phase')


def create_summary_table(ax, df):
    """Create summary table on the given axis."""
    ax.axis('off')

    summary_rows = []
    for phase in ['pp', 'tg']:
        for conc in sorted(df['concurrency'].unique()):
            subset = df[(df['phase'] == phase) & (df['concurrency'] == conc)]
            if not subset.empty and subset['t_s_mean'].notna().any():
                row = {
                    'Phase': phase.upper(),
                    'Conc.': conc,
                    'Avg Speed': (f"{subset['t_s_mean'].mean():.1f} ± {subset['t_s_mean'].std():.1f}"
                                if subset['t_s_mean'].notna().any() else 'N/A'),
                    't/s/r': (f"{subset['t_s_req_mean'].mean():.1f} ± {subset['t_s_req_mean'].std():.1f}"
                            if subset['t_s_req_mean'].notna().any() else 'N/A'),
                    'TTFR': (f"{subset['ttfr_mean'].mean():.1f}ms"
                           if subset['ttfr_mean'].notna().any() else "N/A"),
                    'PPT': (f"{subset['est_ppt_mean'].mean():.2f}ms"
                          if subset['est_ppt_mean'].notna().any() else "N/A")
                }
                summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                        cellLoc='center', loc='center', colColours=[COLORS['pp']]*len(summary_df.columns))
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        for i in range(len(summary_df)):
            color = COLORS['pp_light'] if i % 2 == 0 else COLORS['tg_light']
            for j in range(len(summary_df.columns)):
                table[(i+1, j)].set_facecolor(color)
        ax.set_title('Summary', fontsize=13, fontweight='bold', pad=20)


def generate_dashboard(df, image_path, model_name=None):
    """
    Generate the complete dashboard plot and save to file.

    Args:
        df (pd.DataFrame): Processed dataframe
        image_path (str): Path to save the output image
    """
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    title = DASHBOARD_TITLE
    if model_name:
        title = f"{DASHBOARD_TITLE}{model_name}"
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)

    plot_throughput_subplot(axes[0, 0], df)
    plot_latency_subplot(axes[0, 1], df)
    plot_heatmap_subplot(axes[1, 0], df)
    create_summary_table(axes[1, 1], df)

    plt.tight_layout()
    plt.savefig(image_path, dpi=300, bbox_inches='tight', facecolor='white')
    if plt.get_backend().lower() != 'agg':
        plt.show()


def main():
    """Main entry point."""
    try:
        args = parse_arguments()
        df, model_name = load_and_process_data(args.csv_path)
        generate_dashboard(df, args.image_path, model_name=model_name)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()