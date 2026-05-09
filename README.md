# Benchy-Graph

Benchy-Graph is a tool that generates performance dashboards from llama-benchy CSV benchmark data. It visualizes key metrics like throughput, latency, and performance across different phases and concurrency levels for language model inference.

## Generating the CSV File

To generate the required CSV file, use [llama-benchy](https://github.com/eugr/llama-benchy), a benchmarking tool for llama.cpp servers.

Example command to generate the CSV:

```bash
uvx llama-benchy --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3.6-27B --served-model-name unsloth/Qwen3.6-27B-GGUF --concurrency 1 2 4 8 16 32 --pp 128 --tg 128 --format csv
```

This will produce a CSV file with benchmark results that can be used as input for Benchy-Graph.

## Running the App

To generate a performance dashboard image from a CSV file:

1. Ensure dependencies are installed: `pip install -r requirement.txt`
2. Run the script: `python app.py <input.csv> <output.png>`

Replace `<input.csv>` with the path to your llama-benchy CSV file and `<output.png>` with the desired output image path.

## Running the Notebook

For an interactive experience, open `notebook.ipynb` in Jupyter Notebook or JupyterLab and execute the cells. The notebook contains all the necessary code and explanations for generating visualizations.