# MSB clustering pipeline

This directory contains a cleaned and reorganized version of the MSB clustering workflow for users who want to run it locally.

## What this package does

The pipeline loads a synapse table and a set of SWC morphology files, attaches synapses to the corresponding neuron skeletons, and clusters pre-synaptic connectors using a DBSCAN-based workflow on geodesic distances.

The main outputs are:

- a clustered synapse table containing neuron, synapse, and cluster information
- a Feather file written to the project folder as `cluster_info_output.ftr`

## Repository structure

- `pipeline.py` – main entry point for running the workflow
- `clustering.py` – synapse attachment and clustering logic
- `utils.py` – helper functions for reading SWC files and synapse tables
- `config.py` – configurable paths and clustering parameters
- `requirements.txt` – Python dependencies needed to run the workflow
- `swc/` – directory containing SWC morphology files
- `sample_synapses.ftr` – input synapse table in Feather format

## Requirements

This workflow was developed and tested with Python 3.10+ and the following libraries:

- `navis`
- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`
- `pyarrow`

The easiest way to install them is with `pip`:

```bash
pip install -r requirements.txt
```

If you are using conda, you can also create an environment and install the same requirements there:

```bash
conda create -n msb-pipeline python=3.10
conda activate msb-pipeline
pip install -r requirements.txt
```


## Input files

Before running the workflow, make sure the following files are present in this folder:

1. `swc/` containing 10 neurons' `.swc` files
2. `sample_synapses.ftr` containing the synapse table for those 10 neurons


## Running the pipeline

From the project folder, run:

```bash
cd <path-to-the-project-folder>
python -m MSB_pipeline
```

If you are using a conda environment, the same command can be run as:

```bash
conda activate project_env
cd <path-to-the-project-folder>
python -m MSB_pipeline
```


## Expected output

The pipeline writes an output file named:

```text
cluster_info_output.ftr
```

This file is saved in the MSB project folder and contains the clustered synapse information.

## Notes

- The workflow is designed to be run locally from the folder containing the SWC files and the synapse table.
- If a file or folder is missing, the pipeline will raise an error indicating which input is not found.
- The clustering parameters can be adjusted in `config.py` if needed for different datasets or thresholds.
