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

This workflow was developed and tested with Python 3.12+ and the following libraries:

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
conda create -n msb-pipeline python=3.12
conda activate msb-pipeline
pip install -r requirements.txt
```


## Demo Input files

Before running the workflow, make sure the following are present in `MSB-and-Post-on-MSB-pipeline` folder:

1. `swc/` containing 10 neurons' `.swc` files
2. `sample_synapses.ftr` containing the synapse table for those 10 neurons

## Full Input files

The full input files are available at: https://codex.flywire.ai/api/download?dataset=fafb.

1) `swc` files are under `Neuron Skeletons`
2) `synapses` are under `Synapse Table` but needs pre-processing:
    1. Add the 720575940 prefix for all rows in pre_root_id_720575940 and post_root_id_720575940 columns, and rename them to 'pre' & 'post'.
    2. Rename the index column to `synapse_id`.
    

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

This file is saved in the `MSB-and-Post-on-MSB-pipeline` folder and contains the clustered synapse information.

## Notes

- The workflow is designed to be run locally from the folder containing the SWC files and the synapse table.
- If a file or folder is missing, the pipeline will raise an error indicating which input is not found.
- The clustering parameters can be adjusted in `config.py` if needed for different datasets or thresholds.
