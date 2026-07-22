# Post-on-MSB filopodia pipeline

This directory contains a cleaned and reorganized version of the "post-on-MSB"
filopodia workflow. It runs **downstream of** the MSB clustering pipeline: it
takes the pre-synaptic clusters produced by `MSB_pipeline` and identifies the
filopodia post-synapses that sit on those multi-synapse boutons (MSBs).

## What this package does

For every neuron with an SWC morphology file, the pipeline:

1. **First stage** – attaches the neuron's synapses to its healed SWC skeleton,
   locates the skeleton node nearest each MSB cluster centroid, and keeps the
   post-synapses that lie within a geodesic distance of a centroid (while
   discarding those that are closer to a non-clustered pre-synapse).
2. **Second stage** – builds a directed graph of the skeleton and connectors and
   retains the candidate filopodia post-synapses whose downstream sub-tree
   contains no pre-synapse, then assigns each surviving post-synapse a `base_node`
   (and a compact `base` index) along the path back to its cluster centroid.

The per-neuron results are concatenated and written to a single Feather file.

The main output is:

- a table of filopodia post-synapses annotated with their MSB cluster, centroid,
  base node, and source neuron
- The Feather file is written to
  `mixed-polarity-paper\MSB-and-Post-on-MSB-pipeline\data` as `post_on_MSB.ftr`

## Repository structure

- `pipeline.py` – main entry point (`run_pipeline` / `main`) and the two
  per-neuron stages (`process_first_part`, `process_second_part`)
- `pipeline_methods.py` – graph helpers (skeleton graph construction, the
  downstream pre-synapse test, and coordinate utilities)
- `config.py` – configurable input/output paths
- `requirements.txt` – Python dependencies needed to run the workflow
- `__main__.py` – enables `python -m Post_on_MSB_pipeline`

This package reuses `upload_swc` and `heal_attach` from the sibling
`MSB_pipeline` package, and reads the shared data files (SWC files, synapse
table, and the cluster-info output) from
`mixed-polarity-paper\MSB-and-Post-on-MSB-pipeline\data`.

## Inputs

Before running, make sure the following are present in the shared
`MSB-and-Post-on-MSB-pipeline/data` folder:

1. `swc/` containing the neurons' `.swc` files
2. `sample_synapses.ftr` – the synapse table for those neurons
3. `cluster_info_output.ftr` – the MSB clustering output produced by
   `MSB_pipeline`

> **Run `MSB_pipeline` first.** This pipeline consumes `cluster_info_output.ftr`;
> if it is missing, run `python -m MSB_pipeline` to generate it. See the
> `MSB_pipeline/README.md` for details on obtaining the full input dataset.

## Requirements

This workflow was developed and tested with Python 3.12+ and the following
libraries:

- `navis`
- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`
- `networkx`
- `pyarrow`

The easiest way to install them is with `pip`:

```bash
pip install -r requirements.txt
```

If you are using conda, you can also create an environment and install the same
requirements there:

```bash
conda create -n post-on-msb python=3.12
conda activate post-on-msb
pip install -r requirements.txt
```

## Running the pipeline

From the project folder (`MSB-and-Post-on-MSB-pipeline`), run:

```bash
cd <path-to-MSB-and-Post-on-MSB-pipeline>
python -m Post_on_MSB_pipeline
```

If you are using a conda environment, the same command can be run as:

```bash
conda activate project_env
cd <path-to-MSB-and-Post-on-MSB-pipeline>
python -m Post_on_MSB_pipeline
```

## Expected output

The pipeline writes an output file named:

```text
post_on_MSB.ftr
```

This file is saved in the `MSB-and-Post-on-MSB-pipeline/data` folder and contains
the filopodia post-synapses with their MSB cluster, centroid, base-node, and
neuron annotations.

## Using the package programmatically

```python
from Post_on_MSB_pipeline import Config, run_pipeline

df = run_pipeline(Config())
```

`run_pipeline` returns the combined result as a `pandas.DataFrame` without
writing to disk; `main()` (invoked by `python -m Post_on_MSB_pipeline`) wraps it
and saves the Feather file.

## Notes

- The workflow is designed to be run locally from the folder containing the SWC
  files and the synapse table.
- The input/output paths can be adjusted in `config.py` if needed for different
  datasets or directory layouts.
- Neurons with more than 80,000 skeleton nodes are skipped in the first stage,
  and any per-neuron error is logged and skipped so the remaining neurons still
  run.
