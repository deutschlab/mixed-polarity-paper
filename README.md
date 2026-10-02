# Organizational principles governing synapse types in a whole-brain connectome

Code for the manuscript *Organizational principles governing synapse types in a whole-brain connectome* by Amit Gross, Majd Farah and Dr David Deutsch.

## About

The analysis uses the FlyWire FAFB v783 connectome of the adult fruit fly brain, with one SWC skeleton per neuron and the Princeton synapse table.

Each neuron's skeleton is split into an axon and a dendrite, sometimes with a short linker between them. We then measure how well the neuron keeps its inputs and outputs apart on those two sides. That is the segregation index (SI), which runs from 0 (mixed) to 1 (fully separated). Each synapse gets a two-letter type, the sending neuron's compartment followed by the receiving neuron's: AD, AA, DD or DA. The rest of the code builds tables from these labels and makes the figures in the paper, which look at synapse types across neuron classes, neurotransmitters, neuron shape, brain regions and reciprocal connections.

The folder [MSB-and-Post-on-MSB-pipeline/](MSB-and-Post-on-MSB-pipeline/) holds two separate pipelines. `MSB_pipeline` groups presynapses into multi-synapse boutons (MSBs) using DBSCAN on distances along the skeleton, and `Post_on_MSB_pipeline`, which runs after it, finds the postsynapses on spines coming off those boutons.

## What you can reproduce

The demo runs without downloading anything, since its 200 neurons, their synapses and skeletons are in `demo/`. The two MSB pipelines also come with their own sample data and READMEs.

The full pipeline cannot be run from public data alone. With the Codex downloads you can run steps 01 to 04, but step 05, which builds the per-neuron table, also needs a file called `swc_data.ftr` that no script here makes. Steps 06 to 08, the Phi-cutoff script, the two synapse-type tools and almost all figure scripts need step 05's table, so from the downloads alone none of them can run.

Six tables that the scripts read are not made anywhere in this repository. They are listed in [docs/data_availability.md](docs/data_availability.md), and the scripts that need them will not run until someone provides them. Three of them come from the earlier Buhmann synapse detection. The pipeline here is written for the Princeton synapse table; the Buhmann section of docs/data_availability.md says what would have to change and which panels need those tables. The larval curve in Figure 2A can't be rebuilt either: it needs raw files from the corresponding author, and the larva script currently gives no usable output (see [docs/pipeline_overview.md](docs/pipeline_overview.md)).

The derived tables in `data/derived/` are not included.

## Quick-start demo

The notebook `demo/demo.ipynb` runs the core steps on 200 sample neurons: it splits each neuron, labels each synapse and computes SI.

To run it on Colab, open it here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deutschlab/mixed-polarity-paper/blob/main/demo/demo.ipynb). Note that Colab installs the latest navis rather than the pinned `navis==1.7.0`, so the SI values there can come out slightly different.

To run it locally, set up the environment first (see [Setup](#setup)), install Jupyter (`pip install notebook`, it isn't in the pinned requirements), then run `cd demo && jupyter notebook demo.ipynb`. It took 56 seconds on a laptop. It writes two files: `demo/output/SI_results_demo.csv`, with the SI of every neuron whose split worked (all 200 in our run), and `demo/output/connectors_demo.csv`, with the compartment (A, D or L) of every synapse.

The demo only covers the SI steps, not the MSB pipelines.

## Setup

You need Python 3.12.4 on Linux, macOS or Windows. Other Python versions may not work with the pinned packages. For the full pipeline we recommend 32 GB of memory, because the synapse table is large and memory use goes above 16 GB. The optional `processing/build_raw_synapse_table.py` needs more: it peaked at about 40 GB (22 GB resident) on a 36 GB Mac, and writes about 7 GB. No special hardware is needed.

All packages are pinned in `requirements.txt` and `environment.yml`, including `navis==1.7.0`, `pandas==2.2.2`, `numpy==1.26.4`, `scikit-learn==1.5.2`, `scipy==1.13.1`, `matplotlib==3.9.2` and `seaborn==0.13.2`. Keep navis at 1.7.0, since other versions can give different SI values. Installing takes about 5 to 10 minutes with conda, or 10 to 20 with pip.

```bash
# conda (recommended)
conda env create -f environment.yml
conda activate fafb783-synapse-polarity

# or pip, in a new virtual environment
python3.12 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On macOS with pip, `xgboost` also needs `brew install libomp`. Conda takes care of this.

### Data

Most of the data is not in this repository. Two inputs are included in `data/raw/`: the neuron annotation table (`Supplemental_file1_neuron_annotations.csv`) and the 15 sensory-rank files in `data/raw/ranks/`, both from Dorkenwald et al. 2024. The connectome files come from the [FlyWire Codex download page](https://codex.flywire.ai/api/download?dataset=fafb). [data/README.md](data/README.md) and [docs/data_availability.md](docs/data_availability.md) list every file, where to get it and where to put it.

Skeletons go in subfolders of `data/raw/swc/783/`, one file per neuron named `<root_id>.swc`. Any number of subfolders is fine (we used batches of about 1,500 neurons), but files placed directly in `data/raw/swc/783/` won't be found. If the archive unpacks into one flat folder, make a subfolder and move the files into it. Step 01 gives no error when it finds no skeletons; it just writes nothing.

### Paths

All paths are set in `config.py`, and `ROOT` finds the repository folder on its own. You only need to edit `config.py` if you keep a file in a different place or under a different name.

## Reproducing the analysis

Run the steps in this order, from the repository root:

```bash
python processing/01_extract_compartments_SI.py        # takes hours; compartment labels and SI per neuron
python processing/02a_large_neurons_pipeline.py        # the very large neurons, in a separate pass
python processing/02b_merge_connectors.py
python processing/03_merge_synapses_connectors.py
python processing/04_build_master_synapse_table.py     # -> data/derived/synapses_783_article_princeton.ftr
python processing/05_build_neuron_metadata_table.py    # -> data/derived/neuron_data_full_article_princeton.ftr (needs swc_data.ftr)
python processing/06_build_connection_reciprocity_table.py
python processing/07_pca_morphology.py                 # -> data/derived/neurons_pca_princeton.ftr
python processing/build_raw_synapse_table.py           # optional, before 08 -> the raw synapse table (two versions)
python processing/08_alternative_split_methods.py      # optional -> data/derived/SI_comparisons.ftr
python processing/phi_threshold.py                     # optional, after 05 and 08
```

Step 05 reads `swc_data.ftr`, which no script here writes. The code that used to write it is still inside step 05, but it is commented out and points to a local Windows path, so from the downloads alone the pipeline stops at step 05.

The work is split into steps mostly to keep memory use down, since the synapse table and all the skeletons don't fit in memory together. Step 01 skips neurons with more than 80,000 skeleton points. It also treats any skeleton subfolder that already has an output folder as done, and it makes that folder before it starts, so if a run of step 01 is interrupted, delete the unfinished folders in `data/intermediate/processed_swc_data/` before running it again. Step 02a processes those large neurons, but it works from a list of neuron IDs written into the script, not from step 01's size check.

Step 08 is optional. It compares the published axon/dendrite split with two single-node cuts, MaxSI and MinFisherP. It needs step 05's table, the skeletons of the intrinsic neurons, and `synapses_783_article_princeton_raw.ftr`, which `processing/build_raw_synapse_table.py` builds from the Codex CSV (run it first). That file must have `size` and `neuropil` columns, so the synapse table from step 04 can't be used in its place. The build writes two versions, without and with self-synapses; `RAW_TABLE_KEEP_SELF_SYNAPSES` in `config.py` chooses which one the scripts read.

`phi_threshold.py` is also optional. It works out which Phi value matches the SI = 0.1 cutoff, using step 08's output plus the neuron table and `SI_updated.ftr` from step 05. It prints the results and saves a plot to `outputs/phi_threshold/phi_threshold.svg`.

The larval SI for Figure 2A comes from `processing/larva/larva_process.py`, which runs separately from these steps. As committed it gives no usable output (see above).

[docs/pipeline_overview.md](docs/pipeline_overview.md) describes each step with its inputs and outputs.

### Figures

The figure scripts are in `figures/fig1/` to `figures/fig5/` and `figures/fig7/`, usually one script per panel; `figures/split_methods/` and `figures/supporting/` hold scripts for the split-method comparison and for numbers in the text. They are written as `#%%` cell scripts for an editor like Spyder, but you can also run them as normal Python scripts from any folder. If you do, each plot opens in a window and the script waits until you close it; `tools/run_figure.py` gets around this.

Fig 1C and Fig 3C make a Neuroglancer link through `caveclient` before they draw the panel, so they stop with an error unless you have a FlyWire account and a CAVE token saved on your machine (see the caveclient documentation). The `caveclient` and `nglui` packages are imported by `methods/methods_all.py`, so every script that uses it needs them installed (both are in the pinned requirements); only these two panels need the token.

Three scripts need others to run first: `figures/fig4/build_pc1_table.py` before `figures/fig4/syntype_x_pc1.py`, `figures/fig5/reciprocal_fraction.py` before `figures/fig5/reciprocal_fraction_model.py`, and the three `figures/fig7/si_sim_*.py` scripts before `figures/fig7/si_comb_analysis.py`.

Figures are saved in `outputs/figN/`, almost all as SVG and a few as PNG or PDF. A few figure scripts also write tables into `data/derived/`. [docs/figure_to_script_map.md](docs/figure_to_script_map.md) shows which script makes which panel.

A note on SI: step 04 stores each neuron's SI in the synapse table as `SI_pre` and `SI_post`, and step 05 stores it in the neuron table and in `SI_updated.ftr`. All three come from the same output of steps 01 and 02a, so after a fresh run they are identical. Tables from an earlier run may not match, which is why the synapse-type tools refer to the table SI as "older" and to `SI_updated.ftr` as "corrected". Most figure scripts take SI from the neuron table or the synapse table, and six take it from `SI_updated.ftr`.

## Tools

Run these from the repository root with the environment active.

### Checking that a figure reproduces

`tools/run_figure.py` runs a figure script in a way that makes two runs easy to compare. No plot windows open, and all figures go into one output folder, which has to be new or empty. SVG files are written without a date, random numbers are fixed (including the ones seaborn uses for error bands), and every printed table is followed by a checksum of its contents, so a change in a hidden row or decimal still shows up. The file `run_status.txt` says whether the script finished. Figures are saved at 72 dpi by default, the same as the published SVGs; `--dpi` changes this. With `--replace OLD NEW` you can swap an exact piece of text in the script before it runs, for example to skip a very slow step, but a run changed that way is no longer a full reproduction.

`tools/compare_outputs.py` then compares two output folders, matching files by name. An SVG passes if it is identical, or identical once element ids and metadata are ignored, and any other file has to be identical. The printed output in `stdout.log` has to match too; with `--allow-extra-lines` it is enough that every line of the first run appears, in order, in the second. `--figures-only` compares figure files only (.svg, .png, .pdf), which is useful against a folder of published SVGs. `stderr.log` is never compared. The script exits with status 1 if anything differs, if a run didn't finish, or if neither folder has a figure; use `--allow-no-figures` for a script that only prints or writes tables, such as `figures/fig4/build_pc1_table.py`. With `--content`, an SVG whose lines still differ passes if it draws the same text and colours, in the same order, and, inside each plot, the same markers and shapes once the plot's size and position are removed. That is how to check a port against an original saved with a different figure size or font. Positions are compared to 0.1% of the plot, line widths and font sizes are not compared, and an SVG whose text was saved as outlines fails.

```bash
python tools/run_figure.py figures/fig1/canonicality_axon_dend.py --out outputs/check/before
# make your change, then run again into a new folder
python tools/run_figure.py figures/fig1/canonicality_axon_dend.py --out outputs/check/after
python tools/compare_outputs.py outputs/check/before outputs/check/after
```

A check run still writes any tables the figure script writes, in their usual place, replacing what is there. Since all figures land in one folder, two figures saved under the same file name in different subfolders will overwrite each other.

To check every figure at once, `tools/run_all_figures.py` runs each script under `figures/` through `run_figure.py`, one after another, each into its own subfolder of `--out`. Scripts that others read from run first (`build_pc1_table.py`, `reciprocal_fraction.py` and the three Fig 7 `si_sim_*.py` scripts). Each script has a time limit, 20 minutes or 3 hours for the few slow ones, and one that runs out of time is stopped with any worker processes it started. `progress.txt` gets one line per script and a summary; `commit.txt` the git commit, and `inputs.txt` the size and date of every file under `data/raw` and `data/derived`, so a difference between two runs can be traced to changed input data. Running it again with the same `--out` continues an interrupted run: scripts that finished or stopped with an error are skipped, the others run again (`--rerun-failed` also reruns the ones that stopped). `--only fig4` runs a subset. It exits with status 1 if any script did not finish, which on this repository includes every script whose input tables are not supplied. A full run takes several hours, and scripts that write tables still replace them in their usual place (for example the 3 GB `data/derived/PC1_table.csv`).

`tools/compare_runs.py` then compares two such runs script by script: the run status (and, for a script that stopped, the last line of its error), every saved figure (as `compare_outputs.py` does) and the printed output. It reports the scripts that are the same and finished separately from those that are the same but did not finish. `--mask` replaces text that changes on every run, such as the date and time in statsmodels summaries, before the printed outputs are compared, so the rest of those lines still counts. It exits with status 1 if any script differs or is in only one run.

```bash
python tools/run_all_figures.py --out outputs/runs/before
# make your change, then run again into a new folder
python tools/run_all_figures.py --out outputs/runs/after
python tools/compare_runs.py outputs/runs/before outputs/runs/after \
    --mask "[A-Z][a-z]{2}, \d{2} [A-Z][a-z]{2} \d{4}" --mask "\d{2}:\d{2}:\d{2}"
```

`tools/check_raw_synapse_table.py` checks the raw synapse tables that `processing/build_raw_synapse_table.py` writes: the columns and their types, `synapse_id` in order and unique, no self-synapses in the version without them, both versions differing by exactly the self-synapses, and every synapse of the processed synapse table present with the same pre and post cell. It also compares the six coordinates with the processed table, which catches a pre/post column swap. It needs the step 04 synapse table and about 10 GB of free memory; `--csv` also checks the row count against the Codex CSV. It takes one to two minutes.

### Synapse-type tools

Both tools read the synapse and neuron tables from `data/derived/`, so they need step 05's output (and `SI_updated.ftr` if you use `--si corrected`). "Intrinsic" neurons here means super-class `central`, `optic`, `visual_projection` or `visual_centrifugal`, with `axon_correct`, `dend_correct` and `primary_type` filled in, the same set the figure scripts use. The SI filter keeps synapses where `SI_pre` and `SI_post` are both at least 0.1; `--si corrected` takes SI from `SI_updated.ftr` instead.

`tools/synapse_type_shares.py` prints the share of each synapse type (AD, AA, DD, DA), for all synapses and for intrinsic neurons, with and without the SI filter. Synapses with a linker end are left out and counted separately. It needs about 4 GB of memory, or about 5 GB with `--si corrected`.

`tools/split_agreement.py` compares the types that two or more splits give to the same synapses. For each pair of splits, and for four groups of synapses (all of them, those passing the SI filter, intrinsic neurons, and intrinsic neurons passing the filter), it reports how many synapses have a type in both, the agreement, the agreement expected by chance, Cohen's kappa, the agreement for each type and the 4 x 4 table of counts. After that it gives the agreement by SI band and a per-neuron summary. `--bootstrap N` adds 95% intervals by resampling whole cell types (`--cluster neuron` resamples single neurons instead, and `--seed` sets the seed); use at least 1000 replicates. The intervals cover the agreement and kappa only. It needs about 12 GB of memory.

The SI filter and the SI bands use one SI for every split compared, by default `SI_pre` and `SI_post`, which come from the published split. So they pick out neurons that the published split separates well, and agreement with the published split will look higher there. Keep that in mind when you compare splits on the filtered synapses.

```bash
python tools/synapse_type_shares.py
python tools/synapse_type_shares.py --si corrected
```

Both tools accept `--table` to point at another synapse table with the same columns. `synapse_type_shares.py --column` counts a different column of synapse types, and `split_agreement.py` takes the columns to compare as arguments. They were written for a synapse table with one type column per split method, like the one the alternative-split pipeline produces. Step 08 only scores the splits per neuron; the pipeline that labels each synapse under each split hasn't been added to the repository yet, so for now no table here has more than one type column.

### Checking the split methods

`tools/check_split_methods_on_subset.py` runs step 08 on some of the neurons, by default every intrinsic neuron whose skeleton is under `SWC_DIR`, so the split methods can be checked without the full skeleton download. It needs the raw synapse table (`processing/build_raw_synapse_table.py`). It does not keep its own copy of step 08: it patches five places in `processing/08_alternative_split_methods.py` (the repository path, the two output paths, the neuron list, the synapse read and the batch range of the combine step) and stops if any of them is not found exactly once. Everything is written to the `--out` folder, which has to be new or empty, including the patched copy and a `provenance.txt` with the commit, the step 08 checksum and the pandas and numpy versions. `--ids FILE` and `--limit N` choose other neurons, and `--resume` continues an interrupted run, refusing if step 08 or the list of neurons has changed.

Afterwards it writes `per_neuron.csv`. This holds step 08's four scores for each neuron, whether step 08 listed the neuron as failed, and two values computed from the per-node tables: the MinFisherP Phi under a second tie rule (the first node in table order with the lowest p) and MaxPhi (the highest |Phi| over the candidate nodes, a fourth split method that step 08 does not compute). Ties for the lowest p are common, but usually between nodes that give the same cut; the two rules give a different Phi only when the tied nodes give different cuts, which on the 1,144 neurons tested happened in 37, all where the p-value underflowed to 0 or to the smallest float. It also prints how many neurons have tied nodes and how many have MaxSI below the SFC SI. With `--compare REF.ftr`, each score is compared, neuron by neuron, with a reference table that has a `neuron_id` column (`original_SI` is read as `SFC_SI`). A neuron counts as a problem if a score differs by more than `--tol` (default 1e-9), is empty in one table only, is missing from the reference, or failed in the run; a reference with no score column in common fails. A failing `MaxPhi` means the tool's own computation disagrees with the reference, not step 08. The tool prints PASS or FAIL for each reference, lists the problem neurons in `differences_*.csv`, and exits with status 1 if any reference fails. `--compare-only` repeats the analysis on a finished run. Without `--compare` nothing is checked and the exit status is 0. A PASS covers only the neurons run; against a table made by the same step 08 code it shows that the result does not depend on which neurons are run together, not that the method is right. `other_summary.ftr` below stands for any table with a `neuron_id` column.

```bash
python tools/check_split_methods_on_subset.py --out outputs/split_methods_check/run1 \
    --compare data/derived/SI_comparisons.ftr
python tools/check_split_methods_on_subset.py --out outputs/split_methods_check/run1 --compare-only \
    --compare data/derived/SI_comparisons.ftr other_summary.ftr
```

On 1,144 intrinsic neurons (macOS, 2 Oct 2026) it took 28 minutes, with a peak memory footprint of 24.7 GB, almost all of it step 08's own work on a batch of 500 neurons; 20 neurons took 13 seconds and 1.0 GB. Use `--limit` on a smaller machine.

### Checking the simple PC1 model

`tools/check_simple_pc1_model.py` rebuilds the model of `figures/fig4/syntype_x_pc1_simple_model.py` (a random forest that predicts a synapse's type from PC1 of its two cells) in its own code, following the script's filters, balancing, split and settings, and prints the counts, the test accuracy, the confusion matrix and the feature importances. With `--run DIR`, a `tools/run_figure.py` run of the figure script, it first checks that the run finished (and warns if the synapse or neuron table, from step 05, or the PCA table, from step 07, is newer than the run), then compares: the accuracy in `stdout.log` exactly, the 16 confusion-matrix percentages in the SVG to 0.005 (after checking the axis labels), and the two feature importances measured from the bar lengths in the SVG, to `--tol-importance` (default 0.001), after checking they sum to 1. It prints PASS or FAIL and exits with status 1 on FAIL. A PASS means the script computes what its code says on the same tables; it does not test the tables or whether the model is statistically sound (rows are resampled before the split, and the split is by synapse).

`--variants` repeats the rebuild with one change each (balancing seed 7, PC1 sign flipped, synapses of cells without PC1 left out) and prints how much the accuracy and each confusion-matrix cell move; three runs give only a rough idea of the noise. `--earlier DIR` (with `--run`) compares the run with an earlier run's two SVGs, for information only; the earlier accuracy is the mean of its confusion-matrix diagonal. `--report FILE` also writes everything printed to a file.

```bash
python tools/run_figure.py figures/fig4/syntype_x_pc1_simple_model.py --out outputs/check/simple_model
python tools/check_simple_pc1_model.py --run outputs/check/simple_model --variants
```

The `--out` folder of `run_figure.py` must be new or empty.

It reads only the five needed columns of the synapse table, batch by batch, and needs about 7 GB of memory; with `--variants` it took about 8 minutes (macOS, 2 Oct 2026). The result is exact only with the pinned library versions.

## Main generated tables

The pipeline writes these to `data/derived/`.

| File | Contents |
|------|----------|
| `synapses_783_article_princeton.ftr` | One row per synapse between two neurons that were both split: the two neurons, positions, brain region (`npil`), synapse type (`comp`) and `SI_pre`/`SI_post`. `comp` can take nine values, because a linker end gives types like AL or LD, so filter to AA/AD/DA/DD. |
| `neuron_data_full_article_princeton.ftr` | One row per neuron that could be split: type, SI, shape measurements and connection counts |
| `SI_updated.ftr` | Each neuron's SI (`root_id`, `SI`) |
| `connections_by_syn_type_reciprocal_types_filtered_article_princeton.ftr` | One row per connection: synapse counts per type, and whether the pair is reciprocal |
| `neurons_pca_princeton.ftr` | Each neuron's PC1 and PC2 |

The full list, with columns, is in [docs/generated_tables.md](docs/generated_tables.md).

## Documentation

- [docs/data_availability.md](docs/data_availability.md): every input file, where it comes from and where it goes
- [docs/pipeline_overview.md](docs/pipeline_overview.md): each processing step, with inputs and outputs
- [docs/figure_to_script_map.md](docs/figure_to_script_map.md): which script makes which figure panel, and from which tables
- [docs/generated_tables.md](docs/generated_tables.md): every derived table, which script writes it and its columns
- [data/README.md](data/README.md): the data folder and where each file goes

## Repository layout

```
config.py           all file paths
requirements.txt    pinned packages (pip)
environment.yml     the same packages for conda

processing/         the pipeline, from raw data to derived tables
  01_extract_compartments_SI.py            split each neuron, label synapses, compute SI
  02a_large_neurons_pipeline.py            the same for very large neurons
  02b_merge_connectors.py                  collect step 01's synapse labels into two tables
  03_merge_synapses_connectors.py          add the labels back to the synapse list
  04_build_master_synapse_table.py         the per-synapse table
  05_build_neuron_metadata_table.py        the per-neuron table
  06_build_connection_reciprocity_table.py the per-connection and reciprocity tables
  07_pca_morphology.py                     PCA of the shape measurements
  build_raw_synapse_table.py               optional: the raw synapse table from the Codex CSV (for 08 and the split-method figures)
  08_alternative_split_methods.py          optional: two single-node cuts compared with the published split
  phi_threshold.py                         optional: the Phi cutoff that matches SI = 0.1
  larva/larva_process.py                   larval SI for Figure 2A

methods/
  methods_all.py    shared functions, used by most scripts

figures/            one folder per figure (fig1 to fig5, and fig7), plus split_methods/ and supporting/

tools/
  run_figure.py            run a figure script with fixed, comparable output
  compare_outputs.py       compare the output folders of two runs
  run_all_figures.py       run every figure script, one after another
  compare_runs.py          compare two full runs script by script
  check_raw_synapse_table.py  check the two raw synapse tables after building them
  synapse_type_shares.py   share of each synapse type in a synapse table
  split_agreement.py       agreement of synapse types between splits
  check_split_methods_on_subset.py  run step 08 on some neurons and compare with a reference table
  check_simple_pc1_model.py  rebuild the Fig 4 simple PC1 model and check a run of its script

demo/               the demo notebook and its 200-neuron data set

data/               mostly empty; the data goes here (see data/README.md)
  raw/              downloaded inputs, plus the included annotation table and ranks/
  derived/          tables made by the pipeline
  intermediate/     working files made by the pipeline
  larva/            larval data, from the corresponding author

docs/               the reference documents listed above
outputs/            figures made by the scripts (not included)

MSB-and-Post-on-MSB-pipeline/   the two MSB pipelines, each with its own README and sample data
```

## Citation

Please cite the manuscript. [CITATION.cff](CITATION.cff) has the citation for this code (DOI 10.64898/2026.06.23.733969).

## License

MIT, see [LICENSE](LICENSE). The two MSB pipelines have their own LICENSE files.
