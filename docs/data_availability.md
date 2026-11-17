# Data Availability

Every input the code reads: where to get it and where to put it.

Three inputs are in this repository, in `data/raw/`: the neuron annotation table
(`Supplemental_file1_neuron_annotations.csv`), the 15 sensory-rank files in `ranks/`, and the list of cell types left
out of the Fig 7 bouton analysis (`msb_excluded_cell_types.csv`). Most other raw files are downloads (below). The
pipeline makes most tables itself ([pipeline_overview.md](pipeline_overview.md)); the ones it can't make are listed
under "Inputs read by the code but not written by any script" below.

**Which setting you can use.** With only public data, run the pipeline under `PIPELINE_RUN = "rebuild"`. Its tables
go to `data/derived_rebuild/` (and `data/derived_rebuild_maxsi/` for the MaxSI split, which the main figures use), and
the figure scripts read them from there. The default, `"published"`, reads the authors' delivered tables in
`data/derived/`, which are available from the corresponding authors on request. The co-authors' extra tables (the
Fig 7 tables, the revised Fig 4E per-type table, the Buhmann tables) work only under `"published"`: under
`"rebuild"` the scripts look for them in `data/derived_rebuild/`, where nothing writes them, so those scripts stop
instead of mixing delivered and rebuilt tables. The exceptions are `neuron_data.ftr` and the two per-synapse flag
tables, which are read from where you put them under both settings, and the two bouton tables and two Buhmann tables,
which have MaxSI versions built under `"rebuild"` + `"maxsi"` (below).

A fresh run leaves a cell's synapses onto itself out of its SI, as the Methods do. The delivered neuron and synapse
tables carry an older SI that kept them (see `generated_tables.md`), so a fresh run's SI values differ from the
delivered ones.

---

## Public FlyWire / Codex inputs

From the FlyWire Codex download page:

> https://codex.flywire.ai/api/download?dataset=fafb

The page offers several datasets; this project uses FAFB v783 only. Files from another dataset can have the same
column names and load without an error, so check the dataset before you download. At the time of writing the site
asks you to sign in with a Google account and accept the FlyWire terms of service first.

Decompress the `.csv.gz` files (all but `consolidated_cell_types.csv.gz`, see its row) and put them at the paths
below, relative to the repository root. If a file name differs, change `config.py`.

| File | Local path | Config variable | Description |
|------|-----------|----------------|-------------|
| Princeton synapse table | `data/raw/fafb_v783_princeton_synapse_table.csv` | `PRINCETON_SYNAPSE_CSV` | All synapses of FAFB v783, about 7 GB. Columns used: `pre_root_id_720575940`, `post_root_id_720575940`, `pre_x/y/z`, `post_x/y/z`, `size`, `neuropil`. Rename the decompressed file to this name, or change `PRINCETON_SYNAPSE_CSV`. |
| Consolidated cell types | `data/raw/consolidated_cell_types.csv.gz` | `CONSOLIDATED_CELL_TYPES_CSV` | Keep it compressed. Read by the Supp 7-S2 scripts in `figures/fig7/` and by `processing/bouton_maxsi_tables.py`. Columns used: `root_id`, `primary_type`. |
| Neurotransmitter predictions | `data/raw/neurons.csv` | `NEURONS_CSV` | Per-neuron neurotransmitter predictions. Columns used: `root_id`, `nt_type`. Download `neurons.csv.gz` from the Neurotransmitter Type Predictions section. |
| Cell size statistics | `data/raw/cell_stats.csv` | `CELL_STATS_CSV` | Per-neuron size measurements: `root_id`, `length_nm`, `area_nm`, `size_nm`. Download `cell_stats.csv.gz` from the Cell Size Measurements section. |
| Neuron classification | `data/raw/classification.csv` | `CLASSIFICATION_CSV` | Per-neuron labels: `root_id`, `flow`, `super_class`, `class`, `sub_class`, `hemilineage`, `side`, `nerve`. Download `classification.csv.gz` from the Classification section. Read by `figures/fig3/syntype_x_identity.py`, `figures/fig5/syn_type_x_reci_x_dominance_x_sides.py`, `figures/fig6/aadd_specific_circuits.py` and `processing/bouton_maxsi_tables.py` (`super_class`). |
| SWC skeletons | `data/raw/swc/783/` | `SWC_DIR` | One skeleton file per neuron, about 31 GB. Download `sk_lod1_783_healed.zip` from the Neuron Skeletons section and put the `.swc` files in subfolders of `data/raw/swc/783/` (one per batch, for example about 1,500 files each), each named `<root_id>.swc`. Steps 01, 02a, 08, 09 and `build_swc_data.py` read only those subfolders. Files placed directly in `SWC_DIR` are not read: step 01 then ends with no output and no error, and `build_swc_data.py` stops (as it does on any stray file there). Keep the zip outside `SWC_DIR`. |

## Included in this repository

Small files in `data/raw/`: two from Dorkenwald et al. 2024, and the authors' list of the cell types left out of the
Fig 7 bouton analysis.

| File | Local path | Config variable | Description |
|------|-----------|----------------|-------------|
| `Supplemental_file1_neuron_annotations.csv` | `data/raw/Supplemental_file1_neuron_annotations.csv` | `NEURON_ANNOTATIONS_CSV` | The neuron annotation table of Dorkenwald et al. 2024 (citation below), 31 columns. Step 05 keeps `root_id`, `super_class`, `top_nt` and `cell_type`; four scripts also read `side`: `figures/fig2/si_x_primary_types_mirror.py`, `figures/split_methods/maxsi_x_primary_types_mirror.py`, `figures/fig7/filopodia_x_mirror_neurons.py` and `figures/split_methods/four_methods_si_comparison.py`. Read by `processing/05_build_neuron_metadata_table.py`, `figures/fig2/si_x_primary_types_mirror.py`, `figures/fig3/si_x_neuron_types.py`, `figures/fig7/filopodia_x_mirror_neurons.py`, `figures/split_methods/maxsi_x_primary_types_mirror.py` and `figures/split_methods/four_methods_si_comparison.py`. |
| Sensory-rank files | `data/raw/ranks/` | `RANKS_DIR` | One feather file per input modality (15 files): `node`, `layer_mean`, `root_id` (uint64). Step 05 names each kept file's `layer_mean` after its modality (`olfactory_layer_mean`, …). From Dorkenwald et al. 2024. Read by `processing/05_build_neuron_metadata_table.py`. |
| `msb_excluded_cell_types.csv` | `data/raw/msb_excluded_cell_types.csv` | `MSB_EXCLUDED_TYPES_CSV` | The cell types left out of the Fig 7 MSB (bouton) analysis, one row per type: `primary_type`; `exclusion_kind`; `reason` (the authors' note, empty for most rows); `n_cells_sheet` (the authors' count). The three kinds: `pre_clustering`, whole types left out before the boutons were clustered; `uniglomerular`, also left out before clustering, but cell by cell, by `sub_class`; `rms_inspection`, whole types left out after clustering, by inspection, which the Methods describe as starting from the neurons with a mean RMS distance above 850 nm between presynapses and their bouton's centre. 141 rows: 32 `pre_clustering`, 38 `rms_inspection`, 71 `uniglomerular`. `n_cells_sheet` sums to 53,561, the number of neurons the Methods give as removed; for `uniglomerular` rows it counts only that type's uniglomerular cells. It is not recomputed. On the v783 cell types the same rule removes 53,630 neurons, because the sheet gives R1-6 8,456 cells and `consolidated_cell_types.csv.gz` gives 8,525. Every cell of a `pre_clustering` or `rms_inspection` type is left out; of a `uniglomerular` type, only the cells whose own `sub_class` in `neuron_data.ftr` is `uniglomerular`, so `M_lvPNm46` and `M_vPNml80` keep some cells in the bouton tables. `tools/check_msb_exclusion.py` checks the list against the two bouton tables. No figure script reads it: the bouton tables already leave these cells out, and `fig7s2b` colours the types from its own copy of the same lists. |

## From the corresponding authors

Not in any public database; ask the corresponding authors.

| File | Local path | Config variable | Description |
|------|-----------|----------------|-------------|
| Neuropil synapse table | `data/raw/Cloud_SQL_fw_mat783_synapses_v3_neuropil.csv` | `NEUROPIL_SYNAPSE_CSV` | Synapses with neuropil labels from the Princeton synapse detection (Yu et al. 2025, https://doi.org/10.1101/2025.07.11.664377), with every partner kept, including segments not connected to any proofread neuron ("twigs", as in Dorkenwald et al. 2024): 192,210,404 rows, no header row, 14 columns (a synapse ID, the presynaptic point, a third point, the postsynaptic point, size, the presynaptic and postsynaptic root IDs as full 18-digit IDs, the neuropil). Its rows between two proofread neurons match the Princeton synapses (row by row for the 31,819 synapses of 30 sampled cells; for 10,000 cells, each cell's MaxSI split on them is the paper's). Read only by `figures/fig2/si_x_twigs.py` (Fig 2 Supp 1A). It is about 22.8 GB; the script reads it in blocks and needs about 7 GB of memory. Ask the corresponding authors for the copy used in the paper; the authors of Yu et al. 2025 may also have it. |

> Yu SC, et al. New synapse detection in the whole-brain connectome of *Drosophila*. bioRxiv (2025). https://doi.org/10.1101/2025.07.11.664377

The larval curve of Figure 2A needs three raw files from the larval connectome of Winding et al. 2023, also from the
corresponding authors:

> Winding M, Pedigo BD, Barnes CL, et al. The connectome of an insect brain.
> *Science* 379(6636), eadd9330 (2023). https://doi.org/10.1126/science.add9330

Put them in `data/larva/raw/`. `processing/larva/larva_maxsi_labels.py` builds the larval tables on the MaxSI split
from them, which Figure 2A reads under `SPLIT_METHOD = "maxsi"`, and `processing/larva/larva_pc1.py` runs on its
output. `processing/larva/larva_process.py`, the larval step for the SFC split, reads the same files but gives no
usable output as committed (see `docs/pipeline_overview.md`).

| File | Local path | Config variable | Description |
|------|-----------|----------------|-------------|
| Larval synapse table | `data/larva/raw/synapse_table.csv` | `LARVA_SYNAPSE_TABLE_CSV` | Columns, named by the script: `rowid`, `pre_skeleton_node_id`, `connector_id`, `pre_x/y/z`, `connector_x/y/z`, `post_skeleton_node_id`, `post_x/y/z`, `pre`, `post`. |
| Neuron list | `data/larva/raw/n_list3.csv` | `LARVA_NEURON_LIST_CSV` | The larval neuron IDs to include, in column `neuron`; rows with `-1` are skipped. |
| Larval skeletons | `data/larva/raw/swc/all_neurons/` | `LARVA_SWC_DIR` | One file per neuron, `<neuron>.swc`, comma-separated, with the header row `id,type,x,y,z,r,parent_id`. `larva_maxsi_labels.py` converts them to the space-separated form navis reads. |

> Dorkenwald S, Matsliah A, Sterling AR, Schlegel P, Yu SC, McKellar CE, et al. Neuronal wiring diagram of an adult brain. *Nature* 634(8032), 124–138 (2024). https://doi.org/10.1038/s41586-024-07558-y

The FlyWire annotations also ask users to cite:

> Schlegel P, Yin Y, Bates AS, Dorkenwald S, Eichler K, Brooks P, et al. Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature* 634(8032), 139–152 (2024). https://doi.org/10.1038/s41586-024-07686-5

> Matsliah A, Yu SC, Kruk K, Bland D, Burke AT, Gager J, et al. Neuronal parts list and wiring diagram for a visual system. *Nature* 634(8032), 166–180 (2024). https://doi.org/10.1038/s41586-024-07981-1

Two figure scripts, `figures/fig1/create_split_axon_dendrite_princeton.py` and `figures/fig3/fig3c_mixed_example_sfc.py`,
also need a FlyWire account and a CAVE token saved on your machine (see `figure_to_script_map.md`).

---

## Made by this repository

The scripts make these files; they are not committed. The folders below are those of `PIPELINE_RUN = "published"`. Under `"rebuild"`
the same files go to `data/derived_rebuild/` and `data/intermediate_rebuild/` (and `_rebuild_maxsi`,
`_rebuild_minfisherp`, `_rebuild_maxphi` with the matching `SPLIT_METHOD`), except the two raw synapse tables,
which stay in `data/derived/`, and the larval files, which stay in `data/larva/` and `data/larva_maxsi/` under both settings. Steps 01 to 07 and `build_swc_data.py` run only under `"rebuild"`.
[pipeline_overview.md](pipeline_overview.md) gives the run order and each script's inputs and outputs, and
[generated_tables.md](generated_tables.md) the columns.

| File | Location | Made by |
|------|----------|-------------|
| `synapses_783_non_processed_princeton.ftr`, then `synapses_783_article_princeton.ftr` | `data/derived/` | `03_merge_synapses_connectors.py`, then `04_build_master_synapse_table.py` |
| `swc_data.ftr` | `data/derived/` (the authors' copy; a fresh run writes `data/derived_rebuild/`) | `processing/build_swc_data.py`, from the skeletons |
| `neuron_data_full_article_princeton.ftr`, `SI_updated.ftr` | `data/derived/` | `05_build_neuron_metadata_table.py` |
| `connections_by_syn_type_reciprocal_types_filtered_article_princeton.ftr`, `full_reci_connections_article_filtered_princeton.ftr`, `reciprocity_list_full_article_princeton.ftr`, `connections_by_syn_type_filtered_article_princeton.ftr`, `connections_by_syn_type_reciprocal_filtered_article_princeton.ftr` | `data/derived/` | `06_build_connection_reciprocity_table.py` |
| `neurons_pca_princeton.ftr` | `data/derived/` | `07_pca_morphology.py` |
| `synapses_783_article_princeton_raw.ftr`, `synapses_783_article_princeton_raw_with_self.ftr` | `data/derived/` | `processing/build_raw_synapse_table.py` (needs the Codex CSV in its original 13-column order; runs only under `"published"`) |
| `SI_comparisons.ftr`, and its batch files in `data/intermediate/alternative_to_SI/all_neurons_batches/` | `data/derived/` | `08_alternative_split_methods.py` (optional) |
| `synapses_783_article_princeton.ftr`, `neuron_data_full_article_princeton.ftr`, `SI_updated.ftr` on the MaxSI split, and the label batch files in `data/intermediate_maxsi/maxsi_labels/` | `data/derived_maxsi/` | `09_maxsi_synapse_labels.py` (see `docs/pipeline_overview.md`, Script 09) |
| The same three tables and batch files on the MinFisherP and MaxPhi splits | `data/derived_minfisherp/`, `data/intermediate_minfisherp/minfisherp_labels/`, `data/derived_maxphi/`, `data/intermediate_maxphi/maxphi_labels/` | `09_maxsi_synapse_labels.py` with `METHOD = "minfisherp"` or `"maxphi"` (optional) |
| The tables of steps 06 and 07 and of the table-writing figure scripts, on the MaxSI split (same file names) | `data/derived_maxsi/`, `data/intermediate_maxsi/` | The same scripts, run with `SPLIT_METHOD = "maxsi"` after step 09 (the `_minfisherp` and `_maxphi` folders with those methods) |
| `PC1_table.csv` | `data/derived/` | `figures/fig4/build_pc1_table.py` |
| `PC1_table_with_predictions.csv` | `data/derived/` | `figures/fig4/syntype_x_pc1.py` |
| `reci_prop_princeton.ftr` | `data/derived/` | `figures/fig5/reciprocal_fraction.py` |
| `neurons_nt_bwf_frac.ftr` | `data/derived/` | `figures/fig7/fig7i_filopodia_fraction_x_nt.py` (from the Fig 7 bouton tables, which are not deposited) |
| Four PKL files per skeleton subfolder | `data/intermediate/processed_swc_data/` | `01_extract_compartments_SI.py` |
| Large-neuron PKL files | `data/intermediate/processed_big_neurons/` | `02a_large_neurons_pipeline.py` |
| Connector feather files | `data/intermediate/connectors/` | `02b_merge_connectors.py`, `02a_large_neurons_pipeline.py`, `03_merge_synapses_connectors.py` |
| Figure files | `outputs/figN/` (`outputs_maxsi/` under `SPLIT_METHOD = "maxsi"`, `outputs_rebuild/` and `outputs_rebuild_maxsi/` under `"rebuild"`; likewise for the other two methods) | Figure scripts |
| The larval tables on the MaxSI split, with `larva_maxsi_summary.ftr` and `larva_maxsi_failures.ftr` | `data/larva_maxsi/output/`, `data/larva_maxsi/results/` | `processing/larva/larva_maxsi_labels.py` |
| `larva_maxsi_pc1.ftr` (each larval cell's PC1) | `data/larva_maxsi/output/` | `processing/larva/larva_pc1.py`; read by `figures/fig4/larva_syntype_x_pc1_map.py` |
| Larval skeletons in the form navis reads | `data/larva/swc_navis/` | `processing/larva/larva_maxsi_labels.py` |
| `all_synapses_unprocess_larva_th0.9_SI_filt_issue_solved.ftr` and `SI_list_larva_th_0.9_linker_SI_filt_issue_solved` | `data/larva/output/`, `data/larva/results/` | `processing/larva/larva_process.py` (gives no usable output as committed) |

Inputs read by the code but not written by any script in this repository:

- `syn_bouton_filopodia.ftr`, read by the three `si_sim_*.py` scripts, and `syn_bouton_filopodia(new).ftr`, read by `post_on_filo_ratio.py` (Fig 7 section below)
- the Fig 7 bouton tables (`b_table_full_prince(new)_filtered.ftr`, `bwf_table_full_prince(new)_filtered.ftr`, `filopodia_reciprocal.ftr`, `filopodia_filtered.ftr`, `inertia_all.ftr`), `neuron_data.ftr` and `t2.ftr`, read by the `figures/fig7/fig7*.py` scripts, and the per-synapse bouton table `cluster_info_filtered.ftr`, read by `processing/bouton_maxsi_tables.py` (Fig 7 section below). Under `PIPELINE_RUN = "rebuild"` and `SPLIT_METHOD = "maxsi"` the scripts read MaxSI versions of the first two, in `data/derived_rebuild_maxsi/`, which `processing/bouton_maxsi_tables.py` builds from the per-synapse tables and the MaxSI synapse table
- `ex_table_npil_comp_pre_post_hiSI_PerTypeSummary.csv` and `Olfactory_ClassNeuropilStats.csv`, read by `figures/fig4/syntype_x_pathway_layers.py` (the revised Fig 4E). The first is the authors' per-type table: one row per pre type, post type and connection type, with its synapse count and one neuropil label, kept where the mean SI over the row's records, and over its two types' records, is at least 0.1. The script reads it only with `DATA_SOURCE = 'csv'`; by default it builds the per-type rows itself from the synapse tables, counting each synapse in its own neuropil. The second is only printed, and skipped if missing. The script also reads `neuron_data.ftr` (above) in place of the authors' neuron_data table, for the olfactory cell classes
- two tables from the earlier Buhmann synapse detection: `neuron_data_full_article.ftr` and `synapses_783_article.ftr`, read by `figures/fig2/si_comparisons.py`, `figures/fig3/si_x_correct_percent_per_compartment_buhmann.py`, `figures/fig3/si_x_npil_x_synapse_detection.py` and `figures/fig6/aadd_specific_circuits.py`, and with `DATA_SOURCE = "buhmann"` by `figures/fig3/fig3d_si_compartment_correct.py` and `figures/fig4/syntype_x_pathway_layers.py`. Under `PIPELINE_RUN = "rebuild"` and `SPLIT_METHOD = "maxsi"` the scripts read MaxSI versions of both, in `data/derived_rebuild_maxsi/`, which `processing/buhmann_maxsi_labels.py` builds from the published ones and step 09's cuts

None of these is deposited with this repository, and the scripts that need them stop until they are supplied (the
per-type CSV and the olfactory CSV are needed only as noted above). `config.py` also names a third Buhmann table,
`connections_by_syn_type_reciprocal_types_filtered_article.ftr` (`CONNECTIONS_TABLE_NONP_FTR`), which no script reads.

### The Buhmann tables (earlier synapse detection)

These tables come from the synapse predictions of Buhmann et al. (2021). The pipeline in `processing/` is written for
the Princeton synapse table from Codex (`PRINCETON_SYNAPSE_CSV`), and this repository includes neither the published
Buhmann tables nor the raw Buhmann synapse table they were built from. From the published tables it builds MaxSI
versions (`processing/buhmann_maxsi_labels.py`, under `PIPELINE_RUN = "rebuild"` and `SPLIT_METHOD = "maxsi"`), but
not the published (SFC) ones. Building those would mean running steps 01 to 06 again on that table, with at least
these changes:

- Attach synapses with `heal_attach` instead of `heal_attach_princeton` (both in `methods/methods_all.py`).
  `heal_attach` goes through `attach_synapses`, which keeps only synapses with a `cleft_score` of at least 30 (its
  `min_score` default), so the table needs a `cleft_score` column. The Princeton route applies no score cut.
- Steps 01 and 02a each define their own `heal_attach_princeton` (01 line 102, 02a line 63), which replaces the one
  in `methods/methods_all.py`; the calls to change are at 01 line 176 and 02a line 129.
- The Princeton route carries each synapse's `synapse_id` and `size` through steps 01, 02a, 02b and 03 (by column
  name in 01, partly by column position in the others), and step 03 merges on `synapse_id`. `attach_synapses` keeps
  neither column.
- The root-ID repair in steps 01 and 02a, and the column names they read, are written for the Codex CSV.

Whether the Buhmann run differed in anything else is not recorded here.

| Panel | Script | Buhmann table it needs |
|-------|--------|------------------------|
| Fig 2 Supp 1B/C | `figures/fig2/si_comparisons.py` | `neuron_data_full_article.ftr`. The script also loads `synapses_783_article.ftr` at line 195 but never uses it, so without that file it stops there, after saving `SI_corr_all.svg` and `SI_corr_per_sclass.svg` and before `SI_corr_combined_general_left_blue_clean.svg` |
| Fig 3 Supp 1A (Buhmann curve) | `figures/fig3/si_x_correct_percent_per_compartment_buhmann.py` | `neuron_data_full_article.ftr` |
| Fig 3 Supp 2A/B | `figures/fig3/fig3d_si_compartment_correct.py` with `DATA_SOURCE = "buhmann"` (the old-rule version: `si_x_correct_percent_per_compartment_buhmann.py`) | `neuron_data_full_article.ftr`, `synapses_783_article.ftr` |
| Fig 3 Supp 2C | `figures/fig3/si_x_npil_x_synapse_detection.py` | `neuron_data_full_article.ftr`, `synapses_783_article.ftr` (with `C_RULE = "all_cells"` only `synapses_783_article.ftr`) |
| Fig 6A–D, Supp 6-S1, 6-S2A, 6-S3, as submitted | `figures/fig6/aadd_specific_circuits.py` with `DATA_SOURCE = 'buhmann'` (its default is `'princeton'`, the tables of the rest of the paper) | `neuron_data_full_article.ftr`, `synapses_783_article.ftr` |

Under `PIPELINE_RUN = "rebuild"` and `SPLIT_METHOD = "maxsi"` these settings read the MaxSI Buhmann tables, so they
draw the panels on the MaxSI split, not as submitted; under `"published"` and `"maxsi"` they stop, as no MaxSI
Buhmann table is there.

---

## Filopodia and bouton analysis (Fig 7)

The Fig 7 scripts in `figures/fig7/` read tables no script here writes, and no code here builds them from the MSB
pipelines' output (the two bouton tables have MaxSI versions, built by `processing/bouton_maxsi_tables.py`). They are
not deposited, so ask the authors for them and save them under the names below. The submitted 7J was drawn from the
plain `syn_bouton_filopodia.ftr` and the submitted 7H from `syn_bouton_filopodia(new).ftr`, so each panel reads its
own copy.

| File | Config variable | Notes |
|------|----------------|-------|
| `data/raw/syn_bouton_filopodia.ftr` | `SYN_BOUTON_FTR` | Per-synapse table with `synapse_id` and `filopodia_on_post` (0/1), plus `bouton_on_pre` (0/1), `pre_x_cluster` and `pre_x_filopodia` (unused). Read by the three `si_sim_*.py` scripts in `figures/fig7/`, under every `SPLIT_METHOD` and `PIPELINE_RUN`. The scripts merge it on `synapse_id`, a row number of the Codex CSV (see `generated_tables.md`), so it must come from the same synapse table as `SYNAPSE_TABLE_FTR`; otherwise the merge runs without error and gives wrong flags. It covers the published synapse table's synapses; the scripts leave out, and count, any synapse of `SYNAPSE_TABLE_FTR` without a flag (on `data/derived_rebuild_maxsi`, the 760,723 synapses with an end in one of the 52 cells the published table lacks). It must also have exactly one row per `synapse_id`, store `filopodia_on_post` as integer or bool with no missing values, and have no other column whose name is also a column of `SYNAPSE_TABLE_FTR` (such as `pre`, `post` or `comp`). A repeated `synapse_id` silently duplicates synapses; a shared column name stops the scripts with a KeyError. |
| `data/raw/syn_bouton_filopodia(new).ftr` | `SYN_BOUTON_NEW_FTR` | The same rows and columns as `SYN_BOUTON_FTR`, with `bouton_on_pre` set to 0 for the 6,335,713 synapses whose presynaptic cell is not in `BOUTON_TABLE_FTR`, `filopodia_on_post` set to 0 for the 1,588,585 whose postsynaptic cell is not, and `pre_x_cluster` and `pre_x_filopodia` blanked on the same rows; nothing else differs. Read by `post_on_filo_ratio.py` (Fig 7H), so inputs onto cells outside the bouton analysis count there as not on a terminal. The same requirements as `SYN_BOUTON_FTR` apply. |
| `data/derived/b_table_full_prince(new)_filtered.ftr` | `BOUTON_TABLE_FTR` | One row per bouton, keyed `neuron` (the presynaptic neuron) and `cluster`: `count` (its output synapses; 2,552 rows have 1), `SI` (`NEURON_TABLE_FTR`'s, for the 84,956 of its 84,969 neurons that are in that table), `npil`, `comp` (A, D or empty), `primary_type`, `super_class` (not identical to any other table's) and `filopodia` (bool: the bouton carries a postsynaptic terminal). It leaves out the cells `msb_excluded_cell_types.csv` lists (`MSB_EXCLUDED_TYPES_CSV`, above), by the same rule as the next table: it has no neuron of a `pre_clustering` or `rms_inspection` type, and none whose sub_class in `neuron_data.ftr` is `uniglomerular`. `neuron` reads back as pandas' nullable `Int64`. Read by `fig7e`, `fig7f`, `fig7g` and `fig7i`. `processing/bouton_maxsi_tables.py` also reads it, as `BOUTON_TABLE_PUBLISHED_FTR` (this file under every setting), for its check against the published tables and for its cell labels. |
| `data/derived/bwf_table_full_prince(new)_filtered.ftr` | `BOUTON_FILOPODIA_TABLE_FTR` | The boutons with a postsynaptic terminal, keyed `neuron` and `bwf` (equal to `cluster` above for 2,877,620 of its 2,877,645 rows; the other 25 have no row in `BOUTON_TABLE_FTR` and drop out of the merges in `fig7g` and `fig7i`). `count` is a per-bouton count of postsynaptic synapses, by the rule written out in `processing/bouton_maxsi_tables.py` (which reproduces this table). Read by `fig7g` and `fig7i`. `processing/bouton_maxsi_tables.py` also reads it, as `BOUTON_FILOPODIA_TABLE_PUBLISHED_FTR` (this file under every setting), for its check against the published tables. |
| `data/derived/cluster_info_filtered.ftr` | `CLUSTER_INFO_FILTERED_FTR` | One row per presynaptic synapse in a bouton: `neuron` (the presynaptic cell), `cluster` (the bouton), `synapse_id`, `partner_id`, `x`, `y`, `z`, and a `comp` column the step does not use; 50,559,574 rows, 192,979 of them with no `synapse_id`. `processing/bouton_maxsi_tables.py` builds the two bouton tables from it and `filopodia_filtered.ftr`. On the published synapse table the step's rules give back the two tables above (it checks this on every run); on the MaxSI synapse table it writes their MaxSI versions to `data/derived_rebuild_maxsi/`. Only that step reads it, always from `data/derived/` whatever `PIPELINE_RUN` and `SPLIT_METHOD` say, and the same holds for its other published inputs, each under a config name that no setting moves: the two published bouton tables (`BOUTON_TABLE_PUBLISHED_FTR`, `BOUTON_FILOPODIA_TABLE_PUBLISHED_FTR`), `filopodia_filtered.ftr` (`FILOPODIA_FILTERED_PUBLISHED_FTR`) and the published synapse table `synapses_783_article_princeton.ftr` (`SYNAPSE_TABLE_PUBLISHED_FTR`). |
| `data/derived/filopodia_reciprocal.ftr` | `FILOPODIA_RECIPROCAL_FTR` | One row per synapse onto a bouton terminal (`pre`, `post`, `synapse_id`, `comp`, `cluster`, `bwf`, coordinates) with `global_reciprocal` and `local_reciprocal` (bool); `post` is the neuron that owns the bouton. `comp` takes all nine values (21,493 rows contain L, and `fig7k` keeps them), `cluster` is empty in 4,374,028 rows, and `synapse_id` is stored as a float. Read by `fig7k`. |
| `data/derived/filopodia_filtered.ftr` | `FILOPODIA_FILTERED_FTR` | One row per synapse onto a bouton terminal, with `cluster` the bouton of `post`: the 13,382,704 synapses of `FILOPODIA_RECIPROCAL_FTR`, plus 29,552 rows that repeat a `synapse_id` and 49,708 rows with no `synapse_id` (stored as a float for that reason). Read by `fig7e` (left pie), which uses its `pre` column and counts every row as a synapse, extra rows included. `processing/bouton_maxsi_tables.py` also reads it, as `FILOPODIA_FILTERED_PUBLISHED_FTR` (this file under every setting), for each postsynaptic synapse's bouton. |
| `data/derived/t2.ftr` | `SYNAPSES_NO_SELF_FTR` | Columns `index`, `pre`, `post` and the pre/post coordinates; 70,312,046 rows. Despite its config name, it is not all synapses minus self-synapses. It is a subset of `SYNAPSE_TABLE_FTR` (which has no self-synapses) that leaves out 1,029,925 whole (pre, post) connections, 5,727,011 synapses, for an unknown reason; no connection is split. Read by `fig7e` (right pie), which uses only `pre`. |
| `data/derived/inertia_all.ftr` | `BOUTON_INERTIA_FTR` | One row per neuron (125,553, including 41,976 that are not in `BOUTON_TABLE_FTR`): `mean_rms`, `mean_centroid_dist` and two other spread measures of its boutons, computed by code not in this repository. Read by `fig7s2a` and `fig7s2b`. |
| `data/derived/neuron_data.ftr` | `NEURON_DATA_FTR` | Neuron metadata with `root_id`, `nt_type` (as in the Codex `neurons.csv`) and `sub_class`. Its `SI` column is not used and is neither of this repository's SIs; it is close to the Buhmann neuron table's (within 0.001 for 111,525 of the 129,903 cells they share). Read by `fig7e`, `fig7i` and `fig7s2b` (`nt_type`, `sub_class`) and by `figures/fig4/syntype_x_pathway_layers.py` (`class`, `primary_type`, `cell_type`), where it stands in for the authors' neuron_data table. It stays in `data/derived/` under both `PIPELINE_RUN` settings. |
| `data/raw/consolidated_cell_types.csv.gz` | `CONSOLIDATED_CELL_TYPES_CSV` | Codex download (`root_id`, `primary_type`, `additional_type(s)`). Read by `fig7s2b`, by `fig7s2a`, which does not use it, and by `processing/bouton_maxsi_tables.py`. |
| `data/derived/neurons_nt_bwf_frac.ftr` | `NEURONS_NT_BWF_FTR` | Written by `figures/fig7/fig7i_filopodia_fraction_x_nt.py`: one row per neuron, with `neuron` (pandas nullable `Int64`), `nt_type` and `filopodia_fraction`. Read by `figures/fig7/filopodia_x_mirror_neurons.py` and `figures/fig5/reciprocal_fraction_x_bwf.py`. The authors' Figure 7 code writes the same table as `neurons_nt_bwf_frac_filtered.ftr`. The original `reciprocal_fraction_x_bwf.py` read an unfiltered `neurons_nt_bwf_frac.ftr`, which has not been supplied, so the script here runs on a different table (it draws no submitted panel). The mirror script merges the whole table, so it must not contain `primary_type`, `side` or `super_class`; a repeated `neuron` silently changes which cell types count as left/right pairs. |
| `data/derived/filopodia/` | `FILOPODIA_DIR` | Written by the three `figures/fig7/si_sim_*.py` scripts (`nodes_baseline.ftr`, `nodes_no_filopodia.ftr`, `nodes_sampleout.ftr`) and read by `figures/fig7/si_comb_analysis.py`. |
