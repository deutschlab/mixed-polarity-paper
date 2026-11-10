# Figure-to-Script Map

This document maps each manuscript figure panel to the script that generates it, the input
tables it requires, and the output directory where figures are saved.

All figure scripts import `config.py` (paths) and `methods/methods_all.py` (utilities).
Figures are saved under `outputs/` (`outputs/figN/` for the figure folders; the other folders and `processing/phi_threshold.py` use their own names, given below); a few scripts also write tables to `data/derived/` (shown as "writes" below). None of these are committed to this repository.
With `SPLIT_METHOD = "maxsi"` in `config.py` (after step 09), every output path below starts with `outputs_maxsi/` instead of `outputs/`; `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR`, `SI_UPDATED_FTR` and the tables built from them are read from `data/derived_maxsi/` or `data/intermediate_maxsi/`, and the tables a script writes go there too. A script that reads a table with no MaxSI version (`MAXSI_NOT_AVAILABLE` in `config.py`) stops. `"minfisherp"` and `"maxphi"` work the same way, with `outputs_minfisherp/`, `data/derived_minfisherp/` and so on.
With `PIPELINE_RUN = "rebuild"` (a fresh run of the pipeline), the same applies with `outputs_rebuild/` and `data/derived_rebuild/` (`outputs_rebuild_maxsi/` and `data/derived_rebuild_maxsi/` with `"maxsi"`; likewise `_rebuild_minfisherp` and `_rebuild_maxphi`).

---

## Running the figure scripts

`figures/fig1/create_split_axon_dendrite_princeton.py` and the Fig 3C script make a Neuroglancer link through `caveclient` before they draw, so they stop with an error unless you have a FlyWire account and a CAVE token saved on your machine (see the caveclient documentation). Every script imports `methods/methods_all.py`, which imports `caveclient` and `nglui`, so both packages must be installed (they are in the pinned requirements); only these two scripts need the token. Both also write a `temp_link_<random hex>.html` file into the folder you run them from and try to open it with Microsoft Edge at its Windows path; on macOS and Linux they print "Edge executable not found" and carry on.

Each script saves its figures as SVG in its own folder, usually `outputs/figN/<name>/`, `outputs/split_methods/<name>/` or `outputs/supporting/<name>/`. One exception: `figures/fig1/create_split_axon_dendrite_princeton.py` writes to `outputs/fig3/pc1_example/`. The two Fig 4 PC1 scripts also write PNG, and `figures/fig5/syn_type_x_reci_x_npil.py` writes PDF. A few figure scripts also write tables into `data/derived/`. Under `SPLIT_METHOD = "maxsi"` the figures go to `outputs_maxsi/` instead of `outputs/`, and those tables to `data/derived_maxsi/` or `data/intermediate_maxsi/`.

A note on SI: step 04 stores each neuron's SI in the synapse table as `SI_pre` and `SI_post`, and step 05 stores it in the neuron table and in `SI_updated.ftr`. All three come from the same output of steps 01 and 02a, so after a fresh run each neuron has the same SI in all three (`SI_updated.ftr` has a few more neurons than the neuron table). Tables from an earlier run may not match, which is why the synapse-type tools refer to the table SI as "older" and to `SI_updated.ftr` as "corrected".

Most figure scripts take SI from the neuron table or the synapse table, and ten always take it from `SI_updated.ftr` (six in `figures/fig2`–`fig3`, `figures/fig7/filopodia_x_mirror_neurons.py` and three in `figures/split_methods/`: the two MaxSI scripts and `four_methods_si_comparison.py`, which reads each split's); the revised Fig 4E and the Figure 6 script can be set to it (`DATA_SOURCE = 'princeton'` and `SI_SOURCE = 'SI_updated'`). The neuron and synapse tables the figures were made from (not included; see the README's [What you can reproduce](../README.md#what-you-can-reproduce)) come from an earlier run whose SI kept synapses from a neuron to itself, except for the 182 large cells from step 02a. The manuscript's Methods exclude self-synapses, as `SI_updated.ftr` does.

---

## Figure 1 — Synaptic polarity and compartment canonicality

Shows that neurons separate synaptic inputs to dendrites and outputs to axons. Panels include
the SI distribution, axon/dendrite split examples, and canonical vs mixed neuron classes.

| Panel | Script | Input tables | Output path |
|-------|--------|-------------|-------------|
| Fig 1C | `figures/fig1/fig1c_example_splits.py` (after step 09 with MaxSI; `PIPELINE_RUN = "rebuild"`): the cells 720575940628463468 (LC31b), 720575940614955304 (LC15) and 720575940633095137 (DNp09 in the legend; FlyWire's type is DNp71), split by MaxSI | `MAXSI_LABEL_BATCH_DIR`, `MAXSI_SYNAPSE_TABLE_FTR`, `SYNAPSE_TABLE_RAW_FTR`, skeletons in `SWC_DIR` | `outputs/fig1/fig1c_example_splits/` (`fig1c_example_splits.svg`, `fig1c_canonical_shares.csv`: each compartment's canonical count, total and percent) |
| Fig 1E | `figures/fig1/canonicality_axon_dend.py` | `NEURON_TABLE_FTR` | `outputs/fig1/canonicality_axon_dend/` |
| Fig 1G | `figures/fig1/si_x_canonicality.py` | `NEURON_TABLE_FTR` | `outputs/fig1/si_x_canonicality/` |
| Fig 1 Supp D | `figures/fig1/canonicality_violin_intrinsic_sclass.py` | `NEURON_TABLE_FTR` | `outputs/fig1/canonicality_violin_intrinsic_sclass/` |
| Fig 1 Supp D | `figures/fig1/canonicality_violin_non_intrinsic_sclass.py` | `NEURON_TABLE_FTR` | `outputs/fig1/canonicality_violin_non_intrinsic_sclass/` |
| Fig 1 Supp E | `figures/fig1/fig1supp_e_sclass_canonicality_corr.py` | `NEURON_TABLE_FTR` | `outputs/fig1/sclass_canonicality_corr/` |

> `figures/fig1/create_split_axon_dendrite_princeton.py` is not the Fig 1C script and draws none of its cells. It reads `NEURON_TABLE_FTR`,
> `SYNAPSE_TABLE_RAW_FTR` (which version is set by `RAW_TABLE_KEEP_SELF_SYNAPSES`) and skeletons in `SWC_DIR`, makes a
> FlyWire link for 720575940627236874, splits 720575940620605776 and 720575940620451974, and writes
> `pc1_example_AA1.svg` to `outputs/fig3/pc1_example/`.

---

## Figure 2 — SI variation and morphological models

Shows how SI varies across neuron super-classes, neurotransmitter types, and primary types.
Builds random-forest and logistic-regression models predicting SI from morphological features.
Includes an adult vs larva SI comparison and PCA of morphological features.

| Panel | Script | Input tables | Output path |
|-------|--------|-------------|-------------|
| Fig 2A | `figures/fig2/fig2a_si_cdf_adult_larva.py` | `LARVA_SI_FTR`, `LARVA_SYNAPSES_FTR`, `NEURON_TABLE_FTR`, `SI_UPDATED_FTR` | `outputs/fig2/SI_CDF_adult_larva/` |
| Fig 2B | `figures/fig2/si_x_sclass.py` | `NEURON_TABLE_FTR` | `outputs/fig2/SI_x_sclass/` |
| Fig 2C | `figures/fig2/si_x_nt.py` | `NEURON_TABLE_FTR` | `outputs/fig2/SI_x_nt/` |
| Fig 2D | `figures/fig2/feat_x_si_x_sclass_corr.py` | `NEURON_TABLE_FTR`, `SI_UPDATED_FTR` | `outputs/fig2/feat_x_si_x_sclass_corr/` |
| Fig 2E (`regplot_with_tree_variance_princeton.svg`, `rsquare_model_comp_princeton.svg`) | `figures/fig2/feat_rf_and_lr_models.py` | `NEURON_TABLE_FTR` | `outputs/fig2/feat_rf_and_lr_models/` |
| Fig 2 Supp 4B (`feat_import_bar_princeton.svg`) | `figures/fig2/feat_rf_and_lr_models.py` | `NEURON_TABLE_FTR` | `outputs/fig2/feat_rf_and_lr_models/` |
| Fig 2F | `figures/fig2/si_x_pca.py` | `NEURON_TABLE_FTR`, `SI_UPDATED_FTR`, `SYNAPSE_TABLE_FTR` (loaded, but nothing that runs uses it) | `outputs/fig2/SI_x_PCA/` |
| Fig 2G | `figures/fig2/si_x_pca.py` | `NEURON_TABLE_FTR`, `SI_UPDATED_FTR`, `SYNAPSE_TABLE_FTR` (loaded, but nothing that runs uses it) | `outputs/fig2/SI_x_PCA/` |
| Fig 2 Supp 1A | `figures/fig2/si_x_twigs.py` | `NEURON_TABLE_FTR`, `NEUROPIL_SYNAPSE_CSV` | `outputs/fig2/SI_x_twigs/` |
| Fig 2 Supp 1B/C | `figures/fig2/si_comparisons.py` | `NEURON_TABLE_FTR`, `NEURON_TABLE_NONP_FTR`, `SYNAPSE_TABLE_FTR`, `SYNAPSE_TABLE_NONP_FTR` | `outputs/fig2/SI_comparisons/` |
| Supp 2-S2 A | `figures/fig2/si_x_primary_types_mirror.py` | `NEURON_ANNOTATIONS_CSV`, `NEURON_TABLE_FTR`, `SI_UPDATED_FTR` | `outputs/fig2/SI_x_primary_types_mirror/` |
| Supp 2-S2 D | `figures/fig2/si_x_primary_types.py` | `NEURON_TABLE_FTR` | `outputs/fig2/SI_x_primary_types/` |
| Not a panel of the submitted paper | `figures/fig2/pca_on_feat.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` (loaded, but nothing that runs uses it) | `outputs/fig2/pca_on_feat/` |

> Fig 2A requires larval data files — see [data_availability.md](data_availability.md). A larval cell is drawn when it has an SI and at least one synapse on the axon and one on the dendrite, the rule the adult side applies through `dend_correct` and `axon_correct`. Under `SPLIT_METHOD = "maxsi"` its two larval tables are the MaxSI ones written by `processing/larva/larva_maxsi_labels.py`, so both the adult and the larval SI come from a MaxSI split. Under `"minfisherp"` or `"maxphi"` Fig 2A stops, as there is no larval split for those methods. Under `"maxsi"` it needs both step 09 (MaxSI, for the adult tables) and the larval step; the larval step itself can run under any `SPLIT_METHOD`.
> Fig 2 Supp 1A requires the neuropil synapse CSV — see [data_availability.md](data_availability.md).
> In `outputs/fig2/SI_x_PCA/`, Fig 2F is `weights.svg` (line 100) and Fig 2G is
> `optic_visual_projection_PC1_SI_CDF_vertical_v2.svg` (last written at line 910; lines 614 and 753 write the same
> name first; only the last write remains). The script also saves variants, saves bar plots in `colored_bars/`, and
> writes `full_info.ftr` to `outputs/fig2/SI_x_PCA/`.
> Supp 2-S2 A is `SI_corr_sides_princeton_v2.svg` and reads the SI from `SI_UPDATED_FTR`; Supp 2-S2 D is
> `std_of_si_within_groups_vs_between_princeton.svg` and uses the SI in `NEURON_TABLE_FTR`. `pca_on_feat.py` saves
> signed PC1 and PC2 contributions and violin plots.
> `si_x_primary_types.py` has a run setting, `SLOW_STATS` (default `False`). With `False` the script finishes in
> seconds: the sums of squares behind its eta squared and omega squared come from the group means, and the post-hoc
> test over every pair of cell types is skipped (the paper reports no pairwise cell-type comparison). `True` runs the
> original code: a model fit with one column per cell type (12 minutes measured) and Games-Howell over 28.2 million
> pairs (an estimated 4-5 days), of which it prints five. Both print the same Welch ANOVA, eta and omega squared.
> `figures/fig7/filopodia_x_mirror_neurons.py` (Supp 7-S4) has the same setting for the same block.

---

## Figure 3 — Synapse-type composition, identity, and classifiers

Shows how the four synapse compartment types (AA/AD/DA/DD) are distributed across neuron types.
Builds classifiers predicting synapse type from morphological and connectivity features. Tests
whether neurons within the same type use the same synapse-type pattern.

| Panel | Script | Input tables | Output path |
|-------|--------|-------------|-------------|
| Fig 3A | `figures/fig3/si_x_neuron_types.py` | `NEURON_ANNOTATIONS_CSV`, `NEURON_TABLE_FTR` | `outputs/fig3/si_x_neuron_types/` |
| Fig 3C | `figures/fig3/fig3c_mixed_example_sfc.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig3/mixed_example/` |
| Fig 3D | `figures/fig3/fig3d_si_compartment_correct.py` | `NEURON_TABLE_FTR`, `SI_UPDATED_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig3/SI_x_correct_compartment/` |
| Fig 3D (AD content, mixed) | `figures/fig3/fig3supp_ad_content_si_mixed.py` | `NEURON_TABLE_FTR`, `SI_UPDATED_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig3/supp_ad_content_si_mixed/` |
| Fig 3E (drawn with the neuron table's SI; see the note) | `figures/fig3/fig3d_si_compartment_correct.py` | `NEURON_TABLE_FTR`, `SI_UPDATED_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig3/SI_x_correct_compartment/` |
| Not a panel of the submitted paper (synapse types per neuropil with an SI cutoff; stops before saving, see the note) | `figures/fig3/fig3e_syntype_composition.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig3/syntype_composition/` |
| Fig 3F (left pie: pairs of different cell types, each cell at SI ≥ 0.1; also the AD content curve, on the same pairs) | `figures/fig3/fig3supp_ad_content_si.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig3/supp_ad_content_si/` |
| Fig 3F (right pie: same-type pairs, each cell at SI ≥ 0.1) | `figures/fig3/syntype_x_identity.py` | `CLASSIFICATION_CSV`, `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig3/syntype_x_identity/` |
| Fig 3G | `figures/fig3/nt_syn_type_same_not_same.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig3/nt_syn_type_same_not_same/` |
| Fig 3H | `figures/fig3/syntype_x_strength_x_identity.py` | `CONNECTIONS_TABLE_FTR`, `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig3/syntype_x_strength_x_identity/` |
| Fig 3I | `figures/fig3/syn_type_prob_based_on_other_syn_type_princeton.py` | `CONNECTIONS_TABLE_FTR`, `NEURON_TABLE_FTR` | `outputs/fig3/syn_type_prob/` |
| Fig 3J | `figures/fig3/syntype_x_features.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig3/syntype_x_features/` |
| Fig 3 Supp 1A, Buhmann curve, and the old-rule 3D, 3E and 3F on the Buhmann synapses | `figures/fig3/fig3d_si_compartment_correct.py` with `FIG3D_DATA_SOURCE=buhmann` and `FIG3D_MIXED_RULE=cutoffs` | `NEURON_TABLE_NONP_FTR`, `SYNAPSE_TABLE_NONP_FTR` | `outputs/fig3/SI_x_correct_compartment_buhmann_cutoffs_rule/` (the curve is `axon_dend_correct_SI_v2.svg`) |
| Fig 3 Supp 1A (Buhmann curve, earlier version; old-rule pies) | `figures/fig3/si_x_correct_percent_per_compartment_buhmann.py` | `NEURON_TABLE_NONP_FTR`, `SYNAPSE_TABLE_NONP_FTR` | `outputs/fig3/si_x_correct_compartment_buhmann/` |
| Fig 3 Supp 2A/B (3D's rule: a cell below SI 0.1 is mixed) | `figures/fig3/fig3d_si_compartment_correct.py` with `DATA_SOURCE = "buhmann"` (or `FIG3D_DATA_SOURCE=buhmann`) | `NEURON_TABLE_NONP_FTR`, `SYNAPSE_TABLE_NONP_FTR` | `outputs/fig3/SI_x_correct_compartment_buhmann_si_rule/` (A: `synaptic_type_pie2_v2.svg`, B: `synaptic_type_npils2_v2.svg`) |
| Fig 3 Supp 2C | `figures/fig3/si_x_npil_x_synapse_detection.py` | `SYNAPSE_TABLE_FTR`, `SYNAPSE_TABLE_NONP_FTR` | `outputs/fig3/si_x_npil_x_synapse_detection/` |

> Fig 3E is `synaptic_type_npils2_v2.svg` (line 440 of `fig3d_si_compartment_correct.py`). The script's
> `MIXED_RULE` (or `FIG3D_MIXED_RULE` in the environment) picks how a compartment is called mixed; `"cutoffs"` writes to
> `outputs/fig3/SI_x_correct_compartment_cutoffs_rule/` (with `"buhmann"`, `..._buhmann_cutoffs_rule/`). `"si"`, the setting for the main 3D and 3E, marks both
> compartments of a cell mixed when the cell's SI (from `SI_UPDATED_FTR`, lines 21-24) is below 0.1. `"cutoffs"`, for
> the supplement, marks a compartment mixed when its share of correct synapses is not above the rolling curve's value
> at SI 0.1, the rule of the published 3D and 3E. The published panels were drawn on the neuron table's SI: with lines
> 21-24 switched off, so that the script keeps the SI in `NEURON_TABLE_FTR`, `"cutoffs"` draws the published 3E bars
> exactly (the published figure also lists three neighbouring pairs of neuropil labels in the opposite order: MB
> CA/BU, MB PED/FB and MB VL/AL). As committed, those lines use `SI_UPDATED_FTR`, which changes the cut-offs, the bars
> and their order. `fig3supp_ad_content_si_mixed.py` has the same setting and applies it at each point of its SI grid.
> `GROUP_MIXED = True` then counts every synapse with a mixed end (AM, MA, MD, DM, MM) as one category, "Mixed", in
> the 3D pie and 3E. The shares are pooled over synapses, and "Mixed" means at least one end is mixed. The same script
> also draws 3F's two pies with mixed ends (`syn_type_same_ptype_pie_mixed.svg` and `syn_type_diff_ptype_pie_mixed.svg`):
> every synapse between these cells, with no SI filter, split into pairs of the same `primary_type` and pairs of
> different types. With `"cutoffs"` they are the supplement's old-rule 3F. `fig3e_syntype_composition.py` instead
> marks each end of a synapse mixed when that cell's SI computed with self-synapses kept (`SI_pre`/`SI_post` in the synapse table) is below 0.1,
> and as committed it stops at line 40 (`KeyError: ['neuron']`) before any figure is saved.

---

## Figure 4 — Synapse type vs PC1 morphological gradient

Tests whether the morphological gradient (PC1) predicts synapse-type composition. Compares
multiple classification models.

| Panel | Script | Input tables | Output path |
|-------|--------|-------------|-------------|
| (table for Fig 4A / 4C) | `figures/fig4/build_pc1_table.py` | `CONNECTIONS_TABLE_FTR`, `NEURON_TABLE_FTR`, `PCA_TABLE_FTR` | writes `PC1_TABLE_CSV` (`data/derived/PC1_table.csv`); run before `syntype_x_pc1.py` |
| (tables for Fig 4A and 4B) | `figures/fig4/fig4_heldout_models.py` | `NEURON_TABLE_FTR`, `PCA_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | writes `FIG4_HELDOUT_MODELS_FTR` and `FIG4_HELDOUT_CONFUSION_FTR`: each model scored on neurons it never saw (five cell splits); run before `models_comparison.py`; about 25 minutes |
| Fig 4C, 4D-2 (cells of each super-class per 0.1 of PC1, `kde_sclass.svg`) / Supp; the published 4A and 4D | `figures/fig4/syntype_x_pc1.py` | `NEURON_TABLE_FTR`, `PC1_TABLE_CSV`, `PCA_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig4/syntype_x_pc1/`, where it also saves and reloads the model `rf_pc1_model.joblib`; writes `PC1_TABLE_PREDICTIONS_CSV` |
| Fig 4D, larval version (proposed) | `figures/fig4/larva_syntype_x_pc1_map.py` | `LARVA_MAXSI_SYNAPSES_FTR`, `LARVA_MAXSI_SUMMARY_FTR`, `LARVA_MAXSI_PC1_FTR`, `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig4/larva_syntype_x_pc1_map/`: `decision_map_pc1_percentile_adult.svg`, `decision_map_pc1_percentile_larva.svg`, both side by side (`decision_map_pc1_percentile_adult_larva.svg`) and `recall_unseen_cells_adult_larva.svg`; writes `LARVA_4D_RESULTS_FTR` and `LARVA_4D_CONFUSION_FTR`. Runs under `SPLIT_METHOD = "maxsi"` only, after `processing/larva/larva_maxsi_labels.py` and `processing/larva/larva_pc1.py`; about 8 minutes |
| Fig 4A, 4B | `figures/fig4/models_comparison.py` | `CONNECTIONS_TABLE_FTR`, `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR`, `FIG4_HELDOUT_MODELS_FTR`, `FIG4_HELDOUT_CONFUSION_FTR` | `outputs/fig4/models_comparison/`: `confusion_matrix_pc1_unseen_neurons.svg` (4A) and `accuracy_values.svg` (4B) |
| Fig 4E (revision) | `figures/fig4/syntype_x_pathway_layers.py` (Fig 4E with `PIPELINE_RUN = "rebuild"` and `SPLIT_METHOD = "maxsi"`) | `SYNAPSE_TABLE_FTR`, `NEURON_TABLE_FTR` and `NEURON_DATA_FTR` (its `class` column gives the olfactory types; not deposited) with the default `DATA_SOURCE = 'princeton'`; `SI_UPDATED_FTR` with `SI_SOURCE = 'SI_updated'`; the Buhmann tables `NEURON_TABLE_NONP_FTR` and `SYNAPSE_TABLE_NONP_FTR` with `'buhmann'` (not deposited); the authors' `PER_TYPE_SUMMARY_CSV` with `'csv'` (not deposited); `OLFACTORY_CLASS_NEUROPIL_CSV` (printed only, optional) | `outputs_rebuild_maxsi/fig4/syntype_x_pathway_layers/` for Fig 4E (`outputs/...` under the default settings): the four panels of the revised 4E are `LayerComposition_AllSynapses_collapsed_lines.svg`, `LayerComposition_WithinType_collapsed_lines.svg`, `LayerComposition_Olfactory_AllSynapses_lines.svg` and `LayerComposition_Olfactory_WithinType_lines.svg` (the olfactory lines show AL and the mushroom body lobes, `OLFACTORY_LINE_LAYERS`); also two stacked-bar plots and the tables as CSV, with all four olfactory layers |
| Fig 4D check, revision (planned): most common type on 4D's PC1 axes; new Fig 4 supplement panel: the steps of two-step AD chains, against PC1 shuffled across all cells (the CSV also holds PC1 shuffled within super-class) | `figures/fig4/ad_chains_x_pc1.py` | `NEURON_TABLE_FTR`, `PCA_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig4/ad_chains_x_pc1/`: `ad_chains_most_common_type.svg` and `ad_chains_steps.svg`, each with a CSV of its numbers |
| Fig 4D-1 (the map, with the four numbered examples of the legend) | `figures/fig4/syntype_x_pc1_simple_model.py` | `NEURON_TABLE_FTR`, `PCA_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig4/syntype_x_pc1_simple_model/` |

> `syntype_x_pc1_simple_model.py` saves a feature-importance plot, a confusion matrix and a decision surface
> for a model on PC1, the same kinds of plot as Fig 4A and 4D. It reads the PCA table step 07 writes
> (`PCA_TABLE_FTR`), like `syntype_x_pc1.py`. The authors' delivered plots were made from a PCA table that is
> not available, and a rerun differs from them (by up to 3.4 points in a cell of the
> confusion matrix; accuracy 40.5% against 39.6%). That is more than the seed or the sign of PC1 changes, and fits
> a different table, but that table cannot be checked. A rerun can be checked against an independent rebuild of
> the model with `tools/check_simple_pc1_model.py` ([docs/tools.md](tools.md#checking-the-simple-pc1-model)). `syntype_x_pc1.py` draws a decision surface too, with its forest of trees without a leaf limit; its save line is
> commented out. That forest drew the published Fig 4D (compared with the published panel's map). The redrawn 4D is
> the 100-leaf model's map, the model 4A reports on neurons it never saw. Fig 4 Supp 1 holds only skeleton
> images of the Fig 4D neurons.
>
> In the revision, Fig 4E is new: `figures/fig4/syntype_x_pathway_layers.py`, ported from the corresponding
> author's MATLAB script. By default it builds its per-type rows from the synapse and neuron tables, so each
> synapse counts in its own neuropil and both cells need SI >= 0.1; `DATA_SOURCE = 'csv'` reads the authors' table.
> `tools/check_fig4e_logic.py` checks its calculations on made-up tables. The submitted 4E (example cells)
> becomes Fig 4 Supplement 1 in the revision and has no script.

---

## Figure 5 — Reciprocal connectivity

Analyses the fraction of reciprocal connections across neuron types. Tests whether reciprocally
connected pairs use predictable synapse-type patterns. Builds a random-forest model for
reciprocity prediction.

| Panel | Script | Input tables | Output path |
|-------|--------|-------------|-------------|
| Fig 5A-C / Supp | `figures/fig5/reciprocal_fraction.py` | `CONNECTIONS_TABLE_FTR`, `NEURON_TABLE_FTR` | `outputs/fig5/reciprocal_fraction/`; writes `RECI_PROP_FTR` |
| Not a panel of the submitted paper (per super-class correlations of `filopodia_fraction` with the reciprocal ratio and with SI) | `figures/fig5/reciprocal_fraction_x_bwf.py` | `CONNECTIONS_TABLE_FTR`, `NEURONS_NT_BWF_FTR`, `NEURON_TABLE_FTR` | `outputs/fig5/reciprocal_fraction_x_bwf/` |
| Fig 5D | `figures/fig5/fig5d_chi_reci_identity.py` | `CONNECTIONS_TABLE_FTR`, `FULL_RECI_CONNECTIONS_FTR`, `NEURON_TABLE_FTR` | `outputs/fig5/chi_reci_identity/` |
| Fig 5E, 5F | No script: example reciprocal pairs (tables of synapse counts, an EM cross-section) | — | — |
| Fig 5G (All) | `figures/fig5/syn_type_x_reci_x_dominance.py` | `FULL_RECI_CONNECTIONS_FTR`, `NEURON_TABLE_FTR` | `outputs/fig5/syn_type_x_reci_x_dominance/` |
| Fig 5G (Ipsilateral, Contralateral) | `figures/fig5/syn_type_x_reci_x_dominance_x_sides.py` | `CLASSIFICATION_CSV`, `FULL_RECI_CONNECTIONS_FTR`, `NEURON_TABLE_FTR` | `outputs/fig5/syn_type_x_reci_x_dominance_x_sides/` |
| Fig 5H | `figures/fig5/syn_type_x_reci_x_dominance.py` | `FULL_RECI_CONNECTIONS_FTR`, `NEURON_TABLE_FTR` | `outputs/fig5/syn_type_x_reci_x_dominance/` |
| Fig 5I | `figures/fig5/syn_type_x_reci_x_npil.py` | `CONNECTIONS_TABLE_FTR`, `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig5/syn_type_x_reci_x_npil/` |
| Not a panel of the submitted paper (Supp 5-S1 has only panels A and B, both examples) | `figures/fig5/reciprocal_fraction_model.py` | `NEURON_TABLE_FTR`, `RECI_PROP_FTR` | `outputs/fig5/reciprocal_fraction_model/`; writes `RF_MODEL_PKL` |
| Not a panel of the submitted paper | `figures/fig5/syn_type_strength_identity.py` | `CONNECTIONS_TABLE_FTR`, `FULL_RECI_CONNECTIONS_FTR`, `NEURON_TABLE_FTR` | `outputs/fig5/syn_type_strength_identity/` |

> Fig 5G and 5H: in `syn_type_x_reci_x_dominance/`, 5G (All) is `same_type_percentage_princeton.svg` and 5H is
> `not_same_type_percentage_princeton.svg`. In `syn_type_x_reci_x_dominance_x_sides/`, 5G Ipsilateral is
> `same_type_percentage_same_side_totalnorm_princeton.svg` and 5G Contralateral is
> `not_same__side_type_percentage_totalnorm_princeton.svg` (same-type pairs on opposite sides, despite the
> name). That script's other two SVGs, which reuse the base script's file names, are not panels.
> 5G (All) and 5H keep pairs whose two cells are in the four intrinsic super-classes, as the sides script does, and give
> each cell as a share of the pairs with one dominant synapse type (more than half of the synapses) in each direction,
> so each panel sums to 100%; the script asserts it. 3G (`nt_syn_type_same_not_same.py`) keeps synapses whose two cells
> are in its filtered neuron table. 3H's inset, the cumulative curve for SI >= 0.1 (`syntype_x_strength_x_identity.py`),
> is on AA, AD, DA and DD synapses between intrinsic cells with a cell type and both compartment flags, as the main
> panel; its "All synapses" curve is every row of the synapse table (every super-class, any SI, all nine comp values). 4B's cell-type model
> (`fig4_heldout_models.py`) picks, for each pair of cell types, the synapse type with the largest share of its own
> type's training synapses (each synapse weighted by 1 / its type's training count), the rule that maximises balanced
> accuracy on the training synapses; the three forests are trained on the balanced resample.

> `reciprocal_fraction.py` also generates `RECI_PROP_FTR` as a side output, which is required
> by `reciprocal_fraction_model.py`. Run `reciprocal_fraction.py` before `reciprocal_fraction_model.py`.

> `reciprocal_fraction_model.py` also generates `RF_MODEL_PKL`
> (`data/intermediate/reciprocity/models/final_random_forest_model_princeton.pkl`).

## Split methods (revision work) and supporting scripts

Scripts ported from the authors' original analysis scripts that are not panels of the submitted paper. The `split_methods/` ones
compare the published (SFC) axon/dendrite cut with the alternative cuts; the `supporting/` one backs a
number in the text.

| Purpose | Script | Input tables | Output path |
|-------|--------|-------------|-------------|
| Supp 2-S2 D layout on the MaxSI cut's SI | `figures/split_methods/maxsi_x_primary_types.py` | `NEURON_TABLE_SFC_FTR`, `SI_UPDATED_SFC_FTR` (the SFC cells and SI under either `SPLIT_METHOD`), `SI_COMPARISONS_FTR` | `outputs/split_methods/maxsi_x_primary_types/` |
| Supp 2-S2 A layout (left vs right) on the MaxSI cut's SI | `figures/split_methods/maxsi_x_primary_types_mirror.py` | `NEURON_TABLE_SFC_FTR`, `SI_UPDATED_SFC_FTR` (the SFC cells and SI under either `SPLIT_METHOD`), `SI_COMPARISONS_FTR`, `NEURON_ANNOTATIONS_CSV` | `outputs/split_methods/maxsi_x_primary_types_mirror/` |
| Example cells: SFC split next to the MaxSI split | `figures/split_methods/skeleton_comparisons.py` | `SI_COMPARISONS_FTR`, `SYNAPSE_TABLE_RAW_FTR`, skeletons in `SWC_DIR` | `outputs/split_methods/skeleton_comparisons/` |
| The same for Kenyon cells (KCg-m) | `figures/split_methods/skeleton_comparisons_kc.py` | `NEURON_TABLE_SFC_FTR` (the SFC cells under either `SPLIT_METHOD`, like step 08), `SI_COMPARISONS_FTR`, `SYNAPSE_TABLE_RAW_FTR`, skeletons in `SWC_DIR` | `outputs/split_methods/skeleton_comparisons_kc/` |
| New Supp 1-S2: six example cells split four ways, and the SI of every candidate cut | `figures/split_methods/method_examples.py` (after steps 08 and 09 with all three cut methods; `PIPELINE_RUN = "rebuild"`, `SPLIT_METHOD = "sfc"`) | `SI_COMPARISONS_FTR`, `NEURON_TABLE_SFC_FTR`, `SI_UPDATED_FTR`, `MAXSI_LABEL_BATCH_DIR`, `MINFISHERP_LABEL_BATCH_DIR`, `MAXPHI_LABEL_BATCH_DIR`, `ALT_SPLIT_BATCH_DIR`, `SYNAPSE_TABLE_RAW_FTR`, skeletons in `SWC_DIR` | `outputs/split_methods/method_examples/` (`method_examples.svg`, `example_cells.csv`: the rule behind each cell) |
| New supplement (revision): the four splits compared. Left/right agreement of SI (ICC on the Supp 2-S2 A pairs), the splits against each other (Spearman, and the share of cells on the same side of SI 0.1), and MaxSI SI minus SFC SI. Written for the revision, not ported; runs after step 09 with all three methods | `figures/split_methods/four_methods_si_comparison.py` | `NEURON_TABLE_SFC_FTR`, `SI_UPDATED_SFC_FTR`, `MAXSI_SI_UPDATED_FTR`, `MINFISHERP_SI_UPDATED_FTR`, `MAXPHI_SI_UPDATED_FTR`, `SI_COMPARISONS_FTR` (a check), `NEURON_ANNOTATIONS_CSV` | `outputs/split_methods/four_methods_si_comparison/`: five SVGs and four CSVs |
| Text: "simple neurons (PC1 < 0.5), which are primarily optic neurons" | `figures/supporting/pc1_x_nt_sclass_pies.py` | `NEURON_TABLE_FTR`, `PCA_TABLE_FTR` | `outputs/supporting/pc1_x_nt_sclass_pies/` |

> Run `processing/build_raw_synapse_table.py` first: it writes `SYNAPSE_TABLE_RAW_FTR`. `SI_COMPARISONS_FTR` comes from step 08
> (`08_alternative_split_methods.py`, optional). With `ONLY_CELLS_WITH_SKELETONS = "auto"` (the default) the skeleton scripts use
> the authors' original cell selection when all of those cells' skeletons are on disk; if even one is missing they replace the
> whole selection with cells that have a skeleton (and print that they did), so the images are then not the original cells.
> The original cells need the full skeleton download.

> `processing/phi_threshold.py` saves `outputs/processing/phi_threshold/phi_threshold.svg`, which shows how it finds the Phi
> cutoff that matches SI = 0.1 (not a panel of the submitted paper).

## Figure 6 — Reciprocal motifs in different neural systems

One script, ported from the corresponding author's MATLAB script `FlyWire_AADD_specificCircuits.m` (not
included in this repository). It reads the **Princeton** tables by default (`DATA_SOURCE = 'princeton'`),
like every other figure of the revised paper: `NEURON_TABLE_FTR` and `SYNAPSE_TABLE_FTR` (and `SI_UPDATED_FTR`
if `SI_SOURCE = 'SI_updated'`). The submitted Figure 6 was made from the Buhmann tables; `DATA_SOURCE = 'buhmann'`
reproduces it: every Fig 6D box and whisker exactly, Fig 6A–C and the per-type bars of Supp 6-S1/S2 closely
(the Buhmann table's `primary_type` is not the Codex consolidated cell types the MATLAB script read), and the
Supp 6-S3 graphs with the same edges. This repository does not build the published Buhmann tables; it builds MaxSI versions of them
(`processing/buhmann_maxsi_labels.py`; see `docs/data_availability.md`). `FIG6_DATA_SOURCE=buhmann` in the environment
sets `DATA_SOURCE`; with `'buhmann'` the figures go to `outputs/fig6/aadd_specific_circuits_buhmann/` instead.

| Panel | Script | Input tables | Output path |
|-------|--------|-------------|-------------|
| Fig 6A–D | `figures/fig6/aadd_specific_circuits.py` | `CLASSIFICATION_CSV`, `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` (with `DATA_SOURCE = 'buhmann'`: `NEURON_TABLE_NONP_FTR`, `SYNAPSE_TABLE_NONP_FTR`) | `outputs/fig6/aadd_specific_circuits/SummaryPlot.svg` (its four panels, top to bottom, are the published 6B, 6A, 6C and 6D) |
| Supp 6-S1A, 6-S1B, 6-S2A | `figures/fig6/aadd_specific_circuits.py` | as above | `outputs/fig6/aadd_specific_circuits/ALPN_conn_SameType.svg`, `Visual projection_conn_SameType.svg`, `Central complex_conn_SameType.svg` |
| Supp 6-S3 | `figures/fig6/aadd_specific_circuits.py` | as above | `outputs/fig6/aadd_specific_circuits/ALPN_conn_BetweenTypes.svg`, `Visual projection_conn_BetweenTypes.svg`, `Central complex_conn_BetweenTypes.svg` |
| Fig 6E–H, Supp 6-S2B | No script | — | Example neurons and EM images |

> The script also saves, for each population, `<population>_SynapsesPerSameConnection.svg`, all three figures
> of the olfactory and Kenyon-cell populations (not panels of the submitted paper), and `PopulationSummary.csv`.
> With `DATA_SOURCE = 'princeton'` on the published SFC tables (`SI_SOURCE = 'neuron_table'`, the SI computed with self-synapses kept), almost no Kenyon cell has
> SI ≥ 0.1, so that population is skipped and writes no figures; on the MaxSI tables 1,327 Kenyon cells pass and it is drawn. `tools/check_fig6_logic.py` checks the script's
> calculations on a made-up connectome, without the Buhmann tables ([docs/tools.md](tools.md#checking-scripts-on-made-up-data)).

## Figure 7 — Postsynaptic terminals on multisynaptic boutons

The bouton and filopodia panels. The two MSB pipelines in `MSB-and-Post-on-MSB-pipeline/` find multi-synapse
boutons and the postsynapses near them, but they do not make any panel, and no script in this
repository writes `SYN_BOUTON_FTR`, `SYN_BOUTON_NEW_FTR` or the bouton tables (`BOUTON_TABLE_FTR`, `BOUTON_FILOPODIA_TABLE_FTR`,
`FILOPODIA_RECIPROCAL_FTR`, `FILOPODIA_FILTERED_FTR`, `BOUTON_INERTIA_FTR`), `NEURON_DATA_FTR` or `SYNAPSES_NO_SELF_FTR`, which these scripts read
(see `docs/data_availability.md`), except the MaxSI versions of `BOUTON_TABLE_FTR` and `BOUTON_FILOPODIA_TABLE_FTR`, which
`processing/bouton_maxsi_tables.py` writes under `PIPELINE_RUN = "rebuild"` and `SPLIT_METHOD = "maxsi"`. Under that
setting, with them and with `NEURON_DATA_FTR` read from `data/derived/` (`REBUILD_KEEPS_PUBLISHED`), `fig7f`, `fig7g` and
`fig7i` run, and after `fig7i` (which writes `NEURONS_NT_BWF_FTR`) so do `filopodia_x_mirror_neurons.py` and
`figures/fig5/reciprocal_fraction_x_bwf.py`; 7H and the four 7J scripts read no bouton table and run too. `fig7e` stops on
`SYNAPSES_NO_SELF_FTR` (and `FILOPODIA_FILTERED_FTR` after it), `fig7k` on `FILOPODIA_RECIPROCAL_FTR`, and `fig7s2a` and
`fig7s2b` on `BOUTON_INERTIA_FTR`, which have no rebuilt version. The `fig7*.py` scripts are ported from the scripts that drew those panels.

| Panel | Script | Input tables | Output path |
|-------|--------|-------------|-------------|
| Fig 7E | `figures/fig7/fig7e_nt_presynaptic_pies.py` | `NEURON_DATA_FTR`, `SYNAPSES_NO_SELF_FTR`, `BOUTON_TABLE_FTR`, `FILOPODIA_FILTERED_FTR` | `outputs/fig7/fig7e_nt_presynaptic_pies/pie_plot_nt_post_frac_and_bwf_3_newest.svg` |
| Fig 7F, Fig 7 Supp 4C | `figures/fig7/fig7f_filopodia_fraction_violins.py` | `BOUTON_TABLE_FTR` | `outputs/fig7/fig7f_filopodia_fraction_violins/super_fractions_split2_new(2).svg` (left half 7F, right half Supp 4C) |
| Fig 7G | `figures/fig7/fig7g_filopodia_fraction_x_si.py` | `BOUTON_TABLE_FTR`, `BOUTON_FILOPODIA_TABLE_FTR` | `outputs/fig7/fig7g_filopodia_fraction_x_si/` (left: `SI_vs_fraction_all_new.svg`; right: `SI_vs_post_on_bouton_ALL_AXON_DENDRITE_2_new.svg`) |
| Fig 7H | `figures/fig7/post_on_filo_ratio.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR`, `SYN_BOUTON_NEW_FTR` | `outputs/fig7/post_on_filo_ratio/` (the panel is `axon_dend_filo_ratio_by_superclass_lines.svg`; the other three SVGs are variants) |
| Fig 7I | `figures/fig7/fig7i_filopodia_fraction_x_nt.py` | `BOUTON_TABLE_FTR`, `BOUTON_FILOPODIA_TABLE_FTR`, `NEURON_DATA_FTR` | `outputs/fig7/fig7i_filopodia_fraction_x_nt/nt_type_frac_new2.svg`; writes `NEURONS_NT_BWF_FTR` |
| Fig 7J (input) | `figures/fig7/si_sim_baseline.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR`, `SYN_BOUTON_FTR` | `outputs/fig7/si_sim_baseline/`; writes `FILOPODIA_NODES_BASELINE_FTR` |
| Fig 7J (input) | `figures/fig7/si_sim_no_filopodia.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR`, `SYN_BOUTON_FTR` | `outputs/fig7/si_sim_no_filopodia/`; writes `FILOPODIA_NODES_NO_FILOPODIA_FTR` |
| Fig 7J (input) | `figures/fig7/si_sim_sampleout.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR`, `SYN_BOUTON_FTR` | `outputs/fig7/si_sim_sampleout/`; writes `FILOPODIA_NODES_SAMPLEOUT_FTR` |
| Fig 7J | `figures/fig7/si_comb_analysis.py` | `FILOPODIA_NODES_BASELINE_FTR`, `FILOPODIA_NODES_NO_FILOPODIA_FTR`, `FILOPODIA_NODES_SAMPLEOUT_FTR`, `NEURON_TABLE_FTR` | `outputs/fig7/si_comb_analysis/sim_results.svg` |
| Fig 7K | `figures/fig7/fig7k_reciprocal_composition.py` | `FILOPODIA_RECIPROCAL_FTR`, `NEURON_TABLE_FTR` | `outputs/fig7/fig7k_reciprocal_composition/super_classes_reciprocal_new.svg` |
| Fig 7 Supp 2A | `figures/fig7/fig7s2a_bouton_spread.py` | `BOUTON_INERTIA_FTR`, `CONSOLIDATED_CELL_TYPES_CSV` | `outputs/fig7/fig7s2a_bouton_spread/inertia_all.svg` |
| Fig 7 Supp 2B | `figures/fig7/fig7s2b_bouton_spread_by_type.py` | `BOUTON_INERTIA_FTR`, `CONSOLIDATED_CELL_TYPES_CSV`, `NEURON_DATA_FTR` | `outputs/fig7/fig7s2b_bouton_spread_by_type/inertia_grouped.svg` |
| Fig 7 Supp 4A/B | `figures/fig7/filopodia_x_mirror_neurons.py` | `NEURON_TABLE_FTR`, `NEURONS_NT_BWF_FTR`, `SI_UPDATED_FTR`, `NEURON_ANNOTATIONS_CSV` | `outputs/fig7/filopodia_x_mirror_neurons/` (A: `std_of_filopodia_fraction_within_groups_vs_between_princeton.svg`; B: `filopodia_frac_corr_sides_princeton_v2.svg`) |

> Run the three `si_sim_*.py` scripts before `si_comb_analysis.py`; it reads the three tables they write
> to `data/derived/filopodia/`. `fig7i_filopodia_fraction_x_nt.py` writes `NEURONS_NT_BWF_FTR`, which
> `filopodia_x_mirror_neurons.py` and `figures/fig5/reciprocal_fraction_x_bwf.py` read, so run it first.

> The `fig7*.py` scripts draw their own neuron sets from the bouton tables, which differ from the other
> figures': no four-super-class filter in 7G's 'All' lines or anywhere in 7I, one value per neuron and compartment
> in 7F's 'All' violins, and the bouton tables carry the neuron table's SI. 7D, Supp 7-S1, Supp 7-S3 and
> Supp 7-S4 D-F have no script. The seven `fig7*.py` scripts are made from the authors' originals by
> `tools/make_fig7_ports.py` (change them there, not in the scripts); `tools/check_fig7_ports.py` checks that
> each draws what its original draws, and `tools/check_fig7_tables.py` checks their input tables.

> The `si_sim_*.py` scripts recompute SI on the existing axon/dendrite labels; they do not re-split any neuron.
> `post_on_filo_ratio.py`, `si_comb_analysis.py` and the three `si_sim_*.py` scripts load the whole synapse table
> (`SYNAPSE_TABLE_FTR`, about 76 million rows), so they need a machine with plenty of memory.
> Fig 7 neuron sets differ from the other figures: `post_on_filo_ratio.py` and `si_comb_analysis.py` apply the
> four-super-class filter but not the `dropna` on `axon_correct`/`dend_correct`/`primary_type`, and `post_on_filo_ratio.py`
> takes the compartment from the second letter of `comp`, so synapses from linker segments (LA, LD) are included.
> Their pie and correctness plots are not panels. The saved 7J bar compares baseline with no-filopodia only; the
> sample-out table is read by `si_comb_analysis.py` and used only in a figure it shows and does not save.
> `si_comb_analysis.py` keeps the cells with SI >= 0.1 (`NEURON_TABLE_FTR`'s SI), as the 7J legend says; the submitted
> 7J has no SI filter, so on the published tables its bar differs from the submitted one. The flag tables (`SYN_BOUTON_FTR`,
> `SYN_BOUTON_NEW_FTR`) cover the published synapse table's synapses only and don't depend on the split, so they are
> read under every `SPLIT_METHOD`; 7H and the three `si_sim_*.py` scripts leave out synapses without a flag (on
> `data/derived_rebuild_maxsi`, the 760,723 synapses with an end in one of the 52 cells the published table lacks).
