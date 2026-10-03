# Figure-to-Script Map

This document maps each manuscript figure panel to the script that generates it, the input
tables it requires, and the output directory where figures are saved.

All figure scripts import `config.py` (paths) and `methods/methods_all.py` (utilities).
Figures are saved under `outputs/` (`outputs/figN/` for the figure folders; the other folders and `processing/phi_threshold.py` use their own names, given below); a few scripts also write tables to `data/derived/` (shown as "writes" below). None of these are committed to this repository.

---

## Figure 1 — Synaptic polarity and compartment canonicality

Shows that neurons separate synaptic inputs to dendrites and outputs to axons. Panels include
the SI distribution, axon/dendrite split examples, and canonical vs mixed neuron classes.

| Panel | Script | Input tables | Output path |
|-------|--------|-------------|-------------|
| Fig 1C | `figures/fig1/create_split_axon_dendrite_princeton.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_RAW_FTR` (from `processing/build_raw_synapse_table.py`; which version is set by `RAW_TABLE_KEEP_SELF_SYNAPSES`), skeletons in `SWC_DIR`, a FlyWire token for the links | `outputs/fig3/pc1_example/` |
| Fig 1E | `figures/fig1/canonicality_axon_dend.py` | `NEURON_TABLE_FTR` | `outputs/fig1/canonicality_axon_dend/` |
| Fig 1G | `figures/fig1/si_x_canonicality.py` | `NEURON_TABLE_FTR` | `outputs/fig1/si_x_canonicality/` |
| Fig 1 Supp D | `figures/fig1/canonicality_violin_intrinsic_sclass.py` | `NEURON_TABLE_FTR` | `outputs/fig1/canonicality_violin_intrinsic_sclass/` |
| Fig 1 Supp D | `figures/fig1/canonicality_violin_non_intrinsic_sclass.py` | `NEURON_TABLE_FTR` | `outputs/fig1/canonicality_violin_non_intrinsic_sclass/` |
| Fig 1 Supp E | `figures/fig1/fig1supp_e_sclass_canonicality_corr.py` | `NEURON_TABLE_FTR` | `outputs/fig1/sclass_canonicality_corr/` |

> `create_split_axon_dendrite_princeton.py` writes all its outputs to `outputs/fig3/pc1_example/`
> (not to `outputs/fig1/`): an example neuron used as a visual reference in both Fig 1 and Fig 3.

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
| Fig 2E | `figures/fig2/feat_rf_and_lr_models.py` | `NEURON_TABLE_FTR` | `outputs/fig2/feat_rf_and_lr_models/` |
| Fig 2F | `figures/fig2/si_x_pca.py` | `NEURON_TABLE_FTR`, `SI_UPDATED_FTR`, `SYNAPSE_TABLE_FTR` (loaded, but nothing that runs uses it) | `outputs/fig2/SI_x_PCA/` |
| Fig 2G | `figures/fig2/si_x_pca.py` | `NEURON_TABLE_FTR`, `SI_UPDATED_FTR`, `SYNAPSE_TABLE_FTR` (loaded, but nothing that runs uses it) | `outputs/fig2/SI_x_PCA/` |
| Fig 2 Supp 1A | `figures/fig2/si_x_twigs.py` | `NEURON_TABLE_FTR`, `NEUROPIL_SYNAPSE_CSV` | `outputs/fig2/SI_x_twigs/` |
| Fig 2 Supp 1B/C | `figures/fig2/si_comparisons.py` | `NEURON_TABLE_FTR`, `NEURON_TABLE_NONP_FTR`, `SYNAPSE_TABLE_FTR`, `SYNAPSE_TABLE_NONP_FTR` | `outputs/fig2/SI_comparisons/` |
| Supp 2-S2 A | `figures/fig2/si_x_primary_types_mirror.py` | `NEURON_ANNOTATIONS_CSV`, `NEURON_TABLE_FTR`, `SI_UPDATED_FTR` | `outputs/fig2/SI_x_primary_types_mirror/` |
| Supp 2-S2 D | `figures/fig2/si_x_primary_types.py` | `NEURON_TABLE_FTR` | `outputs/fig2/SI_x_primary_types/` |
| Not a panel of the submitted paper | `figures/fig2/pca_on_feat.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` (loaded, but nothing that runs uses it) | `outputs/fig2/pca_on_feat/` |

> Fig 2A requires larval data files — see [data_availability.md](data_availability.md).
> Fig 2 Supp 1A requires the neuropil synapse CSV — see [data_availability.md](data_availability.md).
> In `outputs/fig2/SI_x_PCA/`, Fig 2F is `weights.svg` (line 100) and Fig 2G is
> `optic_visual_projection_PC1_SI_CDF_vertical_v2.svg` (last written at line 910; lines 614 and 753 write the same
> name first; only the last write remains). The script also saves variants, saves bar plots in `colored_bars/`, and
> writes `full_info.ftr` to `outputs/fig2/SI_x_PCA/`.
> Supp 2-S2 A is `SI_corr_sides_princeton_v2.svg` and reads the SI from `SI_UPDATED_FTR`; Supp 2-S2 D is
> `std_of_si_within_groups_vs_between_princeton.svg` and uses the SI in `NEURON_TABLE_FTR`. `pca_on_feat.py` saves
> signed PC1 and PC2 contributions and violin plots.

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
| Fig 3F (AD content) | `figures/fig3/fig3supp_ad_content_si.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig3/supp_ad_content_si/` |
| Fig 3F (syntype x identity) | `figures/fig3/syntype_x_identity.py` | `CLASSIFICATION_CSV`, `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig3/syntype_x_identity/` |
| Fig 3G | `figures/fig3/nt_syn_type_same_not_same.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig3/nt_syn_type_same_not_same/` |
| Fig 3H | `figures/fig3/syntype_x_strength_x_identity.py` | `CONNECTIONS_TABLE_FTR`, `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig3/syntype_x_strength_x_identity/` |
| Fig 3I | `figures/fig3/syn_type_prob_based_on_other_syn_type_princeton.py` | `CONNECTIONS_TABLE_FTR`, `NEURON_TABLE_FTR` | `outputs/fig3/syn_type_prob/` |
| Fig 3J | `figures/fig3/syntype_x_features.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig3/syntype_x_features/` |
| Fig 3 Supp 1A (Buhmann curve) / 2A/B | `figures/fig3/si_x_correct_percent_per_compartment_buhmann.py` | `NEURON_TABLE_NONP_FTR`, `SYNAPSE_TABLE_NONP_FTR` | `outputs/fig3/si_x_correct_compartment_buhmann/` |
| Fig 3 Supp 2C | `figures/fig3/si_x_npil_x_synapse_detection.py` | `SYNAPSE_TABLE_FTR`, `SYNAPSE_TABLE_NONP_FTR` | `outputs/fig3/si_x_npil_x_synapse_detection/` |

> Fig 3E is `synaptic_type_npils2_v2.svg` (line 428 of `fig3d_si_compartment_correct.py`), which marks a compartment
> mixed by the same percentage cutoffs as Fig 3D. With lines 19-22 switched off, so that the script keeps the SI in
> `NEURON_TABLE_FTR`, it draws the panel's bars exactly (the published figure also lists three neighbouring pairs of
> neuropil labels in the opposite order: MB CA/BU, MB PED/FB and MB VL/AL). As committed, those lines replace that SI
> with `SI_UPDATED_FTR`, which changes the cutoffs, the bars and their order. `fig3e_syntype_composition.py` instead
> marks each end of a synapse mixed when that cell's older SI (`SI_pre`/`SI_post` in the synapse table) is below 0.1,
> and as committed it stops at line 40 (`KeyError: ['neuron']`) before any figure is saved.

---

## Figure 4 — Synapse type vs PC1 morphological gradient

Tests whether the morphological gradient (PC1) predicts synapse-type composition. Compares
multiple classification models.

| Panel | Script | Input tables | Output path |
|-------|--------|-------------|-------------|
| (table for Fig 4A / 4C) | `figures/fig4/build_pc1_table.py` | `CONNECTIONS_TABLE_FTR`, `NEURON_TABLE_FTR`, `PCA_TABLE_FTR` | writes `PC1_TABLE_CSV` (`data/derived/PC1_table.csv`); run before `syntype_x_pc1.py` |
| Fig 4A / 4C / Supp | `figures/fig4/syntype_x_pc1.py` | `NEURON_TABLE_FTR`, `PC1_TABLE_CSV`, `PCA_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig4/syntype_x_pc1/`, where it also saves and reloads the model `rf_pc1_model.joblib`; writes `PC1_TABLE_PREDICTIONS_CSV` |
| Fig 4B | `figures/fig4/models_comparison.py` | `CONNECTIONS_TABLE_FTR`, `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig4/models_comparison/` |
| Not settled (see note) | `figures/fig4/syntype_x_pc1_simple_model.py` | `NEURON_TABLE_FTR`, `PCA_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig4/syntype_x_pc1_simple_model/` |

> `syntype_x_pc1_simple_model.py` saves a feature-importance plot, a confusion matrix and a decision surface
> for a model on PC1, the same kinds of plot as Fig 4A and 4D. It reads the PCA table step 07 writes
> (`PCA_TABLE_FTR`), like `syntype_x_pc1.py`. The authors' earlier copies of its plots were made from an older
> PCA table that no longer exists, and a rerun differs from them (by up to 3.4 points in a cell of the
> confusion matrix; accuracy 40.5% against 39.6%). That is more than the seed or the sign of PC1 changes, and fits
> the different table, but the older table cannot be checked. A rerun can be checked against an independent rebuild of
> the model with `tools/check_simple_pc1_model.py` (README, Tools). `syntype_x_pc1.py` draws a decision surface too, but its save line (586) is
> commented out, so which script made the published Fig 4D is not settled. Fig 4 Supp 1 holds only skeleton
> images of the Fig 4D neurons.

---

## Figure 5 — Reciprocal connectivity

Analyses the fraction of reciprocal connections across neuron types. Tests whether reciprocally
connected pairs use predictable synapse-type patterns. Builds a random-forest model for
reciprocity prediction.

| Panel | Script | Input tables | Output path |
|-------|--------|-------------|-------------|
| Fig 5A-C / Supp | `figures/fig5/reciprocal_fraction.py` | `CONNECTIONS_TABLE_FTR`, `NEURON_TABLE_FTR` | `outputs/fig5/reciprocal_fraction/`; writes `RECI_PROP_FTR` |
| Not a panel of the submitted paper (per super-class correlations of `filopodia_fraction` with the reciprocal ratio and with SI) | `figures/fig5/reciprocal_fraction_x_bwf.py` | `CONNECTIONS_TABLE_FTR`, `NEURONS_NT_BWF_FTR`, `NEURON_TABLE_FTR` | `outputs/fig5/reciprocal_fraction_x_bwf/` |
| Fig 5D | `figures/fig5/fig5d_chi_reci_identity.py` | `CONNECTIONS_TABLE_FTR`, `CONNECTIONS_TABLE_NONP_FTR`, `FULL_RECI_CONNECTIONS_FTR`, `NEURON_TABLE_FTR`, `NEURON_TABLE_NONP_FTR` | `outputs/fig5/chi_reci_identity/` |
| Fig 5E, 5F | No script: example reciprocal pairs (tables of synapse counts, an EM cross-section) | — | — |
| Fig 5G (All) | `figures/fig5/syn_type_x_reci_x_dominance.py` | `FULL_RECI_CONNECTIONS_FTR`, `NEURON_TABLE_FTR` | `outputs/fig5/syn_type_x_reci_x_dominance/` |
| Fig 5G (Ipsilateral, Contralateral) | `figures/fig5/syn_type_x_reci_x_dominance_x_sides.py` | `CLASSIFICATION_CSV`, `FULL_RECI_CONNECTIONS_FTR`, `NEURON_TABLE_FTR` | `outputs/fig5/syn_type_x_reci_x_dominance_x_sides/` |
| Fig 5H | `figures/fig5/syn_type_x_reci_x_dominance.py` | `FULL_RECI_CONNECTIONS_FTR`, `NEURON_TABLE_FTR` | `outputs/fig5/syn_type_x_reci_x_dominance/` |
| Fig 5I | `figures/fig5/syn_type_x_reci_x_npil.py` | `CONNECTIONS_TABLE_FTR`, `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig5/syn_type_x_reci_x_npil/` |
| Not a panel of the submitted paper (Supp 5-S1 has only panels A and B, both examples) | `figures/fig5/reciprocal_fraction_model.py` | `NEURON_TABLE_FTR`, `RECI_PROP_FTR` | `outputs/fig5/reciprocal_fraction_model/`; writes `RF_MODEL_PKL` |
| Not a panel of the submitted paper | `figures/fig5/syn_type_strength_identity.py` | `CONNECTIONS_TABLE_FTR`, `FULL_RECI_CONNECTIONS_FTR`, `NEURON_TABLE_FTR` | `outputs/fig5/syn_type_strength_identity/` |

> Fig 5G and 5H: in `syn_type_x_reci_x_dominance/`, 5G (All) is `same_type_percentage_princeton.svg` (line 219) and 5H is
> `not_same_type_percentage_princeton.svg` (line 247). In `syn_type_x_reci_x_dominance_x_sides/`, 5G Ipsilateral is
> `same_type_percentage_same_side_totalnorm_princeton.svg` (line 271) and 5G Contralateral is
> `not_same__side_type_percentage_totalnorm_princeton.svg` (line 312; same-type pairs on opposite sides, despite the
> name). That script's other two SVGs, which reuse the base script's file names, are not panels.

> `fig5d_chi_reci_identity.py` reads the Buhmann tables (`*_NONP_FTR`) from line 554, after all its figures are
> saved; nothing from them is saved or printed. See the Buhmann section of [data_availability.md](data_availability.md).

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
| Supp 2-S2 D layout on the MaxSI cut's SI | `figures/split_methods/maxsi_x_primary_types.py` | `NEURON_TABLE_FTR`, `SI_UPDATED_FTR`, `SI_COMPARISONS_FTR` | `outputs/split_methods/maxsi_x_primary_types/` |
| Supp 2-S2 A layout (left vs right) on the MaxSI cut's SI | `figures/split_methods/maxsi_x_primary_types_mirror.py` | `NEURON_TABLE_FTR`, `SI_UPDATED_FTR`, `SI_COMPARISONS_FTR`, `NEURON_ANNOTATIONS_CSV` | `outputs/split_methods/maxsi_x_primary_types_mirror/` |
| Example cells: SFC split next to the MaxSI split | `figures/split_methods/skeleton_comparisons.py` | `SI_COMPARISONS_FTR`, `SYNAPSE_TABLE_RAW_FTR`, skeletons in `SWC_DIR` | `outputs/split_methods/skeleton_comparisons/` |
| The same for Kenyon cells (KCg-m) | `figures/split_methods/skeleton_comparisons_kc.py` | `NEURON_TABLE_FTR`, `SI_COMPARISONS_FTR`, `SYNAPSE_TABLE_RAW_FTR`, skeletons in `SWC_DIR` | `outputs/split_methods/skeleton_comparisons_kc/` |
| Text: "simple neurons (PC1 < 0.5), which are primarily optic neurons" | `figures/supporting/pc1_x_nt_sclass_pies.py` | `NEURON_TABLE_FTR`, `PCA_TABLE_FTR` | `outputs/supporting/pc1_x_nt_sclass_pies/` |

> Run `processing/build_raw_synapse_table.py` first: it writes `SYNAPSE_TABLE_RAW_FTR`. `SI_COMPARISONS_FTR` comes from step 08
> (`08_alternative_split_methods.py`, optional). With `ONLY_CELLS_WITH_SKELETONS = "auto"` (the default) the skeleton scripts use
> the authors' original cell selection when all of those cells' skeletons are on disk; if even one is missing they replace the
> whole selection with cells that have a skeleton (and print that they did), so the images are then not the original cells.
> The original cells need the full skeleton download.

> `processing/phi_threshold.py` saves `outputs/phi_threshold/phi_threshold.svg`, which shows how it finds the Phi
> cutoff that matches SI = 0.1 (not a panel of the submitted paper).

## Figure 6

There are no panel scripts for Figure 6 in this repository yet.

## Figure 7 — Postsynaptic terminals on multisynaptic boutons

The filopodia panels. The two MSB pipelines in `MSB-and-Post-on-MSB-pipeline/` find multi-synapse
boutons and the postsynapses near them, but they do not make any panel, and no script in this
repository writes `SYN_BOUTON_FTR` or `NEURONS_NT_BWF_FTR`, which these scripts read.

| Panel | Script | Input tables | Output path |
|-------|--------|-------------|-------------|
| Fig 7H | `figures/fig7/post_on_filo_ratio.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR`, `SYN_BOUTON_FTR` | `outputs/fig7/post_on_filo_ratio/` (the panel is `axon_dend_filo_ratio_by_superclass_lines.svg`; the other three SVGs are variants) |
| Fig 7J (input) | `figures/fig7/si_sim_baseline.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR`, `SYN_BOUTON_FTR` | `outputs/fig7/si_sim_baseline/`; writes `FILOPODIA_NODES_BASELINE_FTR` |
| Fig 7J (input) | `figures/fig7/si_sim_no_filopodia.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR`, `SYN_BOUTON_FTR` | `outputs/fig7/si_sim_no_filopodia/`; writes `FILOPODIA_NODES_NO_FILOPODIA_FTR` |
| Fig 7J (input) | `figures/fig7/si_sim_sampleout.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR`, `SYN_BOUTON_FTR` | `outputs/fig7/si_sim_sampleout/`; writes `FILOPODIA_NODES_SAMPLEOUT_FTR` |
| Fig 7J | `figures/fig7/si_comb_analysis.py` | `FILOPODIA_NODES_BASELINE_FTR`, `FILOPODIA_NODES_NO_FILOPODIA_FTR`, `FILOPODIA_NODES_SAMPLEOUT_FTR`, `NEURON_TABLE_FTR` | `outputs/fig7/si_comb_analysis/sim_results.svg` |
| Fig 7 Supp 4A/B | `figures/fig7/filopodia_x_mirror_neurons.py` | `NEURON_TABLE_FTR`, `NEURONS_NT_BWF_FTR`, `SI_UPDATED_FTR`, `NEURON_ANNOTATIONS_CSV` | `outputs/fig7/filopodia_x_mirror_neurons/` (A: `std_of_filopodia_fraction_within_groups_vs_between_princeton.svg`; B: `filopodia_frac_corr_sides_princeton_v2.svg`) |

> Run the three `si_sim_*.py` scripts before `si_comb_analysis.py`; it reads the three tables they write
> to `data/derived/filopodia/`.

> The `si_sim_*.py` scripts recompute SI on the existing axon/dendrite labels; they do not re-split any neuron.
> `post_on_filo_ratio.py`, `si_comb_analysis.py` and the three `si_sim_*.py` scripts load the whole synapse table
> (`SYNAPSE_TABLE_FTR`, about 76 million rows), so they need a machine with plenty of memory.
> Fig 7 neuron sets differ from the other figures: `post_on_filo_ratio.py` and `si_comb_analysis.py` apply the
> four-super-class filter but not the `dropna` on `axon_correct`/`dend_correct`/`primary_type`, and `post_on_filo_ratio.py`
> takes the compartment from the second letter of `comp`, so synapses from linker segments (LA, LD) are included.
> Their pie and correctness plots are not panels. The saved 7J bar compares baseline with no-filopodia only; the
> sample-out table is read by `si_comb_analysis.py` and used only in a figure it shows and does not save.
