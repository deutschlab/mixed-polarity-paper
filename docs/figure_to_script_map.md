# Figure-to-Script Map

This document maps each manuscript figure panel to the script that generates it, the input
tables it requires, and the output directory where figures are saved.

All figure scripts import `config.py` (paths) and `methods/methods_all.py` (utilities).
All outputs are saved to `outputs/figN/` and are not committed to this repository.

---

## Figure 1 — Synaptic polarity and compartment canonicality

Shows that neurons separate synaptic inputs to dendrites and outputs to axons. Panels include
the SI distribution, axon/dendrite split examples, and canonical vs mixed neuron classes.

| Panel | Script | Input tables | Output path |
|-------|--------|-------------|-------------|
| Fig 1C | `figures/fig1/create_split_axon_dendrite_princeton.py` | `SYNAPSE_TABLE_RAW_FTR`, `NEURON_TABLE_FTR` | `outputs/fig1/split_axon_dendrite/` |
| Fig 1E | `figures/fig1/canonicality_axon_dend.py` | `NEURON_TABLE_FTR` | `outputs/fig1/canonicality/` |
| Fig 1G | `figures/fig1/si_x_canonicality.py` | `NEURON_TABLE_FTR` | `outputs/fig1/si_x_canonicality/` |
| Fig 1 Supp D | `figures/fig1/canonicality_violin_intrinsic_sclass.py` | `NEURON_TABLE_FTR` | `outputs/fig1/canonicality_violin/` |
| Fig 1 Supp D | `figures/fig1/canonicality_violin_non_intrinsic_sclass.py` | `NEURON_TABLE_FTR` | `outputs/fig1/canonicality_violin/` |
| Fig 1 Supp E | `figures/fig1/fig1supp_e_sclass_canonicality_corr.py` | `NEURON_TABLE_FTR` | `outputs/fig1/fig1supp_e/` |

> `create_split_axon_dendrite_princeton.py` also writes some outputs to `outputs/fig3/pc1_example/`,
> providing an example neuron used as a visual reference in both Fig 1 and Fig 3.

---

## Figure 2 — SI variation and morphological models

Shows how SI varies across neuron super-classes, neurotransmitter types, and primary types.
Builds random-forest and logistic-regression models predicting SI from morphological features.
Includes an adult vs larva SI comparison and PCA of morphological features.

| Panel | Script | Input tables | Output path |
|-------|--------|-------------|-------------|
| Fig 2A | `figures/fig2/fig2a_si_cdf_adult_larva.py` | `NEURON_TABLE_FTR`, `SI_UPDATED_FTR`, `LARVA_SYNAPSES_FTR`, `LARVA_SI_FTR` | `outputs/fig2/fig2a/` |
| Fig 2B | `figures/fig2/si_x_sclass.py` | `NEURON_TABLE_FTR` | `outputs/fig2/si_x_sclass/` |
| Fig 2C | `figures/fig2/si_x_nt.py` | `NEURON_TABLE_FTR` | `outputs/fig2/si_x_nt/` |
| Fig 2D | `figures/fig2/feat_x_si_x_sclass_corr.py` | `NEURON_TABLE_FTR`, `SI_UPDATED_FTR` | `outputs/fig2/feat_corr/` |
| Fig 2E | `figures/fig2/feat_rf_and_lr_models.py` | `NEURON_TABLE_FTR` | `outputs/fig2/feat_models/` |
| Fig 2F | `figures/fig2/pca_on_feat.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig2/pca_on_feat/` |
| Fig 2G | `figures/fig2/si_x_pca.py` | `NEURON_TABLE_FTR`, `SI_UPDATED_FTR` | `outputs/fig2/si_x_pca/` |
| Fig 2 Supp 1A | `figures/fig2/si_x_twigs.py` | `NEURON_TABLE_FTR`, `NEUROPIL_SYNAPSE_CSV` | `outputs/fig2/si_x_twigs/` |
| Fig 2 Supp 1B/C | `figures/fig2/si_comparisons.py` | `NEURON_TABLE_FTR`, `NEURON_TABLE_NONP_FTR`, `SYNAPSE_TABLE_NONP_FTR` | `outputs/fig2/si_comparisons/` |
| Fig 2 Supp 2A | `figures/fig2/si_x_primary_types.py` | `NEURON_TABLE_FTR` | `outputs/fig2/si_x_primary_types/` |
| Fig 2 Supp 2B | `figures/fig2/si_x_primary_types_mirror.py` | `NEURON_TABLE_FTR`, `NEURON_ANNOTATIONS_CSV` | `outputs/fig2/si_x_primary_types_mirror/` |

> Fig 2A requires larval data files — see [data_availability.md](data_availability.md).
> Fig 2 Supp 1A requires the neuropil synapse CSV — see [data_availability.md](data_availability.md).

---

## Figure 3 — Synapse-type composition, identity, and classifiers

Shows how the four synapse compartment types (AA/AD/DA/DD) are distributed across neuron types.
Builds classifiers predicting synapse type from morphological and connectivity features. Tests
whether neurons within the same type use the same synapse-type pattern.

| Panel | Script | Input tables | Output path |
|-------|--------|-------------|-------------|
| Fig 3A | `figures/fig3/si_x_neuron_types.py` | `NEURON_TABLE_FTR`, `NEURON_ANNOTATIONS_CSV` | `outputs/fig3/si_x_neuron_types/` |
| Fig 3C | `figures/fig3/fig3c_mixed_example_sfc.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig3/mixed_example/` |
| Fig 3D | `figures/fig3/fig3d_si_compartment_correct.py` | `NEURON_TABLE_FTR`, `SI_UPDATED_FTR` | `outputs/fig3/fig3d/` |
| Fig 3D (AD content, mixed) | `figures/fig3/fig3supp_ad_content_si_mixed.py` | `NEURON_TABLE_FTR`, `SI_UPDATED_FTR` | `outputs/fig3/fig3supp_ad_mixed/` |
| Fig 3E | `figures/fig3/fig3e_syntype_composition.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig3/fig3e/` |
| Fig 3F (AD content) | `figures/fig3/fig3supp_ad_content_si.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig3/fig3supp_ad/` |
| Fig 3F (syntype x identity) | `figures/fig3/syntype_x_identity.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR`, `CLASSIFICATION_CSV` | `outputs/fig3/syntype_x_identity/` |
| Fig 3G | `figures/fig3/nt_syn_type_same_not_same.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig3/nt_syn_type/` |
| Fig 3H | `figures/fig3/syntype_x_strength_x_identity.py` | `NEURON_TABLE_FTR`, `CONNECTIONS_TABLE_FTR` | `outputs/fig3/syntype_x_strength_x_identity/` |
| Fig 3I | `figures/fig3/syn_type_prob_based_on_other_syn_type_princeton.py` | `NEURON_TABLE_FTR`, `CONNECTIONS_TABLE_FTR` | `outputs/fig3/syn_type_prob/` |
| Fig 3J | `figures/fig3/syntype_x_features.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig3/syntype_x_features/` |
| Fig 3 Supp 1A / 2A/B | `figures/fig3/si_x_correct_percent_per_compartment_buhmann.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_NONP_FTR`, `NEURON_TABLE_NONP_FTR` | `outputs/fig3/si_x_correct_buhmann/` |
| Fig 3 Supp 2C | `figures/fig3/si_x_npil_x_synapse_detection.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR` | `outputs/fig3/si_x_npil/` |

---

## Figure 4 — Synapse type vs PC1 morphological gradient

Tests whether the morphological gradient (PC1) predicts synapse-type composition. Compares
multiple classification models.

| Panel | Script | Input tables | Output path |
|-------|--------|-------------|-------------|
| Fig 4A / 4C / Supp | `figures/fig4/syntype_x_pc1.py` | `NEURON_TABLE_FTR`, `CONNECTIONS_TABLE_FTR`, `PCA_TABLE_FTR`, `PC1_TABLE_CSV`, `PC1_TABLE_PREDICTIONS_CSV` | `outputs/fig4/syntype_x_pc1/` |
| Fig 4B | `figures/fig4/models_comparison.py` | `NEURON_TABLE_FTR`, `SYNAPSE_TABLE_FTR`, `CONNECTIONS_TABLE_FTR` | `outputs/fig4/models_comparison/` |
| Fig 4 Supp 1B-D | `figures/fig4/syntype_x_pc1_simple_model.py` | `NEURON_TABLE_FTR`, `CONNECTIONS_TABLE_FTR`, `PCA_TABLE_NONP_FTR` | `outputs/fig4/simple_model/` |

---

## Figure 5 — Reciprocal connectivity

Analyses the fraction of reciprocal connections across neuron types. Tests whether reciprocally
connected pairs use predictable synapse-type patterns. Builds a random-forest model for
reciprocity prediction.

| Panel | Script | Input tables | Output path |
|-------|--------|-------------|-------------|
| Fig 5A-C / Supp | `figures/fig5/reciprocal_fraction.py` | `CONNECTIONS_TABLE_FTR`, `NEURON_TABLE_FTR` | `outputs/fig5/reciprocal_fraction/` |
| Fig 5A (BWF variant) | `figures/fig5/reciprocal_fraction_x_bwf.py` | `CONNECTIONS_TABLE_FTR`, `NEURON_TABLE_FTR`, `NEURONS_NT_BWF_FTR` | `outputs/fig5/reciprocal_fraction_bwf/` |
| Fig 5D / Supp 1B | `figures/fig5/fig5d_chi_reci_identity.py` | `CONNECTIONS_TABLE_FTR`, `FULL_RECI_CONNECTIONS_FTR`, `NEURON_TABLE_FTR`, `NEURON_TABLE_NONP_FTR`, `CONNECTIONS_TABLE_NONP_FTR` | `outputs/fig5/fig5d/` |
| Fig 5E | `figures/fig5/syn_type_strength_identity.py` | `CONNECTIONS_TABLE_FTR`, `NEURON_TABLE_FTR` | `outputs/fig5/syn_type_strength/` |
| Fig 5H | `figures/fig5/syn_type_x_reci_x_dominance_x_sides.py` | `CONNECTIONS_TABLE_FTR`, `NEURON_TABLE_FTR` | `outputs/fig5/reci_x_dominance_sides/` |
| Fig 5H (base) | `figures/fig5/syn_type_x_reci_x_dominance.py` | `CONNECTIONS_TABLE_FTR`, `NEURON_TABLE_FTR` | `outputs/fig5/reci_x_dominance/` |
| Fig 5I | `figures/fig5/syn_type_x_reci_x_npil.py` | `CONNECTIONS_TABLE_FTR`, `NEURON_TABLE_FTR` | `outputs/fig5/reci_x_npil/` |
| Fig 5 Supp 1C/D | `figures/fig5/reciprocal_fraction_model.py` | `RECI_PROP_FTR`, `NEURON_TABLE_FTR` | `outputs/fig5/reci_model/` |

> `reciprocal_fraction.py` also generates `RECI_PROP_FTR` as a side output, which is required
> by `reciprocal_fraction_model.py`. Run `reciprocal_fraction.py` before `reciprocal_fraction_model.py`.

> `reciprocal_fraction_model.py` also generates `RF_MODEL_PKL`
> (`data/intermediate/reciprocity/models/final_random_forest_model_princeton.pkl`).
