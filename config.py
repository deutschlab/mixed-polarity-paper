# config.py
# Central path configuration for the fafb783-synapse-polarity pipeline.
#
# USAGE: Import at the top of any processing or figure script:
#   from config import RAW_DATA_DIR, DERIVED_DATA_DIR, SYNAPSE_TABLE_FTR, OUTPUT_DIR
#
# All paths resolve relative to the repository root so the pipeline is
# portable across machines — no hardcoded absolute paths in other scripts.

from pathlib import Path

# Repository root (resolved from this file's location)
ROOT = Path(__file__).parent.resolve()

# ── Data directories ────────────────────────────────────────────────────────
RAW_DATA_DIR      = ROOT / "data" / "raw"
LARVA_DATA_DIR    = ROOT / "data" / "larva"
DERIVED_DATA_DIR  = ROOT / "data" / "derived"
INTERMEDIATE_DIR  = ROOT / "data" / "intermediate"

# SWC skeleton files (one subfolder per neuron batch)
SWC_DIR = RAW_DATA_DIR / "swc" / "783"

# Sensory layer rank feather files
RANKS_DIR = RAW_DATA_DIR / "ranks"

# Methods library (shared utilities)
METHODS_DIR = ROOT / "methods"

# Output figure directory
OUTPUT_DIR = ROOT / "outputs"

# ── Key raw input files ──────────────────────────────────────────────────────
PRINCETON_SYNAPSE_CSV  = RAW_DATA_DIR / "fafb_v783_princeton_synapse_table.csv"
NEURON_ANNOTATIONS_CSV = RAW_DATA_DIR / "Supplemental_file1_neuron_annotations.csv"
CELL_STATS_CSV         = RAW_DATA_DIR / "cell_stats.csv"

# ── Key derived table files ──────────────────────────────────────────────────
SYNAPSE_TABLE_FTR         = DERIVED_DATA_DIR / "synapses_783_article_princeton.ftr"
SYNAPSE_TABLE_RAW_FTR     = DERIVED_DATA_DIR / "synapses_783_article_princeton_raw.ftr"
NEURON_TABLE_FTR          = DERIVED_DATA_DIR / "neuron_data_full_article_princeton.ftr"
CONNECTIONS_TABLE_FTR     = DERIVED_DATA_DIR / (
    "connections_by_syn_type_reciprocal_types_filtered_article_princeton.ftr"
)
PCA_TABLE_FTR             = DERIVED_DATA_DIR / "neurons_pca_princeton.ftr"
PCA_TABLE_NONP_FTR        = DERIVED_DATA_DIR / "neurons_pca.ftr"
SI_UPDATED_FTR            = DERIVED_DATA_DIR / "SI_updated.ftr"
SWC_DATA_FTR              = DERIVED_DATA_DIR / "swc_data.ftr"
RECI_PROP_FTR             = DERIVED_DATA_DIR / "reci_prop_princeton.ftr"
FULL_RECI_CONNECTIONS_FTR = DERIVED_DATA_DIR / (
    "full_reci_connections_article_filtered_princeton.ftr"
)

# ── Convenience aliases ──────────────────────────────────────────────────────
PROJECT_ROOT          = ROOT                  # alias
DATA_DIR              = ROOT / "data"
INTERMEDIATE_DATA_DIR = INTERMEDIATE_DIR      # alias
PROCESSED_DATA_DIR    = DERIVED_DATA_DIR      # alias
RESULTS_DIR           = OUTPUT_DIR            # alias  (figures + results share outputs/)
FIGURES_DIR           = OUTPUT_DIR            # alias
DOCS_DIR              = ROOT / "docs"

# ── Intermediate processing subdirectories ───────────────────────────────────
PROCESSED_SWC_DIR         = INTERMEDIATE_DIR / "processed_swc_data"
PROCESSED_BIG_NEURONS_DIR = INTERMEDIATE_DIR / "processed_big_neurons"
CONNECTORS_DIR            = INTERMEDIATE_DIR / "connectors"

# ── Reciprocity working directories ─────────────────────────────────────────
RECIPROCITY_DIR           = INTERMEDIATE_DIR / "reciprocity"
RECIPROCITY_CALC_DIR      = RECIPROCITY_DIR  / "calculations"
RECIPROCITY_MODELS_DIR    = RECIPROCITY_DIR  / "models"

# ── Filopodia analysis (fig6/7) ──────────────────────────────────────────────
FILOPODIA_DIR             = DERIVED_DATA_DIR / "filopodia"

# ── Additional raw input files ───────────────────────────────────────────────
CLASSIFICATION_CSV        = RAW_DATA_DIR / "classification.csv"
NEURONS_CSV               = RAW_DATA_DIR / "neurons.csv"
SYN_BOUTON_FTR            = RAW_DATA_DIR / "syn_bouton_filopodia.ftr"

# ── Additional intermediate/derived files ───────────────────────────────────
SYNAPSE_NON_PROCESSED_FTR     = DERIVED_DATA_DIR / "synapses_783_non_processed_princeton.ftr"
PRE_CONNECTORS_FTR            = CONNECTORS_DIR / "pre_connectors_princeton.ftr"
POST_CONNECTORS_FTR           = CONNECTORS_DIR / "post_connectors_princeton.ftr"
PRE_CONNECTORS_BIGN_FTR       = CONNECTORS_DIR / "pre_connectors_princeton_bign.ftr"
POST_CONNECTORS_BIGN_FTR      = CONNECTORS_DIR / "post_connectors_princeton_bign.ftr"
SYNAPSES_PRE_MERGE_FTR        = CONNECTORS_DIR / "synapses_pre_merge_princeton.ftr"
RECIPROCITY_LIST_FTR          = DERIVED_DATA_DIR / "reciprocity_list_full_article_princeton.ftr"
CONNECTIONS_FILTERED_FTR      = DERIVED_DATA_DIR / "connections_by_syn_type_filtered_article_princeton.ftr"
CONNECTIONS_RECI_FILTERED_FTR = DERIVED_DATA_DIR / "connections_by_syn_type_reciprocal_filtered_article_princeton.ftr"
ALL_R_FULL_PKL                = RECIPROCITY_CALC_DIR / "partners" / "all_r_full_.pkl"
SELF_SDI_FTR                  = RECIPROCITY_CALC_DIR / "self_SDI.ftr"
RF_MODEL_PKL                  = RECIPROCITY_MODELS_DIR / "final_random_forest_model_princeton.pkl"
PC1_TABLE_CSV                 = DERIVED_DATA_DIR / "PC1_table.csv"
PC1_TABLE_PREDICTIONS_CSV     = DERIVED_DATA_DIR / "PC1_table_with_predictions.csv"
FULL_INFO_FTR                 = DERIVED_DATA_DIR / "full_info.ftr"

# --- fig2 additional data paths ---
NEURON_TABLE_NONP_FTR         = DERIVED_DATA_DIR / "neuron_data_full_article.ftr"
SYNAPSE_TABLE_NONP_FTR        = DERIVED_DATA_DIR / "synapses_783_article.ftr"
NEUROPIL_SYNAPSE_CSV          = RAW_DATA_DIR / "Cloud_SQL_fw_mat783_synapses_v3_neuropil.csv"
LARVA_SYNAPSES_FTR            = LARVA_DATA_DIR / "output" / "all_synapses_unprocess_larva_th0.9_SI_filt_issue_solved.ftr"
LARVA_SI_FTR                  = LARVA_DATA_DIR / "results" / "SI_list_larva_th_0.9_linker_SI_filt_issue_solved"

# --- larva processing raw inputs (Winding et al. 2023) ---
LARVA_RAW_DIR             = LARVA_DATA_DIR / "raw"
LARVA_SWC_DIR             = LARVA_RAW_DIR / "swc" / "all_neurons"   # SWC skeletons (one per neuron)
LARVA_SYNAPSE_TABLE_CSV   = LARVA_RAW_DIR / "synapse_table.csv"     # raw larval synapse table
LARVA_NEURON_LIST_CSV     = LARVA_RAW_DIR / "n_list3.csv"           # neuron filter list

# --- larva processing intermediates ---
LARVA_CONNECTORS_PKL          = LARVA_DATA_DIR / "output" / "connectors_th0.9_SI_filt_issue_solved.pkl"
LARVA_PRE_CONNECTORS_FTR      = LARVA_DATA_DIR / "output" / "pre_connectors_larva_th0.9_SI_filt_issue_solved.ftr"
LARVA_POST_CONNECTORS_FTR     = LARVA_DATA_DIR / "output" / "post_connectors_larva_th0.9_SI_filt_issue_solved.ftr"
LARVA_SYNAPSES_PRE_MERGE_FTR  = LARVA_DATA_DIR / "output" / "synapses_pre_merge_larva_th0.9_SI_filt_issue_solved.ftr"

# --- fig5 additional data paths ---
CONNECTIONS_TABLE_NONP_FTR    = DERIVED_DATA_DIR / "connections_by_syn_type_reciprocal_types_filtered_article.ftr"
NEURONS_NT_BWF_FTR            = DERIVED_DATA_DIR / "neurons_nt_bwf_frac.ftr"
