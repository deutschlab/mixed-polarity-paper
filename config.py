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
SYNAPSE_TABLE_RAW_FTR     = DERIVED_DATA_DIR / "synapses_783_article_princeton_raw.ftr"  # the version without self-synapses; see RAW_TABLE_KEEP_SELF_SYNAPSES at the end
NEURON_TABLE_FTR          = DERIVED_DATA_DIR / "neuron_data_full_article_princeton.ftr"
CONNECTIONS_TABLE_FTR     = DERIVED_DATA_DIR / (
    "connections_by_syn_type_reciprocal_types_filtered_article_princeton.ftr"
)
PCA_TABLE_FTR             = DERIVED_DATA_DIR / "neurons_pca_princeton.ftr"
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

# ── Filopodia analysis (fig7) ──────────────────────────────────────────────
FILOPODIA_DIR             = DERIVED_DATA_DIR / "filopodia"

# ── Additional raw input files ───────────────────────────────────────────────
CLASSIFICATION_CSV        = RAW_DATA_DIR / "classification.csv"
NEURONS_CSV               = RAW_DATA_DIR / "neurons.csv"
SYN_BOUTON_FTR            = RAW_DATA_DIR / "syn_bouton_filopodia.ftr"
SYN_BOUTON_NEW_FTR        = RAW_DATA_DIR / "syn_bouton_filopodia(new).ftr"   # SYN_BOUTON_FTR with bouton_on_pre set to 0 where the presynaptic cell is not in BOUTON_TABLE_FTR, and filopodia_on_post set to 0 where the postsynaptic cell is not; read by Fig 7H (the plain table gives the published 7J, this one the published 7H)
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
NEURONS_NT_BWF_FTR            = DERIVED_DATA_DIR / "neurons_nt_bwf_frac.ftr"   # written by figures/fig7/fig7i_filopodia_fraction_x_nt.py

# --- Fig 4E of the revision (figures/fig4/syntype_x_pathway_layers.py; made outside this repository, no script here writes them) ---
PER_TYPE_SUMMARY_CSV          = DERIVED_DATA_DIR / "ex_table_npil_comp_pre_post_hiSI_PerTypeSummary.csv"   # synapses per pre type, post type, comp and npil, both cells SI >= 0.1
OLFACTORY_CLASS_NEUROPIL_CSV  = DERIVED_DATA_DIR / "Olfactory_ClassNeuropilStats.csv"   # printed only

# --- alternative split methods (processing/08_alternative_split_methods.py) ---
ALT_SPLIT_BATCH_DIR           = INTERMEDIATE_DIR / "alternative_to_SI" / "all_neurons_batches"
SI_COMPARISONS_FTR            = DERIVED_DATA_DIR / "SI_comparisons.ftr"

# --- Fig 7 SI simulation (written by figures/fig7/si_sim_*.py, read by figures/fig7/si_comb_analysis.py) ---
FILOPODIA_NODES_BASELINE_FTR     = FILOPODIA_DIR / "nodes_baseline.ftr"
FILOPODIA_NODES_NO_FILOPODIA_FTR = FILOPODIA_DIR / "nodes_no_filopodia.ftr"
FILOPODIA_NODES_SAMPLEOUT_FTR    = FILOPODIA_DIR / "nodes_sampleout.ftr"

# --- Fig 7 bouton tables (made outside this repository; no script here writes them) ---
# Read by the figures/fig7/fig7*.py panel scripts. One row per bouton (neuron, cluster),
# and one row per bouton with a postsynaptic terminal (neuron, bwf); both carry NEURON_TABLE_FTR's SI
# (13 bouton neurons are not in that table). Their neuron and bwf columns read back as pandas' nullable Int64.
BOUTON_TABLE_FTR              = DERIVED_DATA_DIR / "b_table_full_prince(new)_filtered.ftr"
BOUTON_FILOPODIA_TABLE_FTR    = DERIVED_DATA_DIR / "bwf_table_full_prince(new)_filtered.ftr"
FILOPODIA_RECIPROCAL_FTR      = DERIVED_DATA_DIR / "filopodia_reciprocal.ftr"
FILOPODIA_FILTERED_FTR        = DERIVED_DATA_DIR / "filopodia_filtered.ftr"
BOUTON_INERTIA_FTR            = DERIVED_DATA_DIR / "inertia_all.ftr"
NEURON_DATA_FTR               = DERIVED_DATA_DIR / "neuron_data.ftr"   # neuron metadata with neurons.csv's nt_type, class and sub_class; not NEURON_TABLE_FTR. Its SI equals the Buhmann neuron table's to 0.001 for 111,525 of the 129,903 cells they share; read by fig7e, fig7i and fig7s2b (nt_type, sub_class) and by the revised Fig 4E script (class, primary_type, cell_type); none reads its SI
SYNAPSES_NO_SELF_FTR          = DERIVED_DATA_DIR / "t2.ftr"           # a subset of SYNAPSE_TABLE_FTR (which has no self-synapses): 1,029,925 whole (pre, post) connections, 5,727,011 synapses, are left out, for a reason not known
CONSOLIDATED_CELL_TYPES_CSV   = RAW_DATA_DIR / "consolidated_cell_types.csv.gz"   # Codex download

# --- raw synapse table (built by processing/build_raw_synapse_table.py from PRINCETON_SYNAPSE_CSV) ---
# Two versions: without self-synapses (as steps 01/02a and the authors' original split-method script use the synapses;
# the default) and with them. A cell split without them gives the SI in SI_updated.ftr and SI_comparisons.ftr;
# split with them, it gives the SI in the delivered neuron table (the same on 30 of 30 cells checked).
# This switch decides which file SYNAPSE_TABLE_RAW_FTR points to, for every script that reads it: step 08,
# figures/fig1/create_split_axon_dendrite_princeton.py and figures/split_methods/skeleton_comparisons*.py.
RAW_TABLE_KEEP_SELF_SYNAPSES     = False
SYNAPSE_TABLE_RAW_NO_SELF_FTR    = SYNAPSE_TABLE_RAW_FTR   # the name set above
SYNAPSE_TABLE_RAW_WITH_SELF_FTR  = DERIVED_DATA_DIR / "synapses_783_article_princeton_raw_with_self.ftr"
SYNAPSE_TABLE_RAW_FTR = (SYNAPSE_TABLE_RAW_WITH_SELF_FTR if RAW_TABLE_KEEP_SELF_SYNAPSES
                         else SYNAPSE_TABLE_RAW_NO_SELF_FTR)
