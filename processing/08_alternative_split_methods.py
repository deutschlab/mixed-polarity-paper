# -*- coding: utf-8 -*-
"""
Alternative axon/dendrite split methods compared with the published SFC split.

For every intrinsic neuron: the published SFC split, the single-node cut with the
highest SI (MaxSI), and the single-node cut with the lowest Fisher p-value
(MinFisherP). Writes per-batch results to ALT_SPLIT_BATCH_DIR and the combined
table to SI_COMPARISONS_FTR.

A candidate cut is any non-root node with exactly one child. The synapses on the
cut node itself are left out of both sides. If SI_COMPARISONS_FTR already exists,
it is kept and not overwritten.

Needs SYNAPSE_TABLE_RAW_FTR (built by processing/build_raw_synapse_table.py) and the
skeletons of every neuron under SWC_DIR. The batch range in the last section
(X = 1, Y = 236: batches X to X + Y, that is 1 to 237) covers the 118,347
intrinsic neurons of NEURON_TABLE_FTR in batches of 500; change it if that
number changes. A batch is skipped if its files already exist; the run stops if
those files have no record or come from another run (batch_run.json), and, when
combining, if a batch file is missing or the range does not end at the last batch.
"""
#%%1 Imports

import sys
from pathlib import Path
# Make repo root importable so config.py can be found from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (METHODS_DIR, NEURON_TABLE_SFC_FTR as NEURON_TABLE_FTR, SYNAPSE_TABLE_RAW_FTR,
                    ALT_SPLIT_BATCH_DIR, SI_COMPARISONS_FTR)   # the SFC cells, whatever SPLIT_METHOD says

sys.path.insert(0, str(METHODS_DIR))
from methods_all import *

import numpy as np
import pandas as pd
import navis

from math import log
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


#%%2 Settings


FLOW_THRESHOLD = 1

# Column in NEURON_TABLE_FTR that holds the neuron ID.
NEURON_ID_COL = "neuron"



#%%3 Load data

if SI_COMPARISONS_FTR.exists():
    print(f"Note: {SI_COMPARISONS_FTR.name} already exists and will not be "
          "overwritten at the end of this run.")

nodesG = pd.read_feather(NEURON_TABLE_FTR)
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_centrifugal','visual_projection'])]

allsynapses = pd.read_feather(SYNAPSE_TABLE_RAW_FTR)

allsynapses = allsynapses[
    [
        "synapse_id",
        "pre_x", "pre_y", "pre_z",
        "post_x", "post_y", "post_z",
        "size", "pre", "post", "neuropil"
    ]
].copy()


#%%4 SI functions and node-level SI (methods/node_cut_si.py)

from node_cut_si import (SI_calc, calc_s, safe_SI_calc, calculate_node_SI,
                         get_descendants_from_node)


#%%6 Split the neuron at the best SI node

def split_neuron_by_si_node(
    neuron,
    node_si_df,
    node_id=None
):
    """
    Return two navis neurons corresponding to the two components
    around a selected degree-2 node.
    """

    valid_df = node_si_df.dropna(
        subset=["SI"]
    )

    if len(valid_df) == 0:
        raise ValueError(
            "No valid node-level SI values."
        )

    if node_id is None:

        row = (
            valid_df
            .sort_values(
                "SI",
                ascending=False
            )
            .iloc[0]
        )

    else:

        matches = valid_df.loc[
            valid_df["node_id"] == node_id
        ]

        if len(matches) == 0:
            raise ValueError(
                f"Node {node_id} has no valid SI result."
            )

        row = matches.iloc[0]

    cut_node_id = row["node_id"]
    child_id = row["child_id"]

    all_node_ids = set(
        neuron.nodes["node_id"].tolist()
    )

    # Component through unique child
    side_1_ids = get_descendants_from_node(
        neuron,
        child_id
    )

    # Component through parent
    side_2_ids = (
        all_node_ids
        - side_1_ids
        - {cut_node_id}
    )

    if len(side_1_ids) == 0:
        raise ValueError(
            "Child-side component is empty."
        )

    if len(side_2_ids) == 0:
        raise ValueError(
            "Parent-side component is empty."
        )

    side_1_neuron = navis.subset_neuron(
        neuron,
        subset=list(side_1_ids),
        inplace=False
    )

    side_2_neuron = navis.subset_neuron(
        neuron,
        subset=list(side_2_ids),
        inplace=False
    )

    return {
        "row": row,
        "cut_node_id": cut_node_id,
        "child_id": child_id,

        "side_1_neuron": side_1_neuron,
        "side_2_neuron": side_2_neuron,

        "side_1_ids": side_1_ids,
        "side_2_ids": side_2_ids
    }


#%%7 Axon/dendrite assignment

def assign_axon_and_dendrite(split_result):
    """
    The side with the higher presynaptic fraction is called axon.
    The other side is called dendrite.
    """

    row = split_result["row"]

    side_1 = split_result["side_1_neuron"]
    side_2 = split_result["side_2_neuron"]

    side_1_total = (
        row["side_1_pre"]
        + row["side_1_post"]
    )

    side_2_total = (
        row["side_2_pre"]
        + row["side_2_post"]
    )

    side_1_pre_fraction = (
        row["side_1_pre"] / side_1_total
        if side_1_total > 0
        else np.nan
    )

    side_2_pre_fraction = (
        row["side_2_pre"] / side_2_total
        if side_2_total > 0
        else np.nan
    )

    if side_1_pre_fraction >= side_2_pre_fraction:

        axon_neuron = side_1
        dendrite_neuron = side_2
        axon_side = 1

    else:

        axon_neuron = side_2
        dendrite_neuron = side_1
        axon_side = 2

    return {
        "axon_neuron": axon_neuron,
        "dendrite_neuron": dendrite_neuron,
        "axon_side": axon_side,
        "side_1_pre_fraction": side_1_pre_fraction,
        "side_2_pre_fraction": side_2_pre_fraction
    }


#%%8 Detect neuron ID column

def find_neuron_id_column(
    dataframe,
    requested_column=None
):
    """
    Use requested_column when present; otherwise try common names.
    """

    if (
        requested_column is not None
        and requested_column in dataframe.columns
    ):
        return requested_column

    possible_columns = [
        "neuron",
        "root_id",
        "neuron_id",
        "root",
        "id"
    ]

    for column in possible_columns:

        if column in dataframe.columns:
            return column

    raise ValueError(
        "Could not determine the neuron-ID column. "
        f"Available columns: {dataframe.columns.tolist()}"
    )


#%%9 Run neuron batch pipeline

def run_neuron_batch_pipeline(
    nodes_dataframe,
    synapses_dataframe,
    neuron_id_col=None,
    flow_thresh=1
):
    """
    Process every neuron supplied in nodes_dataframe.

    Returns
    -------
    summary_df
        One row per successfully processed neuron.

    all_node_si_df
        Every eligible node-level SI/Fisher measurement.

    failures_df
        Neurons that failed and their error messages.

    """

    id_column = find_neuron_id_column(
        nodes_dataframe,
        requested_column=neuron_id_col
    )

    candidate_ids = (
        nodes_dataframe[id_column]
        .dropna()
        .drop_duplicates()
        .to_numpy()
    )

    if len(candidate_ids) == 0:
        raise ValueError(
            "No candidate neuron IDs found."
        )

    summary_results = []
    node_si_frames = []
    failures = []

    successful_count = 0

    for neuron_number, raw_id in enumerate(
        candidate_ids,
        start=1
    ):

        print(
            f"{neuron_number}/{len(candidate_ids)}"
        )

        if isinstance(raw_id, np.integer):
            neuron_id = int(raw_id)
        else:
            neuron_id = raw_id

        try:

            # -------------------------------------------------
            # Load and attach synapses
            # -------------------------------------------------

            swc = upload_swc(
                neuron_id
            )
            if swc is None:
                raise FileNotFoundError(
                    "No readable SWC file under SWC_DIR."
                )

            neuron_synapses = synapses_dataframe.loc[
                (synapses_dataframe["pre"] == neuron_id)
                |
                (synapses_dataframe["post"] == neuron_id)
            ].copy()

            if len(neuron_synapses) == 0:
                raise ValueError(
                    "No matching synapses were found."
                )

            neuron = heal_attach_princeton_non_process(
                swc,
                neuron_synapses
            )
            if isinstance(neuron, str):
                raise ValueError(neuron)   # 'heal or attach issue'

            total_pre = len(
                neuron.presynapses
            )

            total_post = len(
                neuron.postsynapses
            )

            # -------------------------------------------------
            # Published SFC split
            # -------------------------------------------------

            flow_split = get_split(
                neuron,
                flow_thresh=flow_thresh
            )

            if isinstance(flow_split, str):
                raise ValueError(flow_split)   # 'split issue'

            # the axon and the dendrite by their labels; steps 01 and 02a also reject a split without one of each
            axons = [f for f in flow_split if f.compartment == "axon"]
            dendrites = [f for f in flow_split if f.compartment == "dendrite"]
            if len(axons) != 1 or len(dendrites) != 1:
                raise ValueError(
                    "split_axon_dendrite did not give one axon and one dendrite: "
                    f"{[f.compartment for f in flow_split]}"
                )
            original_dendrite = dendrites[0]
            original_axon = axons[0]

            original_ax_pre = len(
                original_axon.presynapses
            )

            original_ax_post = len(
                original_axon.postsynapses
            )

            original_dend_pre = len(
                original_dendrite.presynapses
            )

            original_dend_post = len(
                original_dendrite.postsynapses
            )

            original_SI, original_IG = safe_SI_calc(
                neuron_id=neuron_id,
                side_1_pre=original_ax_pre,
                side_1_post=original_ax_post,
                side_2_pre=original_dend_pre,
                side_2_post=original_dend_post
            )

            try:

                navis_original_SI = (
                    navis.segregation_index(
                        navis.NeuronList(
                            [
                                original_axon,
                                original_dendrite
                            ]
                        )
                    )
                )

                navis_original_SI = float(
                    np.asarray(
                        navis_original_SI
                    ).reshape(-1)[0]
                )

            except Exception:
                navis_original_SI = np.nan

            # -------------------------------------------------
            # Node-level calculations
            # -------------------------------------------------

            node_SI = calculate_node_SI(
                neuron
            )

            valid_node_SI = (
                node_SI
                .dropna(subset=["SI"])
                .sort_values(
                    "SI",
                    ascending=False
                )
                .reset_index(drop=True)
            )

            if len(valid_node_SI) == 0:
                raise ValueError(
                    "No valid node-level SI values."
                )

            # Maximum-SI node
            best_node = valid_node_SI.iloc[0]

            # Minimum-p Fisher node
            valid_fisher_nodes = (
                node_SI
                .dropna(subset=["fisher_p"])
                .sort_values(
                    "fisher_p",
                    ascending=True
                )
                .reset_index(drop=True)
            )

            if len(valid_fisher_nodes) > 0:
                best_fisher_node = (
                    valid_fisher_nodes.iloc[0]
                )
            else:
                best_fisher_node = None

            # -------------------------------------------------
            # Split at maximum-SI node
            # -------------------------------------------------

            split_result = split_neuron_by_si_node(
                neuron,
                node_SI,
                node_id=best_node["node_id"]
            )

            polarity_result = assign_axon_and_dendrite(
                split_result
            )

            axon_neuron = polarity_result[
                "axon_neuron"
            ]

            dendrite_neuron = polarity_result[
                "dendrite_neuron"
            ]

            # -------------------------------------------------
            # Save neuron-level summary
            # -------------------------------------------------

            summary_results.append({

                "neuron_id": neuron_id,

                "total_pre": total_pre,
                "total_post": total_post,
                "total_synapses": total_pre + total_post,

                # Published SFC split
                "original_SI": original_SI,
                "original_IG": original_IG,
                "navis_original_SI": navis_original_SI,

                "original_axon_pre": original_ax_pre,
                "original_axon_post": original_ax_post,
                "original_dendrite_pre": original_dend_pre,
                "original_dendrite_post": original_dend_post,

                # Maximum-SI node
                "best_node_id": best_node["node_id"],
                "best_node_SI": best_node["SI"],
                "best_node_IG": best_node["IG"],

                # Fisher results at maximum-SI node
                "best_node_fisher_p":
                    best_node["fisher_p"],

                "best_node_fisher_q":
                    best_node["fisher_q"],

                "best_node_fisher_odds_ratio":
                    best_node["fisher_odds_ratio"],

                "best_node_fisher_log2_odds_ratio":
                    best_node["fisher_log2_odds_ratio"],

                "best_side_1_pre":
                    best_node["side_1_pre"],

                "best_side_1_post":
                    best_node["side_1_post"],

                "best_side_2_pre":
                    best_node["side_2_pre"],

                "best_side_2_post":
                    best_node["side_2_post"],

                "best_node_pre_excluded":
                    best_node["node_pre_excluded"],

                "best_node_post_excluded":
                    best_node["node_post_excluded"],

                # Minimum-p Fisher node
                "best_fisher_node_id": (
                    best_fisher_node["node_id"]
                    if best_fisher_node is not None
                    else np.nan
                ),

                "best_fisher_node_SI": (
                    best_fisher_node["SI"]
                    if best_fisher_node is not None
                    else np.nan
                ),

                "best_fisher_node_IG": (
                    best_fisher_node["IG"]
                    if best_fisher_node is not None
                    else np.nan
                ),

                "best_fisher_p": (
                    best_fisher_node["fisher_p"]
                    if best_fisher_node is not None
                    else np.nan
                ),

                "best_fisher_q": (
                    best_fisher_node["fisher_q"]
                    if best_fisher_node is not None
                    else np.nan
                ),

                "best_fisher_odds_ratio": (
                    best_fisher_node["fisher_odds_ratio"]
                    if best_fisher_node is not None
                    else np.nan
                ),

                "best_fisher_log2_odds_ratio": (
                    best_fisher_node[
                        "fisher_log2_odds_ratio"
                    ]
                    if best_fisher_node is not None
                    else np.nan
                ),

                "best_fisher_side_1_pre": (
                    best_fisher_node["side_1_pre"]
                    if best_fisher_node is not None
                    else np.nan
                ),

                "best_fisher_side_1_post": (
                    best_fisher_node["side_1_post"]
                    if best_fisher_node is not None
                    else np.nan
                ),

                "best_fisher_side_2_pre": (
                    best_fisher_node["side_2_pre"]
                    if best_fisher_node is not None
                    else np.nan
                ),

                "best_fisher_side_2_post": (
                    best_fisher_node["side_2_post"]
                    if best_fisher_node is not None
                    else np.nan
                ),

                # Polarity assignment at maximum-SI node
                "axon_side":
                    polarity_result["axon_side"],

                "side_1_pre_fraction":
                    polarity_result["side_1_pre_fraction"],

                "side_2_pre_fraction":
                    polarity_result["side_2_pre_fraction"]
            })

            # -------------------------------------------------
            # Save all node-level results
            # -------------------------------------------------

            node_SI_to_save = node_SI.copy()

            node_SI_to_save.insert(
                0,
                "neuron_id",
                neuron_id
            )

            node_si_frames.append(
                node_SI_to_save
            )



            successful_count += 1

        except Exception as error:

            print(
                f"Failed neuron {neuron_id}: {error}"
            )

            failures.append(
                {
                    "neuron_id": neuron_id,
                    "error": str(error)
                }
            )

    # ---------------------------------------------------------
    # Combine output tables
    # ---------------------------------------------------------

    summary_df = pd.DataFrame(
        summary_results
    )

    if len(node_si_frames) > 0:

        all_node_si_df = pd.concat(
            node_si_frames,
            ignore_index=True
        )

    else:

        all_node_si_df = pd.DataFrame()

    failures_df = pd.DataFrame(
        failures
    )

    print(
        f"\nSuccessful: "
        f"{successful_count}/{len(candidate_ids)}"
    )

    return (
        summary_df,
        all_node_si_df,
        failures_df,
        
    )

#%%10 Execute full-neuron analysis in batches

import gc


# ---------------------------------------------------------
# Create fixed neuron list
# ---------------------------------------------------------

all_neurons = (
    nodesG
    .dropna(subset=[NEURON_ID_COL])
    .drop_duplicates(subset=[NEURON_ID_COL])
    .reset_index(drop=True)
)

# ---------------------------------------------------------
# Batch settings
# ---------------------------------------------------------

BATCH_SIZE = 500

n_batches = int(
    np.ceil(len(all_neurons) / BATCH_SIZE)
)


# ---------------------------------------------------------
# Output folder
# ---------------------------------------------------------

batch_data_path = ALT_SPLIT_BATCH_DIR

batch_data_path.mkdir(
    parents=True,
    exist_ok=True
)
check_batch_record(batch_data_path, all_neurons[NEURON_ID_COL], BATCH_SIZE)   # stops on another run's batches

print("Total neurons:", len(all_neurons))
print("Batch size:", BATCH_SIZE)
print("Number of batches:", n_batches)


# ---------------------------------------------------------
# Run batches
# ---------------------------------------------------------

for batch_number in range(n_batches):

    batch_id = batch_number + 1

    start = batch_number * BATCH_SIZE
    end = min(
        start + BATCH_SIZE,
        len(all_neurons)
    )

    batch_neurons = (
        all_neurons
        .iloc[start:end]
        .copy()
    )


    # -----------------------------------------------------
    # Output files
    # -----------------------------------------------------

    summary_file = (
        batch_data_path
        / f"summary_batch_{batch_id:04d}.ftr"
    )

    nodes_file = (
        batch_data_path
        / f"nodes_batch_{batch_id:04d}.ftr"
    )

    failures_file = (
        batch_data_path
        / f"failures_batch_{batch_id:04d}.ftr"
    )


    # -----------------------------------------------------
    # Skip already completed batches
    # -----------------------------------------------------

    if (
        summary_file.exists()
        and nodes_file.exists()
        and failures_file.exists()
    ):

        print(
            f"\nBatch {batch_id}/{n_batches} already exists. "
            "Skipping."
        )

        continue


    print(
        f"\nBatch {batch_id}/{n_batches}"
    )

    print(
        f"Neurons {start + 1}-{end} "
        f"({len(batch_neurons)} neurons)"
    )


    # -----------------------------------------------------
    # Process batch
    # -----------------------------------------------------

    (
        summary_df,
        all_node_si_df,
        failures_df,
        
    ) = run_neuron_batch_pipeline(

        nodes_dataframe=batch_neurons,
        synapses_dataframe=allsynapses,

        neuron_id_col=NEURON_ID_COL,
        flow_thresh=FLOW_THRESHOLD
    )


    # -----------------------------------------------------
    # Ensure failures table can always be saved
    # -----------------------------------------------------

    if len(failures_df.columns) == 0:

        failures_df = pd.DataFrame(
            columns=[
                "neuron_id",
                "error"
            ]
        )


    # -----------------------------------------------------
    # Save batch
    # -----------------------------------------------------

    summary_df.reset_index(
        drop=True
    ).to_feather(
        summary_file
    )

    all_node_si_df.reset_index(
        drop=True
    ).to_feather(
        nodes_file
    )

    failures_df.reset_index(
        drop=True
    ).to_feather(
        failures_file
    )


    print(
        f"Saved batch {batch_id}/{n_batches}"
    )

    print(
        f"Successful: {len(summary_df)} | "
        f"Failed: {len(failures_df)}"
    )


    # -----------------------------------------------------
    # Clear batch from memory
    # -----------------------------------------------------

    del batch_neurons
    del summary_df
    del all_node_si_df
    del failures_df

    gc.collect()


print("\nFinished all batches.")
#%%

START_BATCH = 1           # batch numbers start at 1
END_BATCH = n_batches     # every batch the run above made
batch_data_path = ALT_SPLIT_BATCH_DIR


#%% 3. Read summary files from selected batches

summary_frames = []
missing_batches = []
for batch_id in range(START_BATCH, END_BATCH + 1):

    print(f"Reading batch {batch_id}")

    summary_file = (
        batch_data_path
        / f"summary_batch_{batch_id:04d}.ftr"
    )

    if summary_file.exists():

        batch_df = pd.read_feather(summary_file)

        summary_frames.append(batch_df)

    else:

        missing_batches.append(
            summary_file.name
        )

if missing_batches: raise SystemExit(f"batch files missing, so their neurons would be left out: {missing_batches}")

# the neurons that failed (no SWC, heal or attach, split, no valid node SI) are in the failures files;
# every neuron of the run must be in one of the two
failure_frames = [pd.read_feather(p) for p in
                  (batch_data_path / f"failures_batch_{b:04d}.ftr" for b in range(START_BATCH, END_BATCH + 1))
                  if p.exists()]
failures_all = pd.concat(failure_frames, ignore_index=True) if failure_frames else pd.DataFrame(columns=["neuron_id", "error"])
n_done = sum(len(f) for f in summary_frames)
if n_done + len(failures_all) != len(all_neurons):
    raise SystemExit(f"{n_done} neurons in the summaries and {len(failures_all)} in the failures, "
                     f"but the run had {len(all_neurons)}")
print(f"Neurons that failed: {len(failures_all)}")
if len(failures_all):
    print(failures_all["error"].str.split(":").str[0].value_counts().to_string())
#%% 4. Combine selected batches

if len(summary_frames) > 0:

    summary_df = pd.concat(
        summary_frames,
        ignore_index=True
    )

else:

    summary_df = pd.DataFrame()


print(
    "Loaded neurons:",
    len(summary_df)
)
del summary_frames

#%% 5. Absolute Phi function

def absolute_phi(a, b, c, d):
    """
    Absolute Phi coefficient for a 2x2 contingency table.

                 pre    post
    side 1        a       b
    side 2        c       d

    Returns a value between 0 and 1.
    """

    denominator = np.sqrt(
        (a + b)
        * (c + d)
        * (a + c)
        * (b + d)
    )

    if denominator == 0:
        return np.nan

    phi = (
        (a * d - b * c)
        / denominator
    )

    return abs(phi)


#%% 6. Calculate Phi for SFC split

summary_df["SFC_Phi"] = summary_df.apply(
    lambda row: absolute_phi(
        row["original_axon_pre"],
        row["original_axon_post"],
        row["original_dendrite_pre"],
        row["original_dendrite_post"]
    ),
    axis=1
)


#%% 7. Calculate Phi for Fisher-selected split

summary_df["Fisher_Phi"] = summary_df.apply(
    lambda row: absolute_phi(
        row["best_fisher_side_1_pre"],
        row["best_fisher_side_1_post"],
        row["best_fisher_side_2_pre"],
        row["best_fisher_side_2_post"]
    ),
    axis=1
)

#%% 8. Create comparison dataframe

comparison_df = summary_df[
    [
        "neuron_id",

        # SFC split + SI
        "original_SI",

        # SFC split + Phi
        "SFC_Phi",

        # Max SI split + SI
        "best_node_SI",

        # Fisher p-value split + Phi
        "Fisher_Phi"
    ]
].copy()


#%% 9. Rename measures clearly

comparison_df = comparison_df.rename(
    columns={
        "original_SI": "SFC_SI",
        "best_node_SI": "MaxSI_SI",
        "Fisher_Phi": "MinFisherP_Phi"
    }
)
#%%
# Keep an existing SI_comparisons.ftr; write one only if none exists
if SI_COMPARISONS_FTR.exists():
    print(f"{SI_COMPARISONS_FTR.name} already exists; not overwriting. "
          "Move or rename it first to write a new one.")
else:
    comparison_df.to_feather(SI_COMPARISONS_FTR)