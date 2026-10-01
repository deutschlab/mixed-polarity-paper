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
number changes. A batch is skipped if its files
already exist, and a missing batch file is reported and skipped when the results
are combined, so check the row count of the combined table.
"""
#%%1 Imports

import sys
from pathlib import Path
# Make repo root importable so config.py can be found from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (METHODS_DIR, NEURON_TABLE_FTR, SYNAPSE_TABLE_RAW_FTR,
                    ALT_SPLIT_BATCH_DIR, SI_COMPARISONS_FTR)

sys.path.insert(0, str(METHODS_DIR))
from methods_all import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import navis

from math import log
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


#%%2 Settings

#X = 5000
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


#%%4 Original SI functions

def SI_calc(neuron_id_data):

    nid = neuron_id_data[0]

    ax_pre = neuron_id_data[1][0]
    ax_post = neuron_id_data[1][1]

    dend_pre = neuron_id_data[2][0]
    dend_post = neuron_id_data[2][1]

    ent_ax = calc_s(ax_pre, ax_post)
    ent_dend = calc_s(dend_pre, dend_post)

    snorm = calc_s(
        ax_pre + dend_pre,
        ax_post + dend_post
    )

    total = (
        ax_pre
        + ax_post
        + dend_pre
        + dend_post
    )

    S = (
        (ax_pre + ax_post) * ent_ax
        + (dend_pre + dend_post) * ent_dend
    ) / total

    SI = 1 - (S / snorm)
    IG = snorm - S

    return SI, IG


def calc_s(pre, post):

    total = pre + post

    if total > 0:

        p_pre = pre / total
        p_post = post / total

        entropy_pre = (
            p_pre * log(p_pre, 2)
            if p_pre > 0
            else 0
        )

        entropy_post = (
            p_post * log(p_post, 2)
            if p_post > 0
            else 0
        )

        entropy = -(
            entropy_pre
            + entropy_post
        )

    else:
        entropy = 0

    return entropy


def safe_SI_calc(
    neuron_id,
    side_1_pre,
    side_1_post,
    side_2_pre,
    side_2_post
):
    """
    Calls the original SI_calc only when its denominators are defined.
    """

    total_pre = side_1_pre + side_2_pre
    total_post = side_1_post + side_2_post
    total = total_pre + total_post

    if total == 0:
        return np.nan, np.nan

    snorm = calc_s(total_pre, total_post)

    if snorm == 0:
        return np.nan, np.nan

    return SI_calc([
        neuron_id,
        (side_1_pre, side_1_post),
        (side_2_pre, side_2_post)
    ])


#%%5 Node-level SI calculation

def calculate_node_SI(neuron):
    """
    Calculate SI for every node with:
        one parent
        one child

    Removing such a node creates exactly two components.

    Synapses attached directly to the candidate node are excluded from
    that node's SI calculation.
    """

    nodes = (
        neuron.nodes[
            ["node_id", "parent_id"]
        ]
        .drop_duplicates("node_id")
        .copy()
    )

    node_ids = nodes["node_id"].to_numpy()
    parent_ids = nodes["parent_id"].to_numpy()

    id_to_index = {
        node_id: index
        for index, node_id in enumerate(node_ids)
    }

    n_nodes = len(node_ids)

    # Direct synapse counts at every node
    if "node_id" not in neuron.presynapses.columns:
        raise ValueError(
            "presynapses has no node_id column."
        )

    if "node_id" not in neuron.postsynapses.columns:
        raise ValueError(
            "postsynapses has no node_id column."
        )

    direct_pre = (
        neuron.presynapses["node_id"]
        .value_counts()
        .reindex(node_ids, fill_value=0)
        .to_numpy(dtype=np.int64)
    )

    direct_post = (
        neuron.postsynapses["node_id"]
        .value_counts()
        .reindex(node_ids, fill_value=0)
        .to_numpy(dtype=np.int64)
    )

    # Build the tree representation
    parent_index = np.full(
        n_nodes,
        -1,
        dtype=np.int64
    )

    children = [
        []
        for _ in range(n_nodes)
    ]

    for child_index, parent_id in enumerate(parent_ids):

        if parent_id in id_to_index:

            p_index = id_to_index[parent_id]

            parent_index[child_index] = p_index
            children[p_index].append(child_index)

    roots = np.where(
        parent_index == -1
    )[0].tolist()

    if len(roots) != 1:
        raise ValueError(
            f"Expected one healed tree, found {len(roots)} roots."
        )

    # Root-to-leaf traversal
    traversal = []
    stack = roots.copy()

    while stack:

        current = stack.pop()
        traversal.append(current)

        stack.extend(
            children[current]
        )

    if len(traversal) != n_nodes:
        raise ValueError(
            "Could not traverse every skeleton node."
        )

    # Accumulate subtree synapse counts
    subtree_pre = direct_pre.copy()
    subtree_post = direct_post.copy()

    for current in reversed(traversal):

        p_index = parent_index[current]

        if p_index != -1:

            subtree_pre[p_index] += (
                subtree_pre[current]
            )

            subtree_post[p_index] += (
                subtree_post[current]
            )

    total_pre = int(direct_pre.sum())
    total_post = int(direct_post.sum())

    neuron_id = getattr(
        neuron,
        "id",
        None
    )

    results = []

    for node_index, node_id in enumerate(node_ids):

        # Exclude root
        if parent_index[node_index] == -1:
            continue

        # Keep only one-parent/one-child nodes
        if len(children[node_index]) != 1:
            continue

        child_index = children[node_index][0]

        # Component through the unique child
        side_1_pre = int(
            subtree_pre[child_index]
        )

        side_1_post = int(
            subtree_post[child_index]
        )

        # Component through the parent
        side_2_pre = int(
            total_pre
            - subtree_pre[node_index]
        )

        side_2_post = int(
            total_post
            - subtree_post[node_index]
        )

        # Synapses attached directly to candidate node
        node_pre = int(
            direct_pre[node_index]
        )

        node_post = int(
            direct_post[node_index]
        )

        SI, IG = safe_SI_calc(
            neuron_id=neuron_id,
            side_1_pre=side_1_pre,
            side_1_post=side_1_post,
            side_2_pre=side_2_pre,
            side_2_post=side_2_post
        )

        side_1_total = side_1_pre + side_1_post
        side_2_total = side_2_pre + side_2_post

        if side_1_total == 0 or side_2_total == 0:
            fisher_odds_ratio = np.nan
            fisher_p = np.nan
        else:
            fisher_odds_ratio, fisher_p = fisher_exact(
                [
                    [side_1_pre, side_1_post],
                    [side_2_pre, side_2_post]
                ],
                alternative="two-sided"
            )

        fisher_log2_odds_ratio = np.log2(
            (
                (side_1_pre + 0.5)
                * (side_2_post + 0.5)
            )
            /
            (
                (side_1_post + 0.5)
                * (side_2_pre + 0.5)
            )
        )

        results.append({
            "node_id": node_id,
            "parent_id": parent_ids[node_index],
            "child_id": node_ids[child_index],

            "side_1_pre": side_1_pre,
            "side_1_post": side_1_post,

            "side_2_pre": side_2_pre,
            "side_2_post": side_2_post,

            "node_pre_excluded": node_pre,
            "node_post_excluded": node_post,

            "n_synapses_used": (
                side_1_pre
                + side_1_post
                + side_2_pre
                + side_2_post
            ),

            "SI": SI,
            "IG": IG,

            "fisher_odds_ratio": fisher_odds_ratio,
            "fisher_log2_odds_ratio": fisher_log2_odds_ratio,
            "fisher_p": fisher_p
        })

    result_df = pd.DataFrame(results)

    if len(result_df) == 0:
        return result_df

    result_df["fisher_q"] = np.nan

    valid_fisher = result_df["fisher_p"].notna()

    if valid_fisher.any():
        result_df.loc[
            valid_fisher,
            "fisher_q"
        ] = multipletests(
            result_df.loc[
                valid_fisher,
                "fisher_p"
            ],
            method="fdr_bh"
        )[1]

    return result_df


#%%6 Split the neuron at the best SI node

def get_descendants_from_node(
    neuron,
    start_node_id
):
    """
    Return start_node_id and all its descendants.
    """

    nodes = neuron.nodes[
        ["node_id", "parent_id"]
    ].copy()

    children_map = (
        nodes
        .groupby("parent_id")["node_id"]
        .apply(list)
        .to_dict()
    )

    descendants = set()
    stack = [start_node_id]

    while stack:

        current = stack.pop()

        if current in descendants:
            continue

        descendants.add(current)

        stack.extend(
            children_map.get(
                current,
                []
            )
        )

    return descendants


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


#%%9 Run the random-neuron pipeline
#%%9 Run neuron batch pipeline

def run_neuron_batch_pipeline(
    nodes_dataframe,
    synapses_dataframe,
    neuron_id_col=None,
    flow_thresh=1,
    plot=True
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

            total_pre = len(
                neuron.presynapses
            )

            total_post = len(
                neuron.postsynapses
            )

            # -------------------------------------------------
            # Original flow-based split
            # -------------------------------------------------

            flow_split = get_split(
                neuron,
                flow_thresh=flow_thresh
            )

            if len(flow_split) < 2:
                raise ValueError(
                    "get_split returned fewer than two fragments."
                )
            #this needs to be fixed
            original_dendrite = flow_split[0]
            original_axon = flow_split[-1]

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

            split_colored = navis.NeuronList(
                [
                    dendrite_neuron,
                    axon_neuron
                ]
            )

            # -------------------------------------------------
            # Save neuron-level summary
            # -------------------------------------------------

            summary_results.append({

                "neuron_id": neuron_id,

                "total_pre": total_pre,
                "total_post": total_post,
                "total_synapses": total_pre + total_post,

                # Original flow-based split
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

#%%10 Execute

#%%10 Execute

#%%10 Execute full-neuron analysis in batches

from pathlib import Path
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

###test

#all_neurons = all_neurons.iloc[:10].copy()
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
        flow_thresh=FLOW_THRESHOLD,

        plot=False
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

X = 1      # first batch (batch numbers start at 1)
Y = 236    # last batch will be X + Y

START_BATCH = X
END_BATCH = X + Y

batch_data_path = ALT_SPLIT_BATCH_DIR


#%% 3. Read summary files from selected batches

summary_frames = []

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

        print(
            f"Missing: {summary_file.name}"
        )


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

#%%
aa=summary_df.head()
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