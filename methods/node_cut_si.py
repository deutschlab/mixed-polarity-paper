"""
Node-level SI for single-node cuts, shared by processing/08_alternative_split_methods.py
and processing/09_maxsi_synapse_labels.py.

A candidate cut is any non-root node with exactly one child. Removing it leaves two
parts: side 1, the child and its descendants, and side 2, the rest of the cell without
the cut node. The synapses on the cut node itself are counted on neither side.

The functions are step 08's own, moved here unchanged so that both steps compute the
same values. SI_calc and calc_s give the same results as methods_all.SI_calc and
methods_all.calc_s.
"""
from math import log

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

__all__ = ["SI_calc", "calc_s", "safe_SI_calc", "calculate_node_SI", "get_descendants_from_node"]


#%% SI functions (same formula as methods_all.SI_calc)

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
    Returns NaN when there are no synapses or Snorm is 0 (only pre or only post synapses);
    otherwise SI_calc.
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


#%% Node-level SI calculation

def calculate_node_SI(neuron, fisher=True):
    """
    Calculate SI for every node with:
        one parent
        one child

    Removing such a node creates exactly two components.

    Synapses attached directly to the candidate node are excluded from
    that node's SI calculation.

    fisher=False leaves fisher_odds_ratio, fisher_p and fisher_q empty (NaN)
    instead of running a Fisher exact test at every node, which takes most of
    the time. Step 09 uses only the SI and the side counts.
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

        if not fisher or side_1_total == 0 or side_2_total == 0:
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


#%% Descendants of a node

def get_descendants_from_node(
    neuron,
    start_node_id
):
    """
    Return start_node_id and all its descendants.
    """

    nodes = neuron.nodes[
        ["node_id", "parent_id"]
    ]

    # parent ID -> its children's IDs, in one pass (groupby().apply(list) was most of the time)
    children_map = {}
    for node_id, parent_id in zip(
        nodes["node_id"].tolist(),
        nodes["parent_id"].tolist()
    ):
        children_map.setdefault(parent_id, []).append(node_id)

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
