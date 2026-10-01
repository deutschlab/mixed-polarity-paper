# Split methods: the same comparison for Kenyon cells (KCg-m) where the MaxSI cut scores higher than the
# published one: a density of the SI difference per KC type, with a histogram over all KC types drawn on the
# same axes (as in Amit's script), then 5 cells from each of five bands.
# Ported from Amit's ALT_SI_COMP/Skeleton_comparisons_KC.py (see ALT_SI_COMP/note_for_kc.txt).
# Changes: paths via config.py and the repository preamble, which also sets editable Arial text in the SVGs;
# each split image is also saved; and the ONLY_CELLS_WITH_SKELETONS option ("auto" keeps Amit's exact
# selection when all 25 skeletons are on disk, and otherwise replaces it). MaxSI scores higher than SFC in
# 5,173 of the 5,174 KCs in SI_COMPARISONS_FTR, so the SFC_SI < MaxSI_SI filter removes almost nothing.
# Reads SYNAPSE_TABLE_RAW_FTR (see processing/build_raw_synapse_table.py) and the skeletons under SWC_DIR.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import METHODS_DIR, SWC_DIR, NEURON_TABLE_FTR, OUTPUT_DIR, SI_COMPARISONS_FTR, SYNAPSE_TABLE_RAW_FTR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *

import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

_out_dir = OUTPUT_DIR / "split_methods/skeleton_comparisons_kc"
_out_dir.mkdir(parents=True, exist_ok=True)
#%%
#%% Imports

from pathlib import Path
import pandas as pd
import numpy as np

import sys

from methods_all import *

import matplotlib.pyplot as plt
import navis

from math import log
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


#%% Settings

FLOW_THRESHOLD = 1

DENDRITE_COLOR = "orange"
LINKER_COLOR = "green"
AXON_COLOR = "purple"


#%% Load SI comparison dataframe

df = pd.read_feather(
    SI_COMPARISONS_FTR
)
#%% Select cases where SFC SI > Max SI



#%%
nodesG = pd.read_feather(
    NEURON_TABLE_FTR
)


nodesG = nodesG[
    nodesG["super_class"].isin(
        [
            "central",
            "optic",
            "visual_centrifugal",
            "visual_projection"
        ]
    )
].copy()


#%%

nodesG=nodesG.dropna(subset=['primary_type'])
KC=nodesG[nodesG['primary_type'].str.contains('KC')]
#%%
KC_outliers=KC.merge(df,how='left',left_on='neuron',right_on='neuron_id')
#%%
KC_outliers=KC_outliers[['neuron','primary_type','SFC_SI','MaxSI_SI','MinFisherP_Phi']]
#%%

KC_outliers['SI_dif'] = (
    KC_outliers['SFC_SI']
    - KC_outliers['MaxSI_SI']
)

SI_ = KC_outliers[
    KC_outliers['SFC_SI'] < KC_outliers['MaxSI_SI']
].copy()




#%% Sort by largest difference

SI_ = SI_.sort_values(
    by='SI_dif',
    ascending=False
).reset_index(drop=True)
#%%

#%%
import seaborn as sns
sns.kdeplot(SI_,x='SI_dif',hue='primary_type')
plt.savefig(_out_dir / "kde_KC.svg")

#%%
KCg_m=SI_[SI_['primary_type']=='KCg-m']
#%%
sns.histplot(SI_,x='SI_dif',bins=10)
plt.savefig(_out_dir / "KCgm_hist.svg")

#%% Count plot of SI difference bins


#%% Load synapses

allsynapses = pd.read_feather(
    SYNAPSE_TABLE_RAW_FTR
)

allsynapses = allsynapses[
    [
        "synapse_id",
        "pre_x", "pre_y", "pre_z",
        "post_x", "post_y", "post_z",
        "size", "pre", "post", "neuropil"
    ]
].copy()


#%% Original SI functions

def SI_calc(neuron_id_data):

    nid = neuron_id_data[0]

    ax_pre = neuron_id_data[1][0]
    ax_post = neuron_id_data[1][1]

    dend_pre = neuron_id_data[2][0]
    dend_post = neuron_id_data[2][1]

    ent_ax = calc_s(
        ax_pre,
        ax_post
    )

    ent_dend = calc_s(
        dend_pre,
        dend_post
    )

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

    total_pre = (
        side_1_pre
        + side_2_pre
    )

    total_post = (
        side_1_post
        + side_2_post
    )

    total = (
        total_pre
        + total_post
    )

    if total == 0:
        return np.nan, np.nan

    snorm = calc_s(
        total_pre,
        total_post
    )

    if snorm == 0:
        return np.nan, np.nan

    return SI_calc(
        [
            neuron_id,
            (
                side_1_pre,
                side_1_post
            ),
            (
                side_2_pre,
                side_2_post
            )
        ]
    )


#%% Node-level SI calculation

def calculate_node_SI(neuron):

    nodes = (
        neuron.nodes[
            [
                "node_id",
                "parent_id"
            ]
        ]
        .drop_duplicates(
            "node_id"
        )
        .copy()
    )

    node_ids = (
        nodes["node_id"]
        .to_numpy()
    )

    parent_ids = (
        nodes["parent_id"]
        .to_numpy()
    )

    id_to_index = {
        node_id: index
        for index, node_id
        in enumerate(node_ids)
    }

    n_nodes = len(
        node_ids
    )


    # -----------------------------------------
    # Direct synapse counts
    # -----------------------------------------

    if (
        "node_id"
        not in neuron.presynapses.columns
    ):
        raise ValueError(
            "presynapses has no node_id column."
        )

    if (
        "node_id"
        not in neuron.postsynapses.columns
    ):
        raise ValueError(
            "postsynapses has no node_id column."
        )

    direct_pre = (
        neuron.presynapses[
            "node_id"
        ]
        .value_counts()
        .reindex(
            node_ids,
            fill_value=0
        )
        .to_numpy(
            dtype=np.int64
        )
    )

    direct_post = (
        neuron.postsynapses[
            "node_id"
        ]
        .value_counts()
        .reindex(
            node_ids,
            fill_value=0
        )
        .to_numpy(
            dtype=np.int64
        )
    )


    # -----------------------------------------
    # Build tree
    # -----------------------------------------

    parent_index = np.full(
        n_nodes,
        -1,
        dtype=np.int64
    )

    children = [
        []
        for _ in range(n_nodes)
    ]

    for (
        child_index,
        parent_id
    ) in enumerate(parent_ids):

        if parent_id in id_to_index:

            p_index = (
                id_to_index[
                    parent_id
                ]
            )

            parent_index[
                child_index
            ] = p_index

            children[
                p_index
            ].append(
                child_index
            )


    roots = np.where(
        parent_index == -1
    )[0].tolist()

    if len(roots) != 1:

        raise ValueError(
            f"Expected one healed tree, "
            f"found {len(roots)} roots."
        )


    # -----------------------------------------
    # Traverse
    # -----------------------------------------

    traversal = []

    stack = (
        roots.copy()
    )

    while stack:

        current = (
            stack.pop()
        )

        traversal.append(
            current
        )

        stack.extend(
            children[current]
        )


    if len(traversal) != n_nodes:

        raise ValueError(
            "Could not traverse every skeleton node."
        )


    # -----------------------------------------
    # Subtree synapse counts
    # -----------------------------------------

    subtree_pre = (
        direct_pre.copy()
    )

    subtree_post = (
        direct_post.copy()
    )

    for current in reversed(
        traversal
    ):

        p_index = (
            parent_index[
                current
            ]
        )

        if p_index != -1:

            subtree_pre[
                p_index
            ] += subtree_pre[
                current
            ]

            subtree_post[
                p_index
            ] += subtree_post[
                current
            ]


    total_pre = int(
        direct_pre.sum()
    )

    total_post = int(
        direct_post.sum()
    )

    neuron_id = getattr(
        neuron,
        "id",
        None
    )


    results = []


    # -----------------------------------------
    # Test each eligible node
    # -----------------------------------------

    for (
        node_index,
        node_id
    ) in enumerate(node_ids):

        # Exclude root
        if (
            parent_index[
                node_index
            ]
            == -1
        ):
            continue

        # Only 1-child nodes
        if (
            len(
                children[
                    node_index
                ]
            )
            != 1
        ):
            continue


        child_index = (
            children[
                node_index
            ][0]
        )


        # Child side

        side_1_pre = int(
            subtree_pre[
                child_index
            ]
        )

        side_1_post = int(
            subtree_post[
                child_index
            ]
        )


        # Parent side

        side_2_pre = int(
            total_pre
            - subtree_pre[
                node_index
            ]
        )

        side_2_post = int(
            total_post
            - subtree_post[
                node_index
            ]
        )


        # Synapses directly on cut node

        node_pre = int(
            direct_pre[
                node_index
            ]
        )

        node_post = int(
            direct_post[
                node_index
            ]
        )


        # -------------------------------------
        # SI
        # -------------------------------------

        SI, IG = safe_SI_calc(
            neuron_id=neuron_id,

            side_1_pre=
                side_1_pre,

            side_1_post=
                side_1_post,

            side_2_pre=
                side_2_pre,

            side_2_post=
                side_2_post
        )


        # -------------------------------------
        # Fisher
        # -------------------------------------

        side_1_total = (
            side_1_pre
            + side_1_post
        )

        side_2_total = (
            side_2_pre
            + side_2_post
        )


        if (
            side_1_total == 0
            or
            side_2_total == 0
        ):

            fisher_odds_ratio = (
                np.nan
            )

            fisher_p = (
                np.nan
            )

        else:

            (
                fisher_odds_ratio,
                fisher_p

            ) = fisher_exact(

                [
                    [
                        side_1_pre,
                        side_1_post
                    ],
                    [
                        side_2_pre,
                        side_2_post
                    ]
                ],

                alternative=
                    "two-sided"
            )


        fisher_log2_odds_ratio = (
            np.log2(
                (
                    (side_1_pre + 0.5)
                    *
                    (side_2_post + 0.5)
                )
                /
                (
                    (side_1_post + 0.5)
                    *
                    (side_2_pre + 0.5)
                )
            )
        )


        results.append(
            {

                "node_id":
                    node_id,

                "parent_id":
                    parent_ids[
                        node_index
                    ],

                "child_id":
                    node_ids[
                        child_index
                    ],


                "side_1_pre":
                    side_1_pre,

                "side_1_post":
                    side_1_post,

                "side_2_pre":
                    side_2_pre,

                "side_2_post":
                    side_2_post,


                "node_pre_excluded":
                    node_pre,

                "node_post_excluded":
                    node_post,


                "n_synapses_used":
                    (
                        side_1_pre
                        + side_1_post
                        + side_2_pre
                        + side_2_post
                    ),


                "SI":
                    SI,

                "IG":
                    IG,


                "fisher_odds_ratio":
                    fisher_odds_ratio,

                "fisher_log2_odds_ratio":
                    fisher_log2_odds_ratio,

                "fisher_p":
                    fisher_p
            }
        )


    result_df = pd.DataFrame(
        results
    )


    if len(result_df) == 0:

        return result_df


    # -----------------------------------------
    # FDR correction
    # -----------------------------------------

    result_df[
        "fisher_q"
    ] = np.nan


    valid_fisher = (
        result_df[
            "fisher_p"
        ]
        .notna()
    )


    if valid_fisher.any():

        result_df.loc[
            valid_fisher,
            "fisher_q"
        ] = multipletests(

            result_df.loc[
                valid_fisher,
                "fisher_p"
            ],

            method=
                "fdr_bh"

        )[1]


    return result_df


#%% Get descendants

def get_descendants_from_node(
    neuron,
    start_node_id
):

    nodes = (
        neuron.nodes[
            [
                "node_id",
                "parent_id"
            ]
        ]
        .copy()
    )


    children_map = (
        nodes
        .groupby(
            "parent_id"
        )[
            "node_id"
        ]
        .apply(list)
        .to_dict()
    )


    descendants = set()

    stack = [
        start_node_id
    ]


    while stack:

        current = (
            stack.pop()
        )

        if (
            current
            in descendants
        ):
            continue


        descendants.add(
            current
        )


        stack.extend(
            children_map.get(
                current,
                []
            )
        )


    return descendants


#%% Split neuron at selected SI node

def split_neuron_by_si_node(
    neuron,
    node_si_df,
    node_id=None
):

    valid_df = (
        node_si_df
        .dropna(
            subset=["SI"]
        )
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

        matches = (
            valid_df.loc[
                valid_df[
                    "node_id"
                ]
                == node_id
            ]
        )


        if len(matches) == 0:

            raise ValueError(
                f"Node {node_id} "
                f"has no valid SI result."
            )


        row = (
            matches.iloc[0]
        )


    cut_node_id = (
        row["node_id"]
    )

    child_id = (
        row["child_id"]
    )


    all_node_ids = set(
        neuron.nodes[
            "node_id"
        ]
        .tolist()
    )


    side_1_ids = (
        get_descendants_from_node(
            neuron,
            child_id
        )
    )


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


    side_1_neuron = (
        navis.subset_neuron(
            neuron,
            subset=list(
                side_1_ids
            ),
            inplace=False
        )
    )


    side_2_neuron = (
        navis.subset_neuron(
            neuron,
            subset=list(
                side_2_ids
            ),
            inplace=False
        )
    )


    return {

        "row":
            row,

        "cut_node_id":
            cut_node_id,

        "child_id":
            child_id,

        "side_1_neuron":
            side_1_neuron,

        "side_2_neuron":
            side_2_neuron,

        "side_1_ids":
            side_1_ids,

        "side_2_ids":
            side_2_ids
    }


#%% Assign axon / dendrite

def assign_axon_and_dendrite(
    split_result
):

    row = (
        split_result[
            "row"
        ]
    )

    side_1 = (
        split_result[
            "side_1_neuron"
        ]
    )

    side_2 = (
        split_result[
            "side_2_neuron"
        ]
    )


    side_1_total = (
        row["side_1_pre"]
        + row["side_1_post"]
    )

    side_2_total = (
        row["side_2_pre"]
        + row["side_2_post"]
    )


    side_1_pre_fraction = (
        row["side_1_pre"]
        / side_1_total

        if side_1_total > 0

        else np.nan
    )


    side_2_pre_fraction = (
        row["side_2_pre"]
        / side_2_total

        if side_2_total > 0

        else np.nan
    )


    if (
        side_1_pre_fraction
        >=
        side_2_pre_fraction
    ):

        axon_neuron = (
            side_1
        )

        dendrite_neuron = (
            side_2
        )

        axon_side = 1


    else:

        axon_neuron = (
            side_2
        )

        dendrite_neuron = (
            side_1
        )

        axon_side = 2


    return {

        "axon_neuron":
            axon_neuron,

        "dendrite_neuron":
            dendrite_neuron,

        "axon_side":
            axon_side,

        "side_1_pre_fraction":
            side_1_pre_fraction,

        "side_2_pre_fraction":
            side_2_pre_fraction
    }


#%% Find neuron ID column

def find_neuron_id_column(
    dataframe,
    requested_column=None
):

    if (
        requested_column is not None
        and
        requested_column
        in dataframe.columns
    ):

        return (
            requested_column
        )


    possible_columns = [

        "neuron",
        "root_id",
        "neuron_id",
        "root",
        "id"
    ]


    for column in possible_columns:

        if (
            column
            in dataframe.columns
        ):

            return column


    raise ValueError(
        "Could not determine the neuron-ID column. "
        f"Available columns: "
        f"{dataframe.columns.tolist()}"
    )


#%% Figure helper
# SAME STYLE AS OLD save_split_image()
# BUT SHOW ONLY - DO NOT SAVE

def show_split_image(
    neuron_list,
    colors,
    title
):

    fig, ax = navis.plot2d(
        neuron_list,
        connectors=False,
        method="3d",
        colors=colors,
        cn_size=10,
        linewidth=0.5
    )


    # Same camera orientation
    ax.elev = -94
    ax.azim = 88
    ax.roll = 170


    # Same white background
    ax.set_facecolor(
        "white"
    )

    fig.patch.set_facecolor(
        "white"
    )

    fig.patch.set_alpha(
        1
    )

    ax.patch.set_alpha(
        1
    )


    ax.set_title(
        title,
        color="black"
    )


    # SHOW ONLY
    # NO savefig
    # Port: Amit's script did not save the split images; this does, named after the title.
    fig.savefig(_out_dir / (re.sub(r"[^A-Za-z0-9_.-]+", "_", str(title))[:120] + ".svg"), bbox_inches="tight")
    plt.show()


#%% Get Max-SI split ordered by polarity
# dendrite first, axon last

def get_colored_split(
    split_result
):

    polarity_result = (
        assign_axon_and_dendrite(
            split_result
        )
    )


    return navis.NeuronList(
        [
            polarity_result[
                "dendrite_neuron"
            ],

            polarity_result[
                "axon_neuron"
            ]
        ]
    )


#%% Pipeline

def run_neuron_batch_pipeline(
    nodes_dataframe,
    synapses_dataframe,
    neuron_id_col=None,
    flow_thresh=1,
    plot=True
):

    """
    Process every neuron supplied in nodes_dataframe.

    If plot=True:

        Figure 1:
            Original SFC split

        Figure 2:
            Maximum-SI split after determining
            which side is dendrite / axon.
    """


    id_column = (
        find_neuron_id_column(
            nodes_dataframe,
            requested_column=
                neuron_id_col
        )
    )


    # IMPORTANT:
    # preserve large FlyWire IDs exactly

    candidate_ids = (
        nodes_dataframe[
            id_column
        ]
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


    # =====================================================
    # LOOP THROUGH NEURONS
    # =====================================================

    for (
        neuron_number,
        raw_id
    ) in enumerate(
        candidate_ids,
        start=1
    ):


        print(
            f"\n"
            f"{neuron_number}/"
            f"{len(candidate_ids)}"
        )


        if isinstance(
            raw_id,
            np.integer
        ):

            neuron_id = int(
                raw_id
            )

        else:

            neuron_id = (
                raw_id
            )


        try:

            # =============================================
            # Load neuron + synapses
            # =============================================

            swc = upload_swc(
                neuron_id
            )


            neuron_synapses = (
                synapses_dataframe.loc[
                    (
                        synapses_dataframe[
                            "pre"
                        ]
                        == neuron_id
                    )
                    |
                    (
                        synapses_dataframe[
                            "post"
                        ]
                        == neuron_id
                    )
                ]
                .copy()
            )


            if len(neuron_synapses) == 0:

                raise ValueError(
                    "No matching synapses were found."
                )


            neuron = (
                heal_attach_princeton_non_process(
                    swc,
                    neuron_synapses
                )
            )


            total_pre = len(
                neuron.presynapses
            )

            total_post = len(
                neuron.postsynapses
            )


            # =============================================
            # ORIGINAL SFC SPLIT
            # =============================================

            flow_split = get_split(
                neuron,
                flow_thresh=
                    flow_thresh
            )

            print(len(flow_split))
            if len(flow_split) < 2:

                raise ValueError(
                    "get_split returned "
                    "fewer than two fragments."
                )


            # Same assumption as original analysis

            original_dendrite = (
                flow_split[0]
            )

            original_axon = (
                flow_split[-1]
            )


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


            original_SI, original_IG = (
                safe_SI_calc(

                    neuron_id=
                        neuron_id,

                    side_1_pre=
                        original_ax_pre,

                    side_1_post=
                        original_ax_post,

                    side_2_pre=
                        original_dend_pre,

                    side_2_post=
                        original_dend_post
                )
            )


            # ---------------------------------------------
            # Navis original SI
            # ---------------------------------------------

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
                    )
                    .reshape(-1)[0]
                )


            except Exception:

                navis_original_SI = (
                    np.nan
                )


            # =============================================
            # SHOW ORIGINAL SFC SPLIT
            # =============================================

            if plot:

                # First = dendrite
                # Intermediate = green
                # Last = axon

                original_colors = (
                    [DENDRITE_COLOR]
                    +
                    [LINKER_COLOR]
                    * max(
                        0,
                        len(flow_split) - 2
                    )
                    +
                    [AXON_COLOR]
                )


                show_split_image(

                    neuron_list=
                        flow_split,

                    colors=
                        original_colors,

                    title=(
                        f"Neuron {neuron_id}\n"
                        f"SFC split | "
                        f"SI = {original_SI:.4f}"
                    )
                )


            # =============================================
            # NODE-LEVEL SI
            # =============================================

            node_SI = (
                calculate_node_SI(
                    neuron
                )
            )


            valid_node_SI = (

                node_SI

                .dropna(
                    subset=[
                        "SI"
                    ]
                )

                .sort_values(
                    "SI",
                    ascending=False
                )

                .reset_index(
                    drop=True
                )
            )


            if len(valid_node_SI) == 0:

                raise ValueError(
                    "No valid node-level SI values."
                )


            # =============================================
            # MAXIMUM SI NODE
            # =============================================

            best_node = (
                valid_node_SI.iloc[0]
            )


            # =============================================
            # MINIMUM-P FISHER NODE
            # =============================================

            valid_fisher_nodes = (

                node_SI

                .dropna(
                    subset=[
                        "fisher_p"
                    ]
                )

                .sort_values(
                    "fisher_p",
                    ascending=True
                )

                .reset_index(
                    drop=True
                )
            )


            if (
                len(
                    valid_fisher_nodes
                )
                > 0
            ):

                best_fisher_node = (
                    valid_fisher_nodes
                    .iloc[0]
                )

            else:

                best_fisher_node = (
                    None
                )


            # =============================================
            # SPLIT AT MAX-SI NODE
            # =============================================

            split_result = (
                split_neuron_by_si_node(

                    neuron,
                    node_SI,

                    node_id=
                        best_node[
                            "node_id"
                        ]
                )
            )


            # =============================================
            # DETERMINE AXON / DENDRITE
            # =============================================

            polarity_result = (
                assign_axon_and_dendrite(
                    split_result
                )
            )


            axon_neuron = (
                polarity_result[
                    "axon_neuron"
                ]
            )

            dendrite_neuron = (
                polarity_result[
                    "dendrite_neuron"
                ]
            )


            split_colored = (
                navis.NeuronList(
                    [
                        dendrite_neuron,
                        axon_neuron
                    ]
                )
            )


            # =============================================
            # SHOW MAX-SI SPLIT
            # =============================================

            if plot:

                max_si_split = (
                    get_colored_split(
                        split_result
                    )
                )


                max_si_colors = [
                    DENDRITE_COLOR,
                    AXON_COLOR
                ]


                show_split_image(

                    neuron_list=
                        max_si_split,

                    colors=
                        max_si_colors,

                    title=(
                        f"Neuron {neuron_id}\n"
                        f"Max-SI split | "
                        f"SI = "
                        f"{best_node['SI']:.4f}"
                    )
                )

            
            total_synapses = (
                len(neuron.presynapses)
                + len(neuron.postsynapses)
            )
            
            
            # Regular SFC split
            sfc_dend_synapses = (
                len(original_dendrite.presynapses)
                + len(original_dendrite.postsynapses)
            )
            
            sfc_axon_synapses = (
                len(original_axon.presynapses)
                + len(original_axon.postsynapses)
            )
            
            
            # Max-SI split
            max_dend_synapses = (
                len(max_si_split[0].presynapses)
                + len(max_si_split[0].postsynapses)
            )
            
            max_axon_synapses = (
                len(max_si_split[1].presynapses)
                + len(max_si_split[1].postsynapses)
            )
            
            
            print(
                f"\nNeuron {neuron_id} | "
                f"Total synapses: {total_synapses}"
            )
            
            print(
                f"SFC: "
                f"dendrite = {sfc_dend_synapses}, "
                f"axon = {sfc_axon_synapses}, "
                f"assigned = {sfc_dend_synapses + sfc_axon_synapses}"
            )
            
            print(
                f"Max-SI: "
                f"dendrite = {max_dend_synapses}, "
                f"axon = {max_axon_synapses}, "
                f"assigned = {max_dend_synapses + max_axon_synapses}"
            )
            # =============================================
            # SAVE NEURON-LEVEL RESULTS TO DATAFRAME
            # NOT IMAGES
            # =============================================

            summary_results.append(
                {

                    "neuron_id":
                        neuron_id,


                    "total_pre":
                        total_pre,

                    "total_post":
                        total_post,

                    "total_synapses":
                        (
                            total_pre
                            + total_post
                        ),


                    # -------------------------
                    # Original SFC
                    # -------------------------

                    "original_SI":
                        original_SI,

                    "original_IG":
                        original_IG,

                    "navis_original_SI":
                        navis_original_SI,

                    "original_axon_pre":
                        original_ax_pre,

                    "original_axon_post":
                        original_ax_post,

                    "original_dendrite_pre":
                        original_dend_pre,

                    "original_dendrite_post":
                        original_dend_post,


                    # -------------------------
                    # Max-SI
                    # -------------------------

                    "best_node_id":
                        best_node[
                            "node_id"
                        ],

                    "best_node_SI":
                        best_node[
                            "SI"
                        ],

                    "best_node_IG":
                        best_node[
                            "IG"
                        ],


                    # -------------------------
                    # Fisher at Max-SI
                    # -------------------------

                    "best_node_fisher_p":
                        best_node[
                            "fisher_p"
                        ],

                    "best_node_fisher_q":
                        best_node[
                            "fisher_q"
                        ],

                    "best_node_fisher_odds_ratio":
                        best_node[
                            "fisher_odds_ratio"
                        ],

                    "best_node_fisher_log2_odds_ratio":
                        best_node[
                            "fisher_log2_odds_ratio"
                        ],


                    "best_side_1_pre":
                        best_node[
                            "side_1_pre"
                        ],

                    "best_side_1_post":
                        best_node[
                            "side_1_post"
                        ],

                    "best_side_2_pre":
                        best_node[
                            "side_2_pre"
                        ],

                    "best_side_2_post":
                        best_node[
                            "side_2_post"
                        ],


                    "best_node_pre_excluded":
                        best_node[
                            "node_pre_excluded"
                        ],

                    "best_node_post_excluded":
                        best_node[
                            "node_post_excluded"
                        ],


                    # -------------------------
                    # Minimum Fisher P node
                    # -------------------------

                    "best_fisher_node_id": (

                        best_fisher_node[
                            "node_id"
                        ]

                        if best_fisher_node
                        is not None

                        else np.nan
                    ),


                    "best_fisher_node_SI": (

                        best_fisher_node[
                            "SI"
                        ]

                        if best_fisher_node
                        is not None

                        else np.nan
                    ),


                    "best_fisher_node_IG": (

                        best_fisher_node[
                            "IG"
                        ]

                        if best_fisher_node
                        is not None

                        else np.nan
                    ),


                    "best_fisher_p": (

                        best_fisher_node[
                            "fisher_p"
                        ]

                        if best_fisher_node
                        is not None

                        else np.nan
                    ),


                    "best_fisher_q": (

                        best_fisher_node[
                            "fisher_q"
                        ]

                        if best_fisher_node
                        is not None

                        else np.nan
                    ),


                    "best_fisher_odds_ratio": (

                        best_fisher_node[
                            "fisher_odds_ratio"
                        ]

                        if best_fisher_node
                        is not None

                        else np.nan
                    ),


                    "best_fisher_log2_odds_ratio": (

                        best_fisher_node[
                            "fisher_log2_odds_ratio"
                        ]

                        if best_fisher_node
                        is not None

                        else np.nan
                    ),


                    "best_fisher_side_1_pre": (

                        best_fisher_node[
                            "side_1_pre"
                        ]

                        if best_fisher_node
                        is not None

                        else np.nan
                    ),


                    "best_fisher_side_1_post": (

                        best_fisher_node[
                            "side_1_post"
                        ]

                        if best_fisher_node
                        is not None

                        else np.nan
                    ),


                    "best_fisher_side_2_pre": (

                        best_fisher_node[
                            "side_2_pre"
                        ]

                        if best_fisher_node
                        is not None

                        else np.nan
                    ),


                    "best_fisher_side_2_post": (

                        best_fisher_node[
                            "side_2_post"
                        ]

                        if best_fisher_node
                        is not None

                        else np.nan
                    ),


                    # -------------------------
                    # Max-SI polarity
                    # -------------------------

                    "axon_side":
                        polarity_result[
                            "axon_side"
                        ],

                    "side_1_pre_fraction":
                        polarity_result[
                            "side_1_pre_fraction"
                        ],

                    "side_2_pre_fraction":
                        polarity_result[
                            "side_2_pre_fraction"
                        ]
                }
            )


            # =============================================
            # NODE-LEVEL OUTPUT
            # =============================================

            node_SI_to_save = (
                node_SI.copy()
            )


            node_SI_to_save.insert(
                0,
                "neuron_id",
                neuron_id
            )


            node_si_frames.append(
                node_SI_to_save
            )


            successful_count += 1


        # =================================================
        # FAILED NEURON
        # =================================================

        except Exception as error:

            print(
                f"Failed neuron "
                f"{neuron_id}: "
                f"{error}"
            )


            failures.append(
                {

                    "neuron_id":
                        neuron_id,

                    "error":
                        str(error)
                }
            )


    # =====================================================
    # COMBINE OUTPUTS
    # =====================================================

    summary_df = (
        pd.DataFrame(
            summary_results
        )
    )


    if len(node_si_frames) > 0:

        all_node_si_df = (
            pd.concat(
                node_si_frames,
                ignore_index=True
            )
        )

    else:

        all_node_si_df = (
            pd.DataFrame()
        )


    failures_df = (
        pd.DataFrame(
            failures
        )
    )


    print(
        f"\nSuccessful: "
        f"{successful_count}/"
        f"{len(candidate_ids)}"
    )


    return (
        summary_df,
        all_node_si_df,
        failures_df
    )


#%% Port option, not in Amit's script.
# "auto" (default): Amit's own selection if every cell in it has a skeleton in SWC_DIR. If even one is
#   missing, the whole selection is replaced (printed): up to 5 cells per band drawn, with the same seed,
#   from the KCg-m cells that have a skeleton. False: always Amit's selection (a cell without a skeleton
#   fails to load and is reported as a failure). True: always the replacement.
ONLY_CELLS_WITH_SKELETONS = "auto"
# The same layout upload_swc reads: SWC_DIR/<folder>/<id>.swc
_have = {int(p.stem) for p in SWC_DIR.glob("*/*.swc") if p.stem.isdigit()}

#%% Select rows of sorted SI_
KCg_m=KCg_m.sort_values(by='SI_dif')

#%%
bins = [
    (-0.10, -0.07),
    (-0.07, -0.06),
    (-0.06, -0.05),
    (-0.05, -0.04),
    (-0.04, -0.03)
]

SI_selected = pd.concat([
    KCg_m[
        (KCg_m["SI_dif"] >= low) &
        (KCg_m["SI_dif"] < high)
    ].sample(n=5, random_state=42)
    
    for low, high in bins
]).reset_index(drop=True)
_missing = ~SI_selected['neuron'].astype('int64').isin(_have)
if ONLY_CELLS_WITH_SKELETONS is True or (ONLY_CELLS_WITH_SKELETONS == "auto" and _missing.any()):
    print(f"Port: {int(_missing.sum())} of the {len(SI_selected)} cells Amit's settings select have no skeleton "
          "in SWC_DIR; drawing up to 5 per band from the KCg-m cells that have one.")
    _kc = KCg_m[KCg_m['neuron'].astype('int64').isin(_have)]
    SI_selected = pd.concat([
        _kc[
            (_kc["SI_dif"] >= low) &
            (_kc["SI_dif"] < high)
        ].pipe(lambda d: d.sample(n=min(5, len(d)), random_state=42))
        for low, high in bins
    ]).reset_index(drop=True)
#%%

#%%


#%% Run

(
    summary_df,
    all_node_si_df,
    failures_df

) = run_neuron_batch_pipeline(

    nodes_dataframe=
        SI_selected,

    synapses_dataframe=
        allsynapses,

    neuron_id_col=
        "neuron_id",

    flow_thresh=
        FLOW_THRESHOLD,

    plot=True
)
    #%%

(
    summary_df,
    all_node_si_df,
    failures_df

) = run_neuron_batch_pipeline(

    nodes_dataframe=
        SI_selected.iloc[[0]],

    synapses_dataframe=
        allsynapses,

    neuron_id_col=
        "neuron_id",

    flow_thresh=
        FLOW_THRESHOLD,

    plot=True
)