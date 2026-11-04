# Larval MaxSI split: every larval cell cut at the node with the highest SI, as step 09 cuts the adult cells, for
# Figure 2A. Runs separately from steps 01-09, on the larval raw files (Winding et al. 2023), in about 2 minutes.
# It writes data/larva_maxsi/ under either PIPELINE_RUN setting.
#
# Every cell of LARVA_NEURON_LIST_CSV is tried (step 09 tries only the cells the SFC split kept; there is no usable
# larval SFC table to take them from). For each cell:
#   1. its skeleton (LARVA_SWC_DIR, comma-separated with a header row) is converted to the space-separated form navis
#      reads, with a JSON metadata line, into LARVA_SWC_NAVIS_DIR. A converted file is reused while it is newer than
#      the raw one; delete the folder to convert every skeleton again;
#   2. the skeleton is healed (navis.heal_skeleton) and the cell's synapses are attached to their nearest skeleton
#      node (attach_synapses_larva2 below: synapses more than 10,000 nm from every node are dropped), after removing
#      the synapses from a cell onto itself;
#   3. the SI of every possible cut is computed (calculate_node_SI in methods/node_cut_si.py) and the cell is cut at
#      the node with the highest SI, the first such node on a tie (choose_cut_node below, step 09's, with METHOD
#      "maxsi"); the synapse ends on each side are labelled A or D (label_by_cut_node below, step 09's). Ends on the
#      cut node itself get no label.
# A cell whose split fails is recorded with its error and left out.
#
# Outputs (the MaxSI larval tables, in data/larva_maxsi/; config.py points LARVA_SI_FTR and LARVA_SYNAPSES_FTR at
# the first two under SPLIT_METHOD = "maxsi", so Figure 2A reads them):
#   LARVA_MAXSI_SI_FTR        one row per labelled cell: neuron, SI (at the cut), linker (always False: MaxSI has none)
#   LARVA_MAXSI_SYNAPSES_FTR  one row per synapse whose two ends are labelled: rowid (the larval synapse CSV's own
#                             row number, an ID within that file only), pre, post, pre_compartment, post_compartment
#   LARVA_MAXSI_SUMMARY_FTR   one row per labelled cell: the cut node, its SI, ties, and the counts on each side
#   LARVA_MAXSI_FAILURES_FTR  one row per cell whose split failed: neuron, error
#
# choose_cut_node and label_by_cut_node are copied from processing/09_maxsi_synapse_labels.py, which cannot be
# imported (it runs on import). Keep them identical to step 09's when step 09 changes.

#%% Imports
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import (METHODS_DIR, LARVA_SWC_DIR, LARVA_SWC_NAVIS_DIR, LARVA_SYNAPSE_TABLE_CSV, LARVA_NEURON_LIST_CSV,
                    LARVA_MAXSI_SI_FTR, LARVA_MAXSI_SYNAPSES_FTR, LARVA_MAXSI_SUMMARY_FTR, LARVA_MAXSI_FAILURES_FTR)

sys.path.insert(0, str(METHODS_DIR))
from node_cut_si import calculate_node_SI, get_descendants_from_node

import navis
import numpy as np
import pandas as pd

METHOD = "maxsi"   # read by choose_cut_node, as in step 09


#%% Functions
def convert_swc(src, dst):
    """Amit's larval SWC files are comma-separated with a header row; navis reads space-separated rows. The metadata
    line must be JSON (double quotes), or navis 1.7.0 cannot read the file."""
    df = pd.read_csv(src)
    df.columns = ['id', 'type', 'x', 'y', 'z', 'r', 'parent_id']
    tmp = dst.with_name(dst.name + f".{os.getpid()}.tmp")   # renamed when complete, so a stopped run leaves no half file
    with open(tmp, 'w') as f:
        f.write('# Meta: ' + json.dumps({"id": src.stem}) + '\n')
        df.to_csv(f, sep=' ', index=False, header=False, mode='a')
    os.replace(tmp, dst)


def attach_synapses_larva2(x, syn, min_score=30):
    """Each synapse of the cell goes to its nearest skeleton node; synapses more than 10,000 nm from every node are
    dropped. Amit's attach_synapses_larva2, without its unused arguments."""
    if min_score:
        syn = syn[syn.cleft_score >= min_score]
    if isinstance(x, navis.core.BaseNeuron):
        x = navis.NeuronList([x])
    for n in x:
        cols = ['pre_x', 'pre_y', 'pre_z', 'cleft_score', 'post', 'rowid']
        presyn = syn.loc[syn.pre == np.int64(n.id), cols].rename(
            {'pre_x': 'x', 'pre_y': 'y', 'pre_z': 'z', 'post': 'partner_id'}, axis=1)
        presyn['type'] = 'pre'
        cols = ['post_x', 'post_y', 'post_z', 'cleft_score', 'pre', 'rowid']
        postsyn = syn.loc[syn.post == np.int64(n.id), cols].rename(
            {'post_x': 'x', 'post_y': 'y', 'post_z': 'z', 'pre': 'partner_id'}, axis=1)
        postsyn['type'] = 'post'
        connectors = pd.concat((presyn, postsyn), axis=0, ignore_index=True)
        connectors['type'] = connectors['type'].astype('category')
        tree = navis.neuron2KDTree(n, data='nodes')
        dist, ix = tree.query(connectors[['x', 'y', 'z']].values)
        too_far = dist > 10_000
        if any(too_far):
            connectors = connectors[~too_far].copy()
            ix = ix[~too_far]
        connectors['node_id'] = n.nodes.node_id.values[ix]
        connectors.insert(0, 'connector_id', np.arange(connectors.shape[0]))
        n.connectors = connectors
    return x


def label_by_cut_node(neuron, node_row):
    """
    Label every synapse of a healed TreeNeuron with attached synapses as axon or
    dendrite, for the cut at node_row (a row of calculate_node_SI's table).

    Returns the labelled connectors (synapse_id, type, label) and a dict of side
    counts and fractions. Synapses on the cut node are not returned.
    """
    cut_node_id = node_row["node_id"]
    nodes = neuron.nodes

    children = nodes.loc[nodes["parent_id"] == cut_node_id, "node_id"].tolist()
    if len(children) != 1:
        raise ValueError(f"Node {cut_node_id} has {len(children)} children; expected exactly 1.")

    side_1_ids = get_descendants_from_node(neuron, children[0])
    side_2_ids = set(nodes["node_id"]) - side_1_ids - {cut_node_id}

    connectors = neuron.connectors[["synapse_id", "type", "node_id"]].copy()
    connectors["type"] = connectors["type"].astype(str)

    on_side_1 = connectors["node_id"].isin(side_1_ids)
    on_side_2 = connectors["node_id"].isin(side_2_ids)
    on_cut = connectors["node_id"] == cut_node_id
    if (~(on_side_1 | on_side_2 | on_cut)).any():
        raise ValueError(f"{int((~(on_side_1 | on_side_2 | on_cut)).sum())} connectors "
                         "are outside both sides.")

    side_1 = connectors.loc[on_side_1]
    side_2 = connectors.loc[on_side_2]
    if len(side_1) == 0:
        raise ValueError("Side 1 contains no synapses.")
    if len(side_2) == 0:
        raise ValueError("Side 2 contains no synapses.")

    # the counts must be the ones the cut's SI was computed from
    counts = {
        "side_1_pre": int(side_1["type"].eq("pre").sum()),
        "side_1_post": int(side_1["type"].eq("post").sum()),
        "side_2_pre": int(side_2["type"].eq("pre").sum()),
        "side_2_post": int(side_2["type"].eq("post").sum()),
    }
    for name, value in counts.items():
        if value != int(node_row[name]):
            raise ValueError(f"{name} is {value} on the labelled cell but {node_row[name]} in the node table.")

    side_1_pre_fraction = side_1["type"].eq("pre").mean()
    side_2_pre_fraction = side_2["type"].eq("pre").mean()
    whole_pre_fraction = pd.concat([side_1, side_2], axis=0)["type"].eq("pre").mean()

    if np.isclose(side_1_pre_fraction, side_2_pre_fraction):
        raise ValueError("Both sides have the same presynaptic fraction.")

    if side_1_pre_fraction > whole_pre_fraction:
        axon_side = 1
    elif side_2_pre_fraction > whole_pre_fraction:
        axon_side = 2
    else:
        raise ValueError("Could not determine axon/dendrite polarity.")

    labelled = connectors.loc[on_side_1 | on_side_2, ["synapse_id", "type"]].copy()
    labelled["label"] = np.where(on_side_1[on_side_1 | on_side_2] == (axon_side == 1), "A", "D")

    return labelled, {
        **counts,
        "node_pre_excluded": int((on_cut & connectors["type"].eq("pre")).sum()),
        "node_post_excluded": int((on_cut & connectors["type"].eq("post")).sum()),
        "axon_side": axon_side,
        "side_1_pre_fraction": side_1_pre_fraction,
        "side_2_pre_fraction": side_2_pre_fraction,
        "whole_pre_fraction": whole_pre_fraction,
    }


def choose_cut_node(node_SI):
    """
    The cut node under METHOD: its row of calculate_node_SI's table, the candidate nodes (rows) and those tied with
    it at the chosen value. On a tie the first node in the table's order is taken.
    """
    if METHOD == "maxsi":
        candidates = node_SI.dropna(subset=["SI"])
        if len(candidates) == 0:
            raise ValueError("No valid node-level SI values.")
        # highest SI; on a tie, the first node in the table's order
        best = candidates.loc[candidates["SI"].idxmax()]
        tied = candidates.loc[candidates["SI"] == best["SI"]]
    elif METHOD == "minfisherp":
        candidates = node_SI.loc[node_SI["fisher_p"].notna()]
        if len(candidates) == 0:
            raise ValueError("No node with a Fisher p-value.")
        # lowest p (p = 0 included); on a tie, the first node in the table's order
        best = candidates.loc[candidates["fisher_p"].idxmin()]
        tied = candidates.loc[candidates["fisher_p"] == best["fisher_p"]]
    else:   # "maxphi"
        a = node_SI["side_1_pre"].to_numpy(dtype=np.float64)
        b = node_SI["side_1_post"].to_numpy(dtype=np.float64)
        c = node_SI["side_2_pre"].to_numpy(dtype=np.float64)
        d = node_SI["side_2_post"].to_numpy(dtype=np.float64)
        den = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
        with np.errstate(divide="ignore", invalid="ignore"):
            phi = np.where(den > 0, np.abs((a * d - b * c) / den), np.nan)
        node_SI = node_SI.assign(Phi=phi)
        candidates = node_SI.loc[node_SI["Phi"].notna()]
        if len(candidates) == 0:
            raise ValueError("No node with a phi value.")
        # highest |phi|; on a tie, the first node in the table's order
        best = candidates.loc[candidates["Phi"].idxmax()]
        tied = candidates.loc[candidates["Phi"] == best["Phi"]]
    return best, candidates, tied


#%% Inputs
t0 = time.time()
for _p in (LARVA_NEURON_LIST_CSV, LARVA_SYNAPSE_TABLE_CSV, LARVA_SWC_DIR):
    if not _p.exists():
        raise SystemExit(f"Missing larval input {_p}; see docs/data_availability.md for the three larval raw files.")
nlist = pd.read_csv(LARVA_NEURON_LIST_CSV)
cells = sorted(set(nlist.loc[nlist.neuron != -1, "neuron"].astype(np.int64)))

syn = pd.read_csv(LARVA_SYNAPSE_TABLE_CSV)
# the raw table's columns, by position (as larva_process.py names them)
syn.columns = ['rowid', 'pre_skeleton_node_id', 'connector_id', 'pre_x', 'pre_y', 'pre_z',
               'connector_x', 'connector_y', 'connector_z', 'post_skeleton_node_id',
               'post_x', 'post_y', 'post_z', 'pre', 'post']
assert syn['rowid'].is_unique
assert syn['pre'].dtype == np.int64 and syn['post'].dtype == np.int64, "pre/post IDs must read as int64"
n_rows = len(syn)
n_self = int((syn.pre == syn.post).sum())
syn = syn[syn.pre != syn.post].copy()
syn['cleft_score'] = 50   # the larval table has no score; as larva_process.py, every synapse passes min_score

LARVA_SWC_NAVIS_DIR.mkdir(parents=True, exist_ok=True)

#%% Split every cell
labels, summaries, failures = [], [], []
for nid in cells:
    try:
        src = LARVA_SWC_DIR / f"{nid}.swc"
        dst = LARVA_SWC_NAVIS_DIR / f"{nid}.swc"
        if not src.exists():
            raise FileNotFoundError(f"no skeleton {src.name}")
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            convert_swc(src, dst)
        n = navis.read_swc(dst)
        n.id = np.int64(nid)
        n = navis.heal_skeleton(n)
        n = attach_synapses_larva2(n, syn)
        cell = n[0]
        if len(cell.connectors) == 0:
            raise ValueError("No matching synapses were found.")
        cell.connectors['synapse_id'] = cell.connectors['rowid'].astype(np.int64)
        node_SI = calculate_node_SI(n, fisher=False)
        if len(node_SI) == 0:
            raise ValueError("No valid node-level SI values.")
        best, candidates, tied = choose_cut_node(node_SI)
        labelled, sides = label_by_cut_node(cell, best)
        n_expected = len(cell.connectors) - sides["node_pre_excluded"] - sides["node_post_excluded"]
        if len(labelled) != n_expected or labelled["synapse_id"].duplicated().any():
            raise ValueError(f"{len(labelled)} labels for {n_expected} synapses off the cut node.")
        labels.append(pd.DataFrame({"synapse_id": labelled["synapse_id"].to_numpy(np.int64),
                                    "neuron": np.int64(nid),
                                    "side": labelled["type"].to_numpy(object),
                                    "label": labelled["label"].to_numpy(object)}))
        summaries.append({"neuron": np.int64(nid), "best_node_id": np.int64(best["node_id"]),
                          "MaxSI_SI": float(best["SI"]), "n_tied_max_SI": len(tied),
                          "n_tied_max_SI_distinct_sides": len(tied[["side_1_pre", "side_1_post", "side_2_pre",
                                                                    "side_2_post"]].drop_duplicates()),
                          "n_candidates": len(node_SI), "n_attached_pre": len(cell.presynapses),
                          "n_attached_post": len(cell.postsynapses), "n_labelled": len(labelled), **sides})
    except MemoryError:
        raise   # as step 09: running out of memory is not a failed cell
    except Exception as e:
        failures.append({"neuron": np.int64(nid), "error": f"{type(e).__name__}: {e}"[:300]})

#%% Tables
fails = pd.DataFrame(failures, columns=["neuron", "error"])
if not labels:
    raise SystemExit(f"No larval cell was split. First errors:\n{fails.head(5).to_string(index=False)}")
labels = pd.concat(labels, ignore_index=True)
summary = pd.DataFrame(summaries)
pre = labels[labels.side == "pre"][["synapse_id", "neuron", "label"]].rename(
    columns={"synapse_id": "rowid", "neuron": "pre", "label": "pre_compartment"})
post = labels[labels.side == "post"][["synapse_id", "neuron", "label"]].rename(
    columns={"synapse_id": "rowid", "neuron": "post", "label": "post_compartment"})
with_pre = syn[["rowid", "pre", "post"]].merge(pre, on=["rowid", "pre"], validate="one_to_one")
synapses = with_pre.merge(post, on=["rowid", "post"], validate="one_to_one").reset_index(drop=True)
si = pd.DataFrame({"neuron": summary.neuron.to_numpy(np.int64), "SI": summary.MaxSI_SI.to_numpy(np.float64),
                   "linker": False})

for _f in (LARVA_MAXSI_SI_FTR, LARVA_MAXSI_SYNAPSES_FTR, LARVA_MAXSI_SUMMARY_FTR, LARVA_MAXSI_FAILURES_FTR):
    _f.parent.mkdir(parents=True, exist_ok=True)
si.to_feather(LARVA_MAXSI_SI_FTR)
synapses.to_feather(LARVA_MAXSI_SYNAPSES_FTR)
summary.to_feather(LARVA_MAXSI_SUMMARY_FTR)
fails.to_feather(LARVA_MAXSI_FAILURES_FTR)

print(f"navis {navis.__version__}; {time.time() - t0:.0f} s")
print(f"synapse table rows {n_rows}; self-synapse rows dropped {n_self}")
print(f"cells tried {len(cells)}; labelled {len(summary)}; failed {len(fails)}")
for err, k in fails["error"].value_counts().items():
    print(f"  failed, {k} cells: {err}")
print(f"MaxSI SI over labelled cells: median {summary.MaxSI_SI.median():.4f}")
print(f"synapses (self-synapses out) {len(syn)}; with the pre end labelled {len(with_pre)}; "
      f"with both ends labelled {len(synapses)}")
print("Figure 2A reads these tables when SPLIT_METHOD = \"maxsi\" in config.py")
