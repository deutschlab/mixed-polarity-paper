# -*- coding: utf-8 -*-
"""
Per-synapse compartment labels for the MaxSI, MinFisherP or MaxPhi split.

For every cell of NEURON_TABLE_SFC_FTR (every super-class): load the skeleton, heal it
(navis.heal_skeleton), attach its synapses from SYNAPSE_TABLE_RAW_FTR
(methods_all.attach_synapses_princeton, which keeps synapse_id; the same two calls as
methods_all.heal_attach_princeton, made here so that an error keeps its own text), compute
the side counts and SI of every candidate cut as step 08 does (methods/node_cut_si.py), and
cut at one node. The METHOD setting picks it:
- "maxsi": the node with the highest SI (nodes with no SI are left out);
- "minfisherp": the node with the lowest p of the two-sided Fisher exact test on its side
  counts (nodes with no p are left out; a node with no SI can be chosen). The test runs at
  every node, so this method is slower than the other two;
- "maxphi": the node with the highest |phi| of its side counts, phi = (ad - bc) /
  sqrt((a + b)(c + d)(a + c)(b + d)) with a, b = side 1 pre, post and c, d = side 2 pre,
  post (nodes where the square root is 0 are left out).
When several nodes share the chosen value, the first in the node table's order is taken
(pandas idxmax or idxmin); the summary records how many nodes tie and how many different
side counts they give, and for MaxSI also how many nodes are within numpy isclose of the
highest SI (for information). Another tied node can change the labels even with the same
side counts, by putting different synapses on each side. A cell's SI is the SI of the
chosen cut. MinFisherP chooses without regard to the SI, so the chosen cut can have an empty
SI (NaN, when the synapses off the cut node are all of one kind). Such a cut leaves both
sides with the same presynaptic fraction, so the cell fails; the run prints the count of
labelled cells with an empty SI, which is therefore 0. A cell with no candidate node fails.

The tables inherit the SFC tables' cells and synapse rows: only the cells of
NEURON_TABLE_SFC_FTR are tried (a cell the SFC split failed on is never tried here), and
only the rows of SYNAPSE_TABLE_SFC_FTR are labelled.

Each method writes to its own folders (config.py): METHOD = "maxsi" to the MAXSI_* paths
(the _maxsi folders), "minfisherp" to the MINFISHERP_* paths (_minfisherp), "maxphi" to the
MAXPHI_* paths (_maxphi). STEP09_METHOD in the environment overrides the setting. Below,
the MaxSI names stand for the method's: MaxSI_SI is the summary's SI column under
"maxsi", MinFisherP_SI and MaxPhi_SI under the other two.

Side 1 is the cut node's one child and its descendants, side 2 the rest of the cell
without the cut node. The synapses on the cut node itself get no label. The axon is the
side whose presynaptic fraction is above the whole cell's, both counted without the cut
node's synapses. A cell fails, and gets no labels, when the two sides have the same
presynaptic fraction (numpy isclose) or a side has no synapse.

Before anything runs, the step stops if a skeleton file name appears in more than one
subfolder of SWC_DIR (methods_all.upload_swc would read whichever folder it meets first).

Writes per batch of 500 cells, to MAXSI_LABEL_BATCH_DIR:
- labels_batch_NNNN.ftr: synapse_id, neuron, side ('pre' or 'post'), label ('A' or 'D');
- summary_batch_NNNN.ftr: one row per cell with its cut node, MaxSI_SI, the ties at the
  highest SI (n_tied_max_SI, n_tied_max_SI_distinct_sides, n_close_max_SI), side counts
  and which side is the axon. Under "minfisherp": MinFisherP_SI, the cut's fisher_p and
  the ties at the lowest p (n_tied_min_fisher_p, n_tied_min_fisher_p_distinct_sides);
  under "maxphi": MaxPhi_SI, the cut's Phi and the ties at the highest Phi (n_tied_max_Phi,
  n_tied_max_Phi_distinct_sides);
- failures_batch_NNNN.ftr: neuron and error, for cells with no labels.
Each file is written under a temporary name and then renamed, the failures file last, and
a batch is skipped if its three files exist. batch_run.json records the method, the cell
list, the batch size, the raw synapse table, the number of skeleton files under SWC_DIR with
a hash of their sorted paths and sizes, the navis version, and a hash of the code that labels a
cell (sections 3 to 6 of this script, methods/node_cut_si.py and methods/methods_all.py; not
the settings); the run stops if the folder holds batches
of a run where any of these differ. A MemoryError stops the run instead of being recorded
as a failed cell.

Before writing any table the step stops if
- a cell failed for want of a usable skeleton (no file under SWC_DIR, or a file navis
  cannot read), unless ACCEPT_CELLS_WITHOUT_SKELETON is True;
- the number of failed cells is not ACCEPT_FAILED_CELLS (None means none may fail);
- no cell was labelled;
- the batches do not hold every cell of the run exactly once;
- some but not all of the three tables below exist (at the start too).
It prints the count and the errors, so the setting can be filled in after reading them.

The combine section then writes
- MAXSI_SYNAPSE_TABLE_FTR: the rows, columns and dtypes of SYNAPSE_TABLE_SFC_FTR, with
  comp = pre label + post label and SI_pre / SI_post = MaxSI_SI of each end. A row with
  an end that has no label (the cell failed, the synapse sits on the cut node, or it was
  more than 10,000 nm from the skeleton) is left out; the number is printed, and so is
  the number of labelled ends that have no row in the SFC synapse table.
- MAXSI_NEURON_TABLE_FTR: the rows and columns of NEURON_TABLE_SFC_FTR, keeping, as step
  05 does, only the cells that have an SI and a row in the synapse table: a cell that
  failed here is dropped, and the number is printed (a MinFisherP cell labelled with an
  empty SI is kept). SI = MaxSI_SI. Recounted from
  MAXSI_SYNAPSE_TABLE_FTR with step 05's code (05:223-256), because they describe the
  split: pre_A ... post_L, axon_correct, dend_correct. linker = 0, as the split has
  no linker compartment. Every other column is the SFC table's, including the cell
  properties out_synapses_count, in_synapses_count, the degree and partner columns and
  the six ratio columns: step 07's PCA and several figures use them as features of the
  cell, which must not depend on the split. The rows keep the SFC table's order. The
  step prints how many kept cells have an empty axon_correct or dend_correct that the
  SFC table fills, since the figures drop such cells.
- MAXSI_SI_UPDATED_FTR: root_id (int64) and SI = MaxSI_SI for every labelled cell, in
  the format of SI_updated.ftr (without its stored index). It holds every labelled cell, so it can
  hold cells the neuron table drops; a cell whose split failed has no row, so the scripts that
  read SI_updated.ftr leave it out.
The three tables are written together, each to a .tmp file and then renamed, the SI table
last: if all exist, the run stops at the start and none is overwritten.

config.SPLIT_METHOD = "maxsi" points SYNAPSE_TABLE_FTR, NEURON_TABLE_FTR and
SI_UPDATED_FTR at these three tables ("minfisherp" and "maxphi" at those of the other two
methods).

ONLY_LOCAL_SKELETONS = True, or a LIMIT, runs only the cells whose skeleton is under
SWC_DIR (the first LIMIT of them, by ID) and writes everything to MAXSI_SUBSET_DIR, for
checking (MINFISHERP_SUBSET_DIR, MAXPHI_SUBSET_DIR for the other methods). The subset
synapse table keeps only the rows whose two ends are both cells of the run, so its counts, and pre_A ... post_L in the subset neuron table, are not the full
cells' counts.

N_SHARDS and SHARD (or STEP09_N_SHARDS and STEP09_SHARD in the environment) split the labelling
over N copies run at once: copy k runs the batches whose number, counted from 0, leaves k when
divided by N, reads only the synapses of their cells, and stops after its batches. A last run
with N_SHARDS = 1 finds every batch done and writes the tables.
"""
#%%1 Imports

import sys
from pathlib import Path
# Make repo root importable so config.py can be found from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (METHODS_DIR, SWC_DIR, NEURON_TABLE_SFC_FTR, SYNAPSE_TABLE_SFC_FTR,
                    SYNAPSE_TABLE_RAW_FTR, MAXSI_LABEL_BATCH_DIR, MAXSI_SUBSET_DIR,
                    MAXSI_SYNAPSE_TABLE_FTR, MAXSI_NEURON_TABLE_FTR, MAXSI_SI_UPDATED_FTR)
from config import (MINFISHERP_LABEL_BATCH_DIR, MINFISHERP_SUBSET_DIR, MINFISHERP_SYNAPSE_TABLE_FTR,
                    MINFISHERP_NEURON_TABLE_FTR, MINFISHERP_SI_UPDATED_FTR,
                    MAXPHI_LABEL_BATCH_DIR, MAXPHI_SUBSET_DIR, MAXPHI_SYNAPSE_TABLE_FTR,
                    MAXPHI_NEURON_TABLE_FTR, MAXPHI_SI_UPDATED_FTR)

sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
from node_cut_si import calculate_node_SI, get_descendants_from_node

import gc
import hashlib
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.feather as feather


#%%2 Settings

# The cut rule: "maxsi" (highest SI), "minfisherp" (lowest Fisher p) or "maxphi" (highest |phi|). Each writes to its
# own folders. STEP09_METHOD in the environment overrides it.
METHOD = "maxsi"
METHOD = os.environ.get("STEP09_METHOD", METHOD)

BATCH_SIZE = 500

# True: only the cells whose skeleton is under SWC_DIR, written to MAXSI_SUBSET_DIR
ONLY_LOCAL_SKELETONS = False

# None, or the number of cells to keep (the first by ID, from the cells with a local skeleton)
LIMIT = None

# A cell with no usable skeleton under SWC_DIR fails. False: stop before writing the tables if any cell
# failed for that reason. True: count them with the other failed cells.
ACCEPT_CELLS_WITHOUT_SKELETON = False

# The tables are written only if exactly this many cells failed (None: only if none failed).
# Run once, read the count and the errors the run prints, then set it. The count differs between methods.
# STEP09_ACCEPT_FAILED_CELLS in the environment overrides it.
ACCEPT_FAILED_CELLS = None
if os.environ.get("STEP09_ACCEPT_FAILED_CELLS"):
    ACCEPT_FAILED_CELLS = int(os.environ["STEP09_ACCEPT_FAILED_CELLS"])

# Shards: N copies of the labelling at once. N_SHARDS = 1 runs every batch and then writes the tables.
# With N_SHARDS = N, the copy with SHARD = k (0 to N-1) runs the batches whose batch_number (from 0, section 6) leaves
# k when divided by N, reads only the synapses of their cells, and stops after its batches. The batches are dealt
# from the full list, so a copy that is restarted gets the same ones. Once every copy has finished, a run with
# N_SHARDS = 1 finds every batch done and writes the tables. STEP09_N_SHARDS and STEP09_SHARD in the environment
# override the two settings, to launch the copies from a shell.
N_SHARDS = 1
SHARD = 0
N_SHARDS = int(os.environ.get("STEP09_N_SHARDS", N_SHARDS))
SHARD = int(os.environ.get("STEP09_SHARD", SHARD))
if not (N_SHARDS >= 1 and 0 <= SHARD < N_SHARDS):
    raise SystemExit(f"SHARD must be 0 to N_SHARDS - 1 (N_SHARDS = {N_SHARDS}, SHARD = {SHARD})")

# per method: the prefix of its config names; its label batch folder, subset folder, synapse, neuron and SI tables;
# and the names of its summary columns (the method's SI, the cut's value, the ties at that value)
METHODS = {
    "maxsi": ("MAXSI", MAXSI_LABEL_BATCH_DIR, MAXSI_SUBSET_DIR, MAXSI_SYNAPSE_TABLE_FTR, MAXSI_NEURON_TABLE_FTR,
              MAXSI_SI_UPDATED_FTR, "MaxSI_SI", None, "n_tied_max_SI", "n_tied_max_SI_distinct_sides"),
    "minfisherp": ("MINFISHERP", MINFISHERP_LABEL_BATCH_DIR, MINFISHERP_SUBSET_DIR, MINFISHERP_SYNAPSE_TABLE_FTR,
                   MINFISHERP_NEURON_TABLE_FTR, MINFISHERP_SI_UPDATED_FTR, "MinFisherP_SI", "fisher_p",
                   "n_tied_min_fisher_p", "n_tied_min_fisher_p_distinct_sides"),
    "maxphi": ("MAXPHI", MAXPHI_LABEL_BATCH_DIR, MAXPHI_SUBSET_DIR, MAXPHI_SYNAPSE_TABLE_FTR, MAXPHI_NEURON_TABLE_FTR,
               MAXPHI_SI_UPDATED_FTR, "MaxPhi_SI", "Phi", "n_tied_max_Phi", "n_tied_max_Phi_distinct_sides"),
}
if METHOD not in METHODS:
    raise SystemExit(f'METHOD must be "maxsi", "minfisherp" or "maxphi", not {METHOD!r}')
(PREFIX, LABEL_BATCH_DIR, SUBSET_DIR, SYNAPSE_TABLE_OUT_FTR, NEURON_TABLE_OUT_FTR, SI_UPDATED_OUT_FTR,
 SI_COLUMN, VALUE_COLUMN, TIED_COLUMN, TIED_SIDES_COLUMN) = METHODS[METHOD]
METHOD_NAME = SI_COLUMN[:-3]   # MaxSI, MinFisherP or MaxPhi, for the messages
print(f"Method: {METHOD}")

SUBSET_RUN = ONLY_LOCAL_SKELETONS or LIMIT is not None
if SUBSET_RUN:
    batch_dir = SUBSET_DIR / "batches"
    batch_dir_setting = f"{PREFIX}_SUBSET_DIR"
    synapse_table_out = SUBSET_DIR / SYNAPSE_TABLE_OUT_FTR.name
    neuron_table_out = SUBSET_DIR / NEURON_TABLE_OUT_FTR.name
    si_updated_out = SUBSET_DIR / SI_UPDATED_OUT_FTR.name
else:
    batch_dir = LABEL_BATCH_DIR
    batch_dir_setting = f"{PREFIX}_LABEL_BATCH_DIR"
    synapse_table_out = SYNAPSE_TABLE_OUT_FTR
    neuron_table_out = NEURON_TABLE_OUT_FTR
    si_updated_out = SI_UPDATED_OUT_FTR

NO_SKELETON = "No usable skeleton"   # the start of the error text of a cell that failed for want of one


def check_final_tables():
    """True if the three tables all exist (none is overwritten), False if none does; stops if only some do."""
    tables = [synapse_table_out, neuron_table_out, si_updated_out]
    exist = [p.exists() for p in tables]
    if all(exist):
        print(f"The three {METHOD_NAME} tables already exist; not overwriting. Move or rename them to write new ones.")
        return True
    if any(exist):
        raise SystemExit(f"Only some of the three {METHOD_NAME} tables exist ({[p.name for p, e in zip(tables, exist) if e]} "
                         f"in {synapse_table_out.parent}), perhaps from a run that stopped. Move them away first.")
    return False


if check_final_tables():   # stops here rather than redo every batch and then write nothing
    raise SystemExit(f"Nothing to do: the three {METHOD_NAME} tables exist. Move them away to build new ones.")


#%%3 Cells to run

# upload_swc reads <SWC_DIR>/<subfolder>/<id>.swc from whichever subfolder it meets first, in no fixed order
swc_files = sorted(p.relative_to(SWC_DIR).as_posix() for p in SWC_DIR.glob("*/*.swc"))
swc_names = pd.Series([f.split("/")[-1] for f in swc_files], dtype=object)
repeated = sorted(set(swc_names[swc_names.duplicated(keep=False)]))
if repeated:
    raise SystemExit(f"{len(repeated):,} skeleton file names appear in more than one subfolder of {SWC_DIR}, "
                     f"for example {repeated[:5]}. Keep one copy of each.")
local = {np.int64(name[:-4]) for name in swc_names if name[:-4].isdigit()}

# what the batches are made from, besides the cells and the synapse table (check_batch_record)
swc_fingerprint = "\n".join(f"{f}\t{(SWC_DIR / f).stat().st_size}" for f in swc_files)
skeleton_record = {"method": METHOD,   # batches of one method are never mixed with another's
                   "swc_files": len(swc_files),
                   "swc_files_sha256": hashlib.sha256(swc_fingerprint.encode()).hexdigest(),
                   "navis_version": navis.__version__,
                   # the code that labels a cell too, so that batches made by two versions of it are never mixed;
                   # sections 3 to 6 of this script, not the settings, so that changing one (ACCEPT_FAILED_CELLS) keeps them
                   "code_sha256": hashlib.sha256(
                       Path(__file__).read_text(encoding="utf-8").split("\n#%%3 ")[1].split("\n#%%7 ")[0].encode()
                       + (METHODS_DIR / "node_cut_si.py").read_bytes()
                       + (METHODS_DIR / "methods_all.py").read_bytes()).hexdigest()}
print(f"Skeleton files under {SWC_DIR}: {len(swc_files):,}; navis {navis.__version__}")

neuron_table = pd.read_feather(NEURON_TABLE_SFC_FTR)
assert neuron_table["neuron"].dtype == np.int64, neuron_table["neuron"].dtype

all_neurons = neuron_table["neuron"].drop_duplicates().reset_index(drop=True)

if SUBSET_RUN:
    all_neurons = pd.Series(sorted(set(all_neurons) & local), dtype=np.int64)
    if LIMIT is not None:
        all_neurons = all_neurons.iloc[:LIMIT].reset_index(drop=True)
    print(f"Subset run: {len(all_neurons):,} cells with a skeleton under {SWC_DIR}")

print("Cells:", len(all_neurons))


#%%4 Load the raw synapses

SYNAPSE_COLUMNS = ["synapse_id", "pre_x", "pre_y", "pre_z",
                   "post_x", "post_y", "post_z", "size", "pre", "post"]

if N_SHARDS > 1:
    # the cells of this copy's batches (batch_number % N_SHARDS == SHARD in section 6); every copy keeps the full
    # cell list in all_neurons, for the batches and their record
    shard_neurons = all_neurons[(np.arange(len(all_neurons)) // BATCH_SIZE) % N_SHARDS == SHARD]
    if len(shard_neurons) == 0:
        raise SystemExit(f"Shard {SHARD} of {N_SHARDS} has no batch: N_SHARDS is above the number of batches")
    print(f"Shard {SHARD} of {N_SHARDS}: {len(shard_neurons):,} cells")

if SUBSET_RUN or N_SHARDS > 1:
    _ids = (all_neurons if N_SHARDS == 1 else shard_neurons).tolist()
    allsynapses = (ds.dataset(str(SYNAPSE_TABLE_RAW_FTR), format="feather")
                   .to_table(columns=SYNAPSE_COLUMNS,
                             filter=ds.field("pre").isin(_ids) | ds.field("post").isin(_ids))
                   .to_pandas())
else:
    allsynapses = pd.read_feather(SYNAPSE_TABLE_RAW_FTR, columns=SYNAPSE_COLUMNS)

assert allsynapses["pre"].dtype == np.int64 and allsynapses["post"].dtype == np.int64
n_self = int((allsynapses["pre"] == allsynapses["post"]).sum())
if n_self:
    raise SystemExit(f"{n_self:,} self-synapses in {SYNAPSE_TABLE_RAW_FTR.name}; the {METHOD_NAME} split "
                     "leaves them out (RAW_TABLE_KEEP_SELF_SYNAPSES = False in config.py)")
print("Synapses read:", len(allsynapses))


#%%5 Label the synapses of one cell

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


def maxsi_labels_one_cell(neuron_id, neuron_synapses):
    """Labels and summary row of one cell; raises on any failure."""
    if np.int64(neuron_id) not in local:
        raise FileNotFoundError(f"{NO_SKELETON}: no {neuron_id}.swc under SWC_DIR.")
    swc = upload_swc(neuron_id)
    if swc is None:
        raise OSError(f"{NO_SKELETON}: {neuron_id}.swc is under SWC_DIR but navis could not read it.")
    if len(neuron_synapses) == 0:
        raise ValueError("No matching synapses were found.")

    # methods_all.heal_attach_princeton's two calls, without its bare except
    healed = navis.heal_skeleton(swc)
    neuron = attach_synapses_princeton(healed, neuron_synapses)

    # the same object step 08 passes to calculate_node_SI (a NeuronList of one cell); the Fisher test runs only for
    # MinFisherP
    node_SI = calculate_node_SI(neuron, fisher=(METHOD == "minfisherp"))
    if len(node_SI) == 0:
        raise ValueError("No valid node-level SI values.")
    best, valid_node_SI, tied = choose_cut_node(node_SI)

    cell = neuron[0] if isinstance(neuron, navis.NeuronList) else neuron
    labelled, sides = label_by_cut_node(cell, best)

    # every attached synapse gets one label, except those on the cut node
    n_expected = len(cell.connectors) - sides["node_pre_excluded"] - sides["node_post_excluded"]
    if len(labelled) != n_expected or labelled["synapse_id"].duplicated().any():
        raise ValueError(f"{len(labelled)} labels for {n_expected} synapses off the cut node.")

    labels = pd.DataFrame({
        "synapse_id": labelled["synapse_id"].to_numpy(dtype=np.int64),
        "neuron": np.full(len(labelled), neuron_id, dtype=np.int64),
        "side": labelled["type"].to_numpy(dtype=object),
        "label": labelled["label"].to_numpy(dtype=object),
    })

    # different side counts among the tied nodes: above 1, another tied node would cut elsewhere (at 1, it can still
    # put different synapses on each side)
    n_tied_sides = len(tied[["side_1_pre", "side_1_post", "side_2_pre", "side_2_post"]].drop_duplicates())
    if METHOD == "maxsi":
        chosen = {
            "MaxSI_SI": float(best["SI"]),
            "n_tied_max_SI": len(tied),
            "n_tied_max_SI_distinct_sides": n_tied_sides,
            # for information only: nodes whose SI is within numpy isclose of the highest
            "n_close_max_SI": int(np.isclose(valid_node_SI["SI"].to_numpy(), best["SI"]).sum()),
        }
    else:
        chosen = {
            SI_COLUMN: float(best["SI"]),   # NaN under MinFisherP when the cell's synapses off the cut are of one kind
            VALUE_COLUMN: float(best[VALUE_COLUMN]),
            TIED_COLUMN: len(tied),
            TIED_SIDES_COLUMN: n_tied_sides,
        }

    summary = {
        "neuron": np.int64(neuron_id),
        "best_node_id": np.int64(best["node_id"]),
        **chosen,
        "n_candidates": len(node_SI),
        "n_attached_pre": len(cell.presynapses),
        "n_attached_post": len(cell.postsynapses),
        "n_labelled": len(labels),
        **sides,
    }
    return labels, summary


#%%6 Run the batches

n_batches = int(np.ceil(len(all_neurons) / BATCH_SIZE))

batch_dir.mkdir(parents=True, exist_ok=True)
# MinFisherP's p-values, and which of them underflow to 0, depend on the scipy version
if METHOD == "minfisherp":
    import scipy
    skeleton_record["scipy_version"] = scipy.__version__

# stops on another run's batches, including one made from another skeleton set or navis version
check_batch_record(batch_dir, all_neurons, BATCH_SIZE, extra=skeleton_record, setting_name=batch_dir_setting)

print("Method:", METHOD)
print("Batch size:", BATCH_SIZE)
print("Number of batches:", n_batches)

# explicit types, so that a batch where no cell labels still has string columns
LABEL_SCHEMA = pa.schema([("synapse_id", pa.int64()), ("neuron", pa.int64()),
                          ("side", pa.string()), ("label", pa.string())])
FAILURE_SCHEMA = pa.schema([("neuron", pa.int64()), ("error", pa.string())])


def write_whole(table, path):
    """Write to a temporary name, then rename, so that a run stopped mid-write leaves no file under the real name.
    The temporary name holds the process id, so two copies writing the same batch (the same bytes) never share it."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    feather.write_feather(table, tmp)
    tmp.replace(path)


for batch_number in range(n_batches):

    if batch_number % N_SHARDS != SHARD:
        continue   # another copy's batch

    batch_id = batch_number + 1
    start = batch_number * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(all_neurons))
    batch_ids = all_neurons.iloc[start:end]

    labels_file = batch_dir / f"labels_batch_{batch_id:04d}.ftr"
    summary_file = batch_dir / f"summary_batch_{batch_id:04d}.ftr"
    failures_file = batch_dir / f"failures_batch_{batch_id:04d}.ftr"

    if labels_file.exists() and summary_file.exists() and failures_file.exists():
        print(f"\nBatch {batch_id}/{n_batches} already exists. Skipping.")
        continue

    print(f"\nBatch {batch_id}/{n_batches}")
    print(f"Cells {start + 1}-{end} ({len(batch_ids)} cells)")

    # the synapses of this batch's cells, in the table's order
    batch_synapses = allsynapses.loc[allsynapses["pre"].isin(batch_ids)
                                     | allsynapses["post"].isin(batch_ids)]

    label_frames, summaries, failures = [], [], []

    for k, neuron_id in enumerate(batch_ids, start=1):
        neuron_id = int(neuron_id)
        print(f"{k}/{len(batch_ids)}")
        try:
            neuron_synapses = batch_synapses.loc[(batch_synapses["pre"] == neuron_id)
                                                 | (batch_synapses["post"] == neuron_id)].copy()
            labels, summary = maxsi_labels_one_cell(neuron_id, neuron_synapses)
            label_frames.append(labels)
            summaries.append(summary)
        except MemoryError:
            raise   # the machine, not the cell: stop, so the batch is rerun rather than saved with a gap
        except Exception as error:
            print(f"Failed neuron {neuron_id}: {type(error).__name__}: {error}")
            failures.append({"neuron": np.int64(neuron_id),
                             "error": f"{type(error).__name__}: {error}"})

    labels_df = (pd.concat(label_frames, ignore_index=True) if label_frames else
                 pd.DataFrame({"synapse_id": pd.Series([], dtype=np.int64),
                               "neuron": pd.Series([], dtype=np.int64),
                               "side": pd.Series([], dtype=object),
                               "label": pd.Series([], dtype=object)}))
    summary_df = pd.DataFrame(summaries)
    if len(summary_df.columns) == 0:
        summary_df = pd.DataFrame({"neuron": pd.Series([], dtype=np.int64)})
    failures_df = pd.DataFrame(failures, columns=["neuron", "error"]).astype({"neuron": np.int64})

    # the failures file last: a batch counts as done only when all three exist
    write_whole(pa.Table.from_pandas(labels_df, schema=LABEL_SCHEMA, preserve_index=False), labels_file)
    write_whole(pa.Table.from_pandas(summary_df, preserve_index=False), summary_file)
    write_whole(pa.Table.from_pandas(failures_df, schema=FAILURE_SCHEMA, preserve_index=False), failures_file)

    print(f"Saved batch {batch_id}/{n_batches}")
    print(f"Successful: {len(summary_df)} | Failed: {len(failures_df)}")

    del batch_synapses, label_frames, labels_df, summary_df, failures_df
    gc.collect()

if N_SHARDS > 1:
    raise SystemExit(f"Shard {SHARD} of {N_SHARDS}: its batches are done. Once every copy has finished, run with "
                     "N_SHARDS = 1 to check the batches and write the tables.")
print("\nFinished all batches.")

del allsynapses
gc.collect()


#%%7 Read the batches

# Hand the memory Arrow keeps after the batch loop back before the tables are built (section 6 has already freed
# the raw synapses; the del below only matters when the cells are run out of order).
if "allsynapses" in globals():
    del allsynapses
gc.collect()
pa.default_memory_pool().release_unused()

def read_label_codes(path):
    """A labels batch with side as 0 (pre) / 1 (post) and label as 0 (A) / 1 (D), both int8."""
    t = feather.read_table(path)
    if t.num_rows == 0:   # also a batch with no labels written before the columns had explicit types
        return pd.DataFrame({"synapse_id": pd.Series([], dtype=np.int64), "neuron": pd.Series([], dtype=np.int64),
                             "side": pd.Series([], dtype=np.int8), "label": pd.Series([], dtype=np.int8)})
    for name, allowed in (("side", ["pre", "post"]), ("label", ["A", "D"])):
        column = t.column(name)
        if column.null_count or (pc.sum(pc.invert(pc.is_in(column, value_set=pa.array(allowed)))).as_py() or 0):
            raise SystemExit(f"{path.name}: {name} holds a value other than {allowed}")
    return pd.DataFrame({
        "synapse_id": t.column("synapse_id").to_numpy(),
        "neuron": t.column("neuron").to_numpy(),
        "side": pc.equal(t.column("side"), "post").to_numpy().astype(np.int8),
        "label": pc.equal(t.column("label"), "D").to_numpy().astype(np.int8),
    })


summary_frames, failure_frames, label_frames, missing = [], [], [], []
for batch_id in range(1, n_batches + 1):
    files = [batch_dir / f"{kind}_batch_{batch_id:04d}.ftr" for kind in ("labels", "summary", "failures")]
    if not all(f.exists() for f in files):
        missing.append(batch_id)
        continue
    label_frames.append(read_label_codes(files[0]))
    summary_frames.append(pd.read_feather(files[1]))
    failure_frames.append(pd.read_feather(files[2]))
if missing:
    raise SystemExit(f"batch files missing, so their cells would be left out: batches {missing}")

maxsi_failures = pd.concat(failure_frames, ignore_index=True).astype({"neuron": np.int64})
labelled_frames = [f for f in summary_frames if len(f)]
print(f"Cells with labels: {sum(len(f) for f in labelled_frames):,}; failed: {len(maxsi_failures):,}")
if len(maxsi_failures):
    print(maxsi_failures["error"].str.split(":").str[:2].str.join(":").value_counts().to_string())
if not labelled_frames:
    raise SystemExit("No cell was labelled, so there is nothing to write. No table written.")
maxsi_summary = pd.concat(labelled_frames, ignore_index=True)
labels = pd.concat(label_frames, ignore_index=True)
del label_frames, summary_frames, failure_frames, labelled_frames

for frame in (maxsi_summary, maxsi_failures, labels):
    if len(frame):
        assert frame["neuron"].dtype == np.int64, frame["neuron"].dtype
assert labels["synapse_id"].dtype == np.int64

# every cell of the run exactly once, either labelled or failed
done = pd.concat([maxsi_summary["neuron"], maxsi_failures["neuron"]], ignore_index=True)
both = set(maxsi_summary["neuron"]) & set(maxsi_failures["neuron"])
if both:
    raise SystemExit(f"{len(both):,} cells are both labelled and failed, for example {sorted(both)[:5]}")
if done.duplicated().any():
    raise SystemExit(f"{int(done.duplicated().sum()):,} cells appear in two batches")
if set(done) != set(all_neurons):
    raise SystemExit(f"the batches hold {len(set(done) - set(all_neurons)):,} cells not in this run and miss "
                     f"{len(set(all_neurons) - set(done)):,} of its cells")
# One label per synapse end. The same test as labels.duplicated(subset=["synapse_id", "side"]).any(), on one
# integer per end (synapse_id * 2 + side, one to one since side is 0 or 1) sorted, which needs far less memory than
# duplicated() on the roughly 154 million ends of a full run.
if len(labels):
    assert -2**62 < labels["synapse_id"].min() and labels["synapse_id"].max() < 2**62, "synapse_id out of range"
end_keys = labels["synapse_id"].to_numpy() * 2 + labels["side"].to_numpy().astype(np.int64)
end_keys.sort()
two_labels = bool((end_keys[1:] == end_keys[:-1]).any())
del end_keys
if two_labels:
    raise SystemExit("a synapse end has two labels")

ties = maxsi_summary[TIED_SIDES_COLUMN]
print(f"Cells whose chosen value ({METHOD_NAME}) is shared by nodes that cut elsewhere (the first node is taken): "
      f"{int((ties > 1).sum()):,}")
print(f"Cells whose chosen value is shared by more than one node (another one can put different synapses on each "
      f"side, even with the same counts): {int((maxsi_summary[TIED_COLUMN] > 1).sum()):,}")
if METHOD == "maxsi":
    print(f"Cells with more nodes within numpy isclose of the highest SI than exactly at it: "
          f"{int((maxsi_summary['n_close_max_SI'] > maxsi_summary['n_tied_max_SI']).sum()):,}")
print(f"Labelled cells with an empty {SI_COLUMN} (all their synapses off the cut node of one kind; kept): "
      f"{int(maxsi_summary[SI_COLUMN].isna().sum()):,}")
print(f"Labelled cells' synapse ends on the cut node, so unlabelled: "
      f"{int((maxsi_summary['node_pre_excluded'] + maxsi_summary['node_post_excluded']).sum()):,}")

# what may be written
n_no_skeleton = int(maxsi_failures["error"].str.contains(NO_SKELETON, regex=False).sum())
if n_no_skeleton and not ACCEPT_CELLS_WITHOUT_SKELETON:
    raise SystemExit(f"{n_no_skeleton:,} cells failed for want of a usable skeleton under {SWC_DIR} (see the "
                     "failures files). Complete the skeletons and run again with an empty batch folder, or set "
                     "ACCEPT_CELLS_WITHOUT_SKELETON = True to count them with the failed cells. No table written.")
accepted = 0 if ACCEPT_FAILED_CELLS is None else ACCEPT_FAILED_CELLS
if len(maxsi_failures) != accepted:
    raise SystemExit(f"{len(maxsi_failures):,} cells failed, and ACCEPT_FAILED_CELLS is {ACCEPT_FAILED_CELLS}. "
                     f"Read the errors above; to write the tables without these cells, set "
                     f"ACCEPT_FAILED_CELLS = {len(maxsi_failures)}. No table written.")

write_tables = not check_final_tables()   # stops if only some of the three exist


#%%8 Synapse table of the method

if write_tables:
    if SUBSET_RUN:
        _ids = all_neurons.tolist()
        syn_filter = ds.field("pre").isin(_ids) | ds.field("post").isin(_ids)
    else:
        syn_filter = None
    # Only the three key columns are read for the label merges; the whole table is read further down one record
    # batch at a time, so that it is never in memory next to its filtered copy.
    syn_dataset = ds.dataset(str(SYNAPSE_TABLE_SFC_FTR), format="feather")
    key_table = syn_dataset.to_table(columns=["synapse_id", "pre", "post"], filter=syn_filter)
    keys = pd.DataFrame({"synapse_id": key_table["synapse_id"].to_numpy(),
                         "pre": key_table["pre"].to_numpy(),
                         "post": key_table["post"].to_numpy()})
    del key_table
    assert all(keys[c].dtype == np.int64 for c in keys), keys.dtypes
    assert not keys["synapse_id"].duplicated().any(), "synapse_id is not unique in the SFC synapse table"
    assert not (keys["pre"] == keys["post"]).any(), "the SFC synapse table holds self-synapses"

    def label_of_end(side, cells):
        """Each row's label at one end (0 = A, 1 = D, NaN if none): the label whose synapse_id is the row's and
        whose cell is the row's cell at that end. What a left merge on (synapse_id, cell) gives, found by binary
        search on the sorted labels instead (each synapse end has at most one label, checked in section 7)."""
        on_side = labels["side"].to_numpy() == side
        label_ids = labels["synapse_id"].to_numpy()[on_side]
        order = np.argsort(label_ids, kind="stable")
        label_ids = label_ids[order]
        label_cells = labels["neuron"].to_numpy()[on_side][order]
        label_values = labels["label"].to_numpy()[on_side][order]
        del on_side, order
        assert not (label_ids[1:] == label_ids[:-1]).any(), "a synapse end has two labels"
        row_ids = keys["synapse_id"].to_numpy()
        at = np.minimum(np.searchsorted(label_ids, row_ids), max(len(label_ids) - 1, 0))
        found = (np.zeros(len(row_ids), dtype=bool) if len(label_ids) == 0
                 else (label_ids[at] == row_ids) & (label_cells[at] == cells))
        values = np.full(len(row_ids), np.nan)
        values[found] = label_values[at[found]]
        return values, len(label_ids)

    keys["pre_label"], n_pre_labels = label_of_end(0, keys["pre"].to_numpy())
    keys["post_label"], n_post_labels = label_of_end(1, keys["post"].to_numpy())
    del labels
    gc.collect()
    pa.default_memory_pool().release_unused()

    no_pre, no_post = keys["pre_label"].isna(), keys["post_label"].isna()
    keep = ~(no_pre | no_post)
    print(f"Rows of {SYNAPSE_TABLE_SFC_FTR.name} read: {len(keys):,}; left out for an unlabelled end: "
          f"{int((~keep).sum()):,} (pre end only {int((no_pre & ~no_post).sum()):,}, post end only "
          f"{int((~no_pre & no_post).sum()):,}, both {int((no_pre & no_post).sum()):,}); kept {int(keep.sum()):,}")
    print(f"Labelled ends lost because the other end of the synapse has no label: "
          f"{int((no_pre & ~no_post).sum() + (~no_pre & no_post).sum()):,}")
    print(f"Labelled ends with no row in {SYNAPSE_TABLE_SFC_FTR.name}: pre {n_pre_labels - int((~no_pre).sum()):,}, "
          f"post {n_post_labels - int((~no_post).sum()):,}")

    si_by_cell = maxsi_summary.set_index("neuron")[SI_COLUMN]
    kept = keys.loc[keep]
    # pre label * 2 + post label indexes AA, AD, DA, DD
    comp_index = (kept["pre_label"].to_numpy(dtype=np.int8) * 2 + kept["post_label"].to_numpy(dtype=np.int8))
    comp = pc.take(pa.array(["AA", "AD", "DA", "DD"], type=pa.string()), pa.array(comp_index))
    si_pre = kept["pre"].map(si_by_cell).to_numpy(dtype=np.float64)
    si_post = kept["post"].map(si_by_cell).to_numpy(dtype=np.float64)
    del kept, comp_index
    gc.collect()

    # The table again, one record batch at a time, in the order of the key read above (checked batch by batch on
    # synapse_id): each batch keeps its rows with both ends labelled and gets the new comp, SI_pre and SI_post.
    # The batches keep the boundaries a whole-table read and filter would give them, and empty ones are left out
    # as that write leaves them out, so the file written is the same (checked byte for byte on the full run and on
    # small subset tables).
    keep_mask = keep.to_numpy()
    synapse_ids = keys["synapse_id"].to_numpy()
    key_pre, key_post = keys["pre"].to_numpy(), keys["post"].to_numpy()
    new_columns = {"comp": comp,
                   "SI_pre": pa.array(si_pre, type=pa.float64()),
                   "SI_post": pa.array(si_post, type=pa.float64())}
    out_batches, row, kept_row = [], 0, 0
    for batch in syn_dataset.to_batches(filter=syn_filter):
        n = batch.num_rows
        if not (np.array_equal(batch.column("synapse_id").to_numpy(), synapse_ids[row:row + n])
                and np.array_equal(batch.column("pre").to_numpy(), key_pre[row:row + n])
                and np.array_equal(batch.column("post").to_numpy(), key_post[row:row + n])):
            raise SystemExit(f"{SYNAPSE_TABLE_SFC_FTR.name}: a record batch is not in the order of the key read "
                             f"(rows {row:,} on). No table written.")
        part = batch.filter(pa.array(keep_mask[row:row + n]))
        k = part.num_rows
        for name, values in new_columns.items():
            i = part.schema.get_field_index(name)
            part = part.set_column(i, part.schema.field(name), values.slice(kept_row, k))
        if k:   # an empty batch is left out, as a whole-table filter and write leave it out
            out_batches.append(part)
        row += n
        kept_row += k
    if row != len(keys) or kept_row != int(keep.sum()):
        raise SystemExit(f"read {row:,} rows of {SYNAPSE_TABLE_SFC_FTR.name} batch by batch against {len(keys):,} "
                         f"in the key read, and kept {kept_row:,} against {int(keep.sum()):,}. No table written.")
    out = pa.Table.from_batches(out_batches, schema=syn_dataset.schema)
    del out_batches, keep_mask, synapse_ids, key_pre, key_post, new_columns
    assert out.schema.equals(syn_dataset.schema, check_metadata=False), (out.schema, syn_dataset.schema)
    out = out.replace_schema_metadata(None)   # the pandas metadata holds the old row count of the index

    synapse_table_out.parent.mkdir(parents=True, exist_ok=True)
    write_whole(out, synapse_table_out)
    print(f"Wrote {synapse_table_out} ({out.num_rows:,} rows)")
    del syn_dataset, out, keys, comp, si_pre, si_post

if "labels" in globals():
    del labels
gc.collect()


#%%9 Neuron table of the method (step 05's compartment counts on its synapse table)

if write_tables:
    syn = feather.read_table(synapse_table_out, columns=["pre", "post", "comp"])
    comp_codes = syn.column("comp").combine_chunks().dictionary_encode()
    comp_values = comp_codes.dictionary.to_pylist()
    comp_index = comp_codes.indices.to_numpy()
    maxsi_syn = pd.DataFrame({"pre": syn.column("pre").to_numpy(), "post": syn.column("post").to_numpy()})
    del syn, comp_codes
    gc.collect()
    assert maxsi_syn["pre"].dtype == np.int64 and maxsi_syn["post"].dtype == np.int64

    # every other column, the cell properties among them, stays as in the SFC table
    maxsi_neurons = neuron_table.copy()

    # 05:223-248, pre_A ... post_L; the first letter of comp is the pre end's, the second the post end's
    for component in ["A", "D", "L"]:
        pre_mask = np.isin(comp_index, [i for i, v in enumerate(comp_values) if v[0] == component])
        post_mask = np.isin(comp_index, [i for i, v in enumerate(comp_values) if v[1] == component])
        pre_counts = maxsi_syn.loc[pre_mask].groupby("pre").size()
        post_counts = maxsi_syn.loc[post_mask].groupby("post").size()
        maxsi_neurons[f"pre_{component}"] = (maxsi_neurons["neuron"].map(pre_counts)
                                             .fillna(0).astype(np.int64))
        maxsi_neurons[f"post_{component}"] = (maxsi_neurons["neuron"].map(post_counts)
                                              .fillna(0).astype(np.int64))
    del comp_index

    # 05:254-256
    maxsi_neurons["axon_correct"] = maxsi_neurons["pre_A"] / (maxsi_neurons["pre_A"] + maxsi_neurons["post_A"])
    maxsi_neurons["dend_correct"] = maxsi_neurons["post_D"] / (maxsi_neurons["pre_D"] + maxsi_neurons["post_D"])
    maxsi_neurons["SI"] = maxsi_neurons["neuron"].map(maxsi_summary.set_index("neuron")[SI_COLUMN]).astype(np.float64)
    maxsi_neurons["linker"] = np.float64(0.0)   # the split has no linker compartment

    in_synapse_table = maxsi_neurons["neuron"].isin(pd.unique(maxsi_syn[["pre", "post"]].values.ravel()))
    del maxsi_syn
    gc.collect()

    assert list(maxsi_neurons.columns) == list(neuron_table.columns)
    assert (maxsi_neurons.dtypes == neuron_table.dtypes).all(), maxsi_neurons.dtypes[maxsi_neurons.dtypes != neuron_table.dtypes]

    # as step 05: only the cells with an SI (05:129) and a row in the synapse table (05:223, 298). A cell has an SI
    # here when it was labelled, even if the SI is empty (a MinFisherP cut with the synapses off it of one kind).
    in_run = maxsi_neurons["neuron"].isin(all_neurons)
    failed = maxsi_neurons["neuron"].isin(maxsi_failures["neuron"])
    has_si = maxsi_neurons["neuron"].isin(maxsi_summary["neuron"])
    keep = has_si & in_synapse_table
    print(f"Cells in {NEURON_TABLE_SFC_FTR.name}: {len(maxsi_neurons):,}; dropped because the {METHOD_NAME} split "
          f"failed: {int(failed.sum()):,}"
          + (f"; dropped because not in this subset run: {int((~in_run).sum()):,}" if SUBSET_RUN else "")
          + f"; dropped with a {SI_COLUMN} but no row in the {METHOD_NAME} synapse table: "
          f"{int((has_si & ~in_synapse_table).sum()):,}; kept {int(keep.sum()):,}")
    assert int((~has_si).sum()) == int(failed.sum()) + int((~in_run).sum())

    # the figures drop cells with an empty axon_correct or dend_correct
    sfc_kept = neuron_table.loc[keep]
    new_axon = maxsi_neurons.loc[keep, "axon_correct"].isna() & sfc_kept["axon_correct"].notna()
    new_dend = maxsi_neurons.loc[keep, "dend_correct"].isna() & sfc_kept["dend_correct"].notna()
    print(f"Kept cells whose axon_correct or dend_correct is empty here but not in the SFC table: "
          f"{int((new_axon | new_dend).sum()):,} (axon_correct {int(new_axon.sum()):,}, "
          f"dend_correct {int(new_dend.sum()):,})")
    maxsi_neurons = maxsi_neurons.loc[keep]

    neuron_table_out.parent.mkdir(parents=True, exist_ok=True)
    write_whole(pa.Table.from_pandas(maxsi_neurons.reset_index(drop=True), preserve_index=False), neuron_table_out)
    print(f"Wrote {neuron_table_out} ({len(maxsi_neurons):,} rows)")


#%%10 SI table of the method (the format of SI_updated.ftr)

if write_tables:
    si_updated = pd.DataFrame({"root_id": maxsi_summary["neuron"].to_numpy(dtype=np.int64),
                               "SI": maxsi_summary[SI_COLUMN].to_numpy(dtype=np.float64)})
    si_updated_out.parent.mkdir(parents=True, exist_ok=True)
    write_whole(pa.Table.from_pandas(si_updated, preserve_index=False), si_updated_out)   # last: its name marks a finished run
    print(f"Wrote {si_updated_out} ({len(si_updated):,} rows)")
