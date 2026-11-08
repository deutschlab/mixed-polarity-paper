# -*- coding: utf-8 -*-
"""
The Buhmann synapses on the MaxSI split, for the panels that compare the two synapse detectors (Supp 2-S1 B-C,
Supp 3-S2 A-C).

For every cell step 09 cut under "maxsi" (its label batches in MAXSI_LABEL_BATCH_DIR) that is also in the published
Buhmann neuron table: load the skeleton, heal it (navis.heal_skeleton), and attach the cell's Buhmann synapses
(BUHMANN_SYNAPSE_SFC_FTR, self-synapses left out) with methods_all.attach_synapses_princeton, as step 09 attaches the
Princeton ones (nearest node, 10,000 nm cutoff). The cut is step 09's, not a new one: side 1 is the cut node's one
child and its descendants, side 2 the rest of the cell without the cut node, and the axon is the side step 09 called
the axon (axon_side). Each synapse end on a side gets that side's label (A or D); ends on the cut node get none, as in
step 09. The cell's Buhmann SI is the SI of the four side counts (methods/node_cut_si.safe_SI_calc).

Every CHECK_EVERY-th cell, and every cell of a subset run, also gets its Princeton synapses (step 09's raw table)
attached and put on the same two sides: the four counts must equal the ones step 09 saved, which shows the cut is
applied as step 09 applied it. A cell whose counts differ fails.

Writes per batch of BATCH_SIZE cells, to BUHMANN_MAXSI_LABEL_BATCH_DIR:
- labels_batch_NNNN.ftr: synapse_id (the Buhmann table's id), neuron, side ('pre' or 'post'), label ('A' or 'D');
- summary_batch_NNNN.ftr: one row per cell: its Buhmann side counts and the side of the axon, the synapses on the cut
  node, SI_buhmann, and for checked cells whether the Princeton counts matched;
- failures_batch_NNNN.ftr: neuron and error.
A batch is skipped if its three files exist; the failures file is written last.

Then, with every batch done, the two tables in the format of the published Buhmann ones, under the names the panel
scripts read under SPLIT_METHOD = "maxsi" (NEURON_TABLE_NONP_FTR and SYNAPSE_TABLE_NONP_FTR):
- the synapse table: the rows of BUHMANN_SYNAPSE_SFC_FTR with both ends labelled, comp from the two labels (pre end
  first), SI_pre and SI_post the two cells' SI_buhmann; every other column as published;
- the neuron table: the rows of BUHMANN_NEURON_SFC_FTR with an SI_buhmann and a row in the new synapse table;
  pre_A to post_L, axon_correct and dend_correct counted on the new synapse table as step 05 counts them, SI the
  SI_buhmann, linker 0 (the split has no linker); every other column as published. Those other columns (synapse and
  partner counts, degrees, flow) are the published table's, counted on the SFC rows; no panel that reads this table
  uses them.

Settings in the environment: BUHMANN_N_SHARDS and BUHMANN_SHARD run N copies at once, each on every N-th batch
(then a run with N_SHARDS = 1 writes the tables); BUHMANN_SUBSET_CELLS, a CSV with a column "neuron", runs only those
cells and writes everything to BUHMANN_MAXSI_SUBSET_DIR instead.

Needs config.py on PIPELINE_RUN = "rebuild" and SPLIT_METHOD = "maxsi".
"""

#%%1 Imports and settings
import gc
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (METHODS_DIR, PIPELINE_RUN, SPLIT_METHOD, MAXSI_LABEL_BATCH_DIR, SYNAPSE_TABLE_RAW_FTR, SWC_DIR,
                    PUBLISHED_DERIVED_DIR, BUHMANN_SYNAPSE_SFC_FTR, BUHMANN_NEURON_SFC_FTR,
                    BUHMANN_MAXSI_LABEL_BATCH_DIR, BUHMANN_MAXSI_SUBSET_DIR, NEURON_TABLE_NONP_FTR,
                    SYNAPSE_TABLE_NONP_FTR)
sys.path.insert(0, str(METHODS_DIR))

import hashlib
import json
import navis
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.feather as feather
from methods_all import upload_swc, attach_synapses_princeton
from node_cut_si import get_descendants_from_node, safe_SI_calc

if PIPELINE_RUN != "rebuild" or SPLIT_METHOD != "maxsi":
    raise SystemExit('Set PIPELINE_RUN = "rebuild" and SPLIT_METHOD = "maxsi" in config.py.')

BATCH_SIZE = 2000
CHECK_EVERY = 20
N_SHARDS = int(os.environ.get("BUHMANN_N_SHARDS", 1))
SHARD = int(os.environ.get("BUHMANN_SHARD", 0))
if not (N_SHARDS >= 1 and 0 <= SHARD < N_SHARDS):
    raise SystemExit(f"SHARD must be 0 to N_SHARDS - 1 (N_SHARDS = {N_SHARDS}, SHARD = {SHARD})")
SUBSET_CELLS = os.environ.get("BUHMANN_SUBSET_CELLS")
# The tables are written only if exactly this many cells failed (None: only if none failed). Run once, read the count
# and the errors the run prints, then set it (BUHMANN_ACCEPT_FAILED_CELLS in the environment).
ACCEPT_FAILED_CELLS = int(os.environ["BUHMANN_ACCEPT_FAILED_CELLS"]) if os.environ.get("BUHMANN_ACCEPT_FAILED_CELLS") else None
STOP_AFTER_FAILURES_IN_A_ROW = 50   # that many cells failing one after another means a fault in the code or the data

if SUBSET_CELLS:
    batch_dir = BUHMANN_MAXSI_SUBSET_DIR / "batches"
    synapse_table_out = BUHMANN_MAXSI_SUBSET_DIR / SYNAPSE_TABLE_NONP_FTR.name
    neuron_table_out = BUHMANN_MAXSI_SUBSET_DIR / NEURON_TABLE_NONP_FTR.name
else:
    batch_dir = BUHMANN_MAXSI_LABEL_BATCH_DIR
    synapse_table_out = SYNAPSE_TABLE_NONP_FTR
    neuron_table_out = NEURON_TABLE_NONP_FTR
if synapse_table_out.exists() or neuron_table_out.exists():
    raise SystemExit(f"{synapse_table_out.name} or {neuron_table_out.name} already exists in "
                     f"{synapse_table_out.parent}; move it away to build new ones.")


#%%2 The cells and step 09's cuts
summaries = sorted(MAXSI_LABEL_BATCH_DIR.glob("summary_batch_*.ftr"))
if not summaries:
    raise SystemExit(f"No step 09 summaries in {MAXSI_LABEL_BATCH_DIR}; run step 09 under \"maxsi\" first.")
cuts = pd.concat([pd.read_feather(f, columns=["neuron", "best_node_id", "axon_side", "side_1_pre", "side_1_post",
                                               "side_2_pre", "side_2_post"]) for f in summaries], ignore_index=True)
assert cuts["neuron"].dtype == np.int64 and cuts["neuron"].is_unique
buhmann_cells = pd.read_feather(BUHMANN_NEURON_SFC_FTR, columns=["neuron"])["neuron"].astype(np.int64)
cuts = cuts[cuts["neuron"].isin(buhmann_cells)].sort_values("neuron").reset_index(drop=True)
print(f"step 09 cuts: {len(summaries)} batch files; cells also in {BUHMANN_NEURON_SFC_FTR.name}: {len(cuts):,}")
if SUBSET_CELLS:
    wanted = pd.read_csv(SUBSET_CELLS, usecols=["neuron"])["neuron"].astype(np.int64)
    cuts = cuts[cuts["neuron"].isin(wanted)].reset_index(drop=True)
    print(f"subset run: {len(cuts)} of the {len(wanted)} listed cells")
all_neurons = cuts["neuron"]
n_batches = int(np.ceil(len(all_neurons) / BATCH_SIZE))
my_batches = [b for b in range(n_batches) if b % N_SHARDS == SHARD]

# the skeletons and navis version must be the ones step 09 cut with (its batch_run.json), and each skeleton file name
# must be in one subfolder only: upload_swc reads whichever subfolder it meets first, in no fixed order
step09_record = json.loads((MAXSI_LABEL_BATCH_DIR / "batch_run.json").read_text())
swc_files = sorted(p.relative_to(SWC_DIR).as_posix() for p in SWC_DIR.glob("*/*.swc"))
swc_names = pd.Series([f.split("/")[-1] for f in swc_files], dtype=object)
repeated = sorted(set(swc_names[swc_names.duplicated(keep=False)]))
if repeated:
    raise SystemExit(f"{len(repeated):,} skeleton file names appear in more than one subfolder of {SWC_DIR}, "
                     f"for example {repeated[:5]}. Keep one copy of each.")
swc_sha = hashlib.sha256("\n".join(f"{f}\t{(SWC_DIR / f).stat().st_size}" for f in swc_files).encode()).hexdigest()
if swc_sha != step09_record["swc_files_sha256"] or navis.__version__ != step09_record["navis_version"]:
    raise SystemExit(f"skeletons or navis differ from step 09's run (skeleton files {len(swc_files):,}, navis "
                     f"{navis.__version__}; step 09: {step09_record['swc_files']:,}, {step09_record['navis_version']})")

run_record = {"cells": len(all_neurons),
              "cells_sha256": hashlib.sha256(all_neurons.to_numpy(dtype=np.int64).tobytes()).hexdigest(),
              "batch_size": BATCH_SIZE, "check_every": CHECK_EVERY,
              "buhmann_neuron_table_bytes": BUHMANN_NEURON_SFC_FTR.stat().st_size,
              "buhmann_synapse_table": BUHMANN_SYNAPSE_SFC_FTR.name,
              "buhmann_synapse_table_bytes": BUHMANN_SYNAPSE_SFC_FTR.stat().st_size,
              "step09_batch_run": step09_record,
              "navis_version": navis.__version__}
batch_dir.mkdir(parents=True, exist_ok=True)
record_file = batch_dir / "batch_run.json"
if not record_file.exists():
    # written under a temporary name and renamed, so a copy started at the same moment never reads half a file
    tmp_record = record_file.with_name(f"{record_file.name}.{os.getpid()}.tmp")
    tmp_record.write_text(json.dumps(run_record, indent=1))
    tmp_record.replace(record_file)
if json.loads(record_file.read_text()) != run_record:
    raise SystemExit(f"{record_file} is from another run (other cells, inputs or settings); move the folder away.")

# the batches of this copy still to do; only their cells' synapses are read
todo = [b for b in my_batches if not all((batch_dir / f"{k}_batch_{b + 1:04d}.ftr").exists()
                                          for k in ("labels", "summary", "failures"))]
my_cells = pd.concat([all_neurons.iloc[b * BATCH_SIZE:(b + 1) * BATCH_SIZE] for b in todo]) \
    if todo else all_neurons.iloc[0:0]


#%%3 The synapses of this copy's cells
SYN_COLUMNS = ["id", "pre_x", "pre_y", "pre_z", "post_x", "post_y", "post_z", "pre", "post"]
cell_filter = ds.field("pre").isin(my_cells.tolist()) | ds.field("post").isin(my_cells.tolist())
if todo:
    buhmann = (ds.dataset(str(BUHMANN_SYNAPSE_SFC_FTR), format="feather")
               .to_table(columns=SYN_COLUMNS, filter=cell_filter).to_pandas()
               .rename(columns={"id": "synapse_id"}))
    assert buhmann["pre"].dtype == np.int64 and buhmann["post"].dtype == np.int64
    n_self = int((buhmann["pre"] == buhmann["post"]).sum())
    buhmann = buhmann[buhmann["pre"] != buhmann["post"]].reset_index(drop=True)
    buhmann["size"] = np.nan   # attach_synapses_princeton carries a size column; the Buhmann table has none
    assert buhmann["synapse_id"].is_unique, "the Buhmann table's id is not unique"
    print(f"Buhmann synapses of this copy's cells: {len(buhmann):,} ({n_self:,} self-synapses left out)")
    # each cell's rows by binary search on sorted pre and post, instead of a mask over the table per cell
    pre_order = np.argsort(buhmann["pre"].to_numpy(), kind="stable")
    post_order = np.argsort(buhmann["post"].to_numpy(), kind="stable")
    pre_sorted = buhmann["pre"].to_numpy()[pre_order]
    post_sorted = buhmann["post"].to_numpy()[post_order]

    checked = all_neurons.iloc[::CHECK_EVERY] if not SUBSET_CELLS else all_neurons
    checked = checked[checked.isin(my_cells)]
    raw_path = PUBLISHED_DERIVED_DIR / run_record["step09_batch_run"]["synapse_table"]
    if not raw_path.exists():
        raw_path = Path(SYNAPSE_TABLE_RAW_FTR).parent / run_record["step09_batch_run"]["synapse_table"]
    princeton = (ds.dataset(str(raw_path), format="feather")
                 .to_table(columns=["synapse_id", "pre_x", "pre_y", "pre_z", "post_x", "post_y", "post_z", "size",
                                    "pre", "post"],
                           filter=ds.field("pre").isin(checked.tolist()) | ds.field("post").isin(checked.tolist()))
                 .to_pandas())
    checked = set(checked.tolist())


def rows_of(neuron_id):
    """The Buhmann rows where the cell is pre or post (a self-synapse is already gone, so no row is taken twice)."""
    a = pre_order[np.searchsorted(pre_sorted, neuron_id, "left"):np.searchsorted(pre_sorted, neuron_id, "right")]
    b = post_order[np.searchsorted(post_sorted, neuron_id, "left"):np.searchsorted(post_sorted, neuron_id, "right")]
    return buhmann.iloc[np.sort(np.concatenate([a, b]))]


#%%4 Label one cell
def side_counts(cell, side_1, side_2, cut_node_id):
    conn = cell.connectors
    kind = conn["type"].astype(str)
    on_1, on_2, on_cut = conn["node_id"].isin(side_1), conn["node_id"].isin(side_2), conn["node_id"] == cut_node_id
    if (~(on_1 | on_2 | on_cut)).any():
        raise SystemExit(f"{int((~(on_1 | on_2 | on_cut)).sum())} synapses are on neither side nor the cut node: "
                         "the cut is not applied as step 09 applies it. Stopped.")
    counts = {"side_1_pre": int((on_1 & kind.eq("pre")).sum()), "side_1_post": int((on_1 & kind.eq("post")).sum()),
              "side_2_pre": int((on_2 & kind.eq("pre")).sum()), "side_2_post": int((on_2 & kind.eq("post")).sum()),
              "on_cut_node": int(on_cut.sum())}
    return counts, on_1, on_2


def label_one_cell(cut):
    neuron_id = np.int64(cut.neuron)
    cut_node_id = np.int64(cut.best_node_id)
    swc = upload_swc(neuron_id)
    if swc is None:
        raise OSError(f"no usable skeleton for {neuron_id}")
    healed = navis.heal_skeleton(swc)
    nodes = healed.nodes
    children = nodes.loc[nodes["parent_id"] == cut_node_id, "node_id"].tolist()
    if len(children) != 1:
        raise ValueError(f"cut node {cut_node_id} has {len(children)} children; expected 1")
    side_1 = get_descendants_from_node(healed, children[0])
    side_2 = set(nodes["node_id"]) - side_1 - {cut_node_id}

    princeton_match = None
    if neuron_id in checked:
        p_cell = attach_synapses_princeton(healed.copy(), princeton[(princeton["pre"] == neuron_id)
                                                                     | (princeton["post"] == neuron_id)])
        p_cell = p_cell[0] if isinstance(p_cell, navis.NeuronList) else p_cell
        p_counts, _, _ = side_counts(p_cell, side_1, side_2, cut_node_id)
        princeton_match = all(p_counts[k] == int(getattr(cut, k))
                              for k in ("side_1_pre", "side_1_post", "side_2_pre", "side_2_post"))
        if not princeton_match:
            raise SystemExit(f"cell {neuron_id}: Princeton side counts {p_counts} differ from step 09's, so the cut "
                             "is not applied as step 09 applies it. Stopped.")

    synapses = rows_of(neuron_id)
    cell = attach_synapses_princeton(healed, synapses)
    cell = cell[0] if isinstance(cell, navis.NeuronList) else cell
    counts, on_1, on_2 = side_counts(cell, side_1, side_2, cut_node_id)
    axon_is_1 = int(cut.axon_side) == 1
    conn = cell.connectors
    on_side = on_1 | on_2
    labels = pd.DataFrame({
        "synapse_id": conn.loc[on_side, "synapse_id"].to_numpy(dtype=np.int64),
        "neuron": np.full(int(on_side.sum()), neuron_id, dtype=np.int64),
        "side": conn.loc[on_side, "type"].astype(str).to_numpy(dtype=object),
        "label": np.where(on_1[on_side].to_numpy() == axon_is_1, "A", "D").astype(object),
    })
    if labels[["synapse_id", "side"]].duplicated().any():
        raise ValueError("a synapse end got two labels")
    if axon_is_1:
        ax, de = (counts["side_1_pre"], counts["side_1_post"]), (counts["side_2_pre"], counts["side_2_post"])
    else:
        ax, de = (counts["side_2_pre"], counts["side_2_post"]), (counts["side_1_pre"], counts["side_1_post"])
    si = safe_SI_calc(neuron_id, ax[0], ax[1], de[0], de[1])[0]
    summary = {"neuron": neuron_id, "best_node_id": cut_node_id, "axon_side": int(cut.axon_side),
               "n_buhmann": len(synapses), "n_attached": len(conn), "n_labelled": len(labels), **counts,
               "SI_buhmann": float(si), "princeton_checked": princeton_match is not None,
               "princeton_match": bool(princeton_match) if princeton_match is not None else False}
    return labels, summary


#%%5 Run the batches
LABEL_SCHEMA = pa.schema([("synapse_id", pa.int64()), ("neuron", pa.int64()),
                          ("side", pa.string()), ("label", pa.string())])
FAILURE_SCHEMA = pa.schema([("neuron", pa.int64()), ("error", pa.string())])


def write_whole(table, path):
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    feather.write_feather(table, tmp)
    tmp.replace(path)


for b in todo:
    part = cuts.iloc[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
    frames, rows, failures, in_a_row = [], [], [], 0
    for cut in part.itertuples(index=False):
        try:
            labels, summary = label_one_cell(cut)
            frames.append(labels)
            rows.append(summary)
            in_a_row = 0
        except MemoryError:
            raise
        except Exception as error:   # recorded per cell; the tables are written only for an accepted failure count
            failures.append({"neuron": np.int64(cut.neuron), "error": f"{type(error).__name__}: {error}"})
            in_a_row += 1
            if in_a_row >= STOP_AFTER_FAILURES_IN_A_ROW:
                raise SystemExit(f"{in_a_row} cells in a row failed (last: {failures[-1]['error']}); stopped, as "
                                 "that points to a fault in the code or the data. This batch is not saved.")
    labels_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        {"synapse_id": pd.Series([], dtype=np.int64), "neuron": pd.Series([], dtype=np.int64),
         "side": pd.Series([], dtype=object), "label": pd.Series([], dtype=object)})
    stem = f"{b + 1:04d}"
    write_whole(pa.Table.from_pandas(labels_df, schema=LABEL_SCHEMA, preserve_index=False),
                batch_dir / f"labels_batch_{stem}.ftr")
    write_whole(pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False), batch_dir / f"summary_batch_{stem}.ftr")
    write_whole(pa.Table.from_pandas(pd.DataFrame(failures, columns=["neuron", "error"]).astype({"neuron": np.int64}),
                                     schema=FAILURE_SCHEMA, preserve_index=False), batch_dir / f"failures_batch_{stem}.ftr")
    print(f"batch {b + 1}/{n_batches}: {len(rows)} cells labelled, {len(failures)} failed", flush=True)
    del frames, labels_df
    gc.collect()

if todo:
    del buhmann, pre_order, post_order, pre_sorted, post_sorted, princeton
    gc.collect()

if N_SHARDS > 1:
    raise SystemExit(f"Shard {SHARD} of {N_SHARDS} done. When every copy has finished, run with BUHMANN_N_SHARDS = 1 "
                     "to write the tables.")


#%%6 Read the batches
missing = [b + 1 for b in range(n_batches) if not (batch_dir / f"failures_batch_{b + 1:04d}.ftr").exists()]
if missing:
    raise SystemExit(f"batches not done: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
summary = pd.concat([pd.read_feather(f) for f in sorted(batch_dir.glob("summary_batch_*.ftr"))], ignore_index=True)
failures = pd.concat([pd.read_feather(f) for f in sorted(batch_dir.glob("failures_batch_*.ftr"))], ignore_index=True)
if not (summary["neuron"].is_unique and failures["neuron"].is_unique
        and not summary["neuron"].isin(failures["neuron"]).any()
        and set(summary["neuron"]) | set(failures["neuron"]) == set(all_neurons)):
    raise SystemExit("the batches don't cover each cell exactly once; no table written")
n_checked = int(summary["princeton_checked"].sum())
if not summary.loc[summary["princeton_checked"], "princeton_match"].all():
    raise SystemExit("a checked cell's Princeton counts differ from step 09's; no table written")
print(f"cells labelled {len(summary):,}, failed {len(failures):,}; Princeton counts checked on {n_checked:,} cells, "
      f"all equal to step 09's")
print(f"synapse ends: {int(summary['n_buhmann'].sum()):,} attached-or-not, {int(summary['n_attached'].sum()):,} "
      f"attached, {int(summary['n_labelled'].sum()):,} labelled, {int(summary['on_cut_node'].sum()):,} on a cut node")
print(f"labelled cells with no Buhmann synapse on either side (SI_buhmann empty): "
      f"{int(summary['SI_buhmann'].isna().sum()):,}")
if len(failures):
    print("failures:", failures["error"].str.split(":").str[0].value_counts().to_dict())
    print(failures.head(10).to_string())
if len(failures) != (ACCEPT_FAILED_CELLS or 0):
    raise SystemExit(f"{len(failures):,} cells failed, but {ACCEPT_FAILED_CELLS or 0} are accepted. Read the errors "
                     "above; if they are expected, set BUHMANN_ACCEPT_FAILED_CELLS to the count. No table written.")

# the labels as small codes (side 0 = pre, 1 = post; label 0 = A, 1 = D), not as text, to keep memory low
label_parts = []
for f in sorted(batch_dir.glob("labels_batch_*.ftr")):
    tb = feather.read_table(f)
    if not (set(tb["side"].to_pylist()) <= {"pre", "post"} and set(tb["label"].to_pylist()) <= {"A", "D"}):
        raise SystemExit(f"{f.name}: a side or label other than pre/post and A/D")
    label_parts.append(pd.DataFrame({
        "synapse_id": tb["synapse_id"].to_numpy(),
        "neuron": tb["neuron"].to_numpy(),
        "side": pc.equal(tb["side"], "post").to_numpy(zero_copy_only=False).astype(np.int8),
        "label": pc.equal(tb["label"], "D").to_numpy(zero_copy_only=False).astype(np.int8)}))
labels = pd.concat(label_parts, ignore_index=True)
del label_parts
end_key = np.sort(labels["synapse_id"].to_numpy() * 2 + labels["side"].to_numpy())
assert not (end_key[1:] == end_key[:-1]).any(), "a synapse end has two labels"
del end_key


#%%7 The synapse table
syn_dataset = ds.dataset(str(BUHMANN_SYNAPSE_SFC_FTR), format="feather")
syn_filter = (ds.field("pre").isin(all_neurons.tolist()) | ds.field("post").isin(all_neurons.tolist())) \
    if SUBSET_CELLS else None
key_table = syn_dataset.to_table(columns=["id", "pre", "post"], filter=syn_filter)
keys = pd.DataFrame({"id": key_table["id"].to_numpy(), "pre": key_table["pre"].to_numpy(),
                     "post": key_table["post"].to_numpy()})
del key_table
assert keys["id"].is_unique, "the Buhmann table's id is not unique"
assert all(keys[c].dtype == np.int64 for c in keys), keys.dtypes


def label_of_end(side, cells):
    """Each row's label code at one end (0 = A, 1 = D, -1 = none), matched on (synapse id, the row's cell there)."""
    sel = labels["side"].to_numpy() == side
    ids = labels["synapse_id"].to_numpy()[sel]
    order = np.argsort(ids, kind="stable")
    ids, owner, value = ids[order], labels["neuron"].to_numpy()[sel][order], labels["label"].to_numpy()[sel][order]
    row_ids = keys["id"].to_numpy()
    at = np.minimum(np.searchsorted(ids, row_ids), max(len(ids) - 1, 0))
    found = (ids[at] == row_ids) & (owner[at] == cells) if len(ids) else np.zeros(len(row_ids), bool)
    out = np.full(len(row_ids), -1, dtype=np.int8)
    out[found] = value[at[found]]
    return out


pre_label = label_of_end(0, keys["pre"].to_numpy())
post_label = label_of_end(1, keys["post"].to_numpy())
del labels
gc.collect()
keep = (pre_label >= 0) & (post_label >= 0) & (keys["pre"].to_numpy() != keys["post"].to_numpy())
si_by_cell = summary.set_index("neuron")["SI_buhmann"]
# pre code * 2 + post code indexes AA, AD, DA, DD (pre end first)
comp = pc.take(pa.array(["AA", "AD", "DA", "DD"], type=pa.string()),
               pa.array(pre_label[keep].astype(np.int64) * 2 + post_label[keep].astype(np.int64)))
si_pre = pa.array(keys.loc[keep, "pre"].map(si_by_cell).to_numpy(dtype=np.float64))
si_post = pa.array(keys.loc[keep, "post"].map(si_by_cell).to_numpy(dtype=np.float64))
print(f"rows of {BUHMANN_SYNAPSE_SFC_FTR.name} read: {len(keys):,}; kept with both ends labelled: {int(keep.sum()):,}")
del pre_label, post_label

# the published table again, one record batch at a time, written straight to a temporary file and renamed at the end
out_schema = syn_dataset.schema.remove_metadata()
synapse_table_out.parent.mkdir(parents=True, exist_ok=True)
tmp_out = synapse_table_out.with_name(f"{synapse_table_out.name}.{os.getpid()}.tmp")
row, kept_row = 0, 0
key_id, key_pre, key_post = keys["id"].to_numpy(), keys["pre"].to_numpy(), keys["post"].to_numpy()
with pa.ipc.new_file(str(tmp_out), out_schema, options=pa.ipc.IpcWriteOptions(compression="lz4")) as writer:
    for batch in syn_dataset.to_batches(filter=syn_filter):
        n = batch.num_rows
        if not (np.array_equal(batch.column("id").to_numpy(), key_id[row:row + n])
                and np.array_equal(batch.column("pre").to_numpy(), key_pre[row:row + n])
                and np.array_equal(batch.column("post").to_numpy(), key_post[row:row + n])):
            raise SystemExit("the synapse table's batches are not in the order of the key read; no table written")
        part = batch.filter(pa.array(keep[row:row + n]))
        k = part.num_rows
        for name, values in (("comp", comp), ("SI_pre", si_pre), ("SI_post", si_post)):
            i = part.schema.get_field_index(name)
            part = part.set_column(i, part.schema.field(name), values.slice(kept_row, k).cast(part.schema.field(name).type))
        if k:
            writer.write_batch(pa.RecordBatch.from_arrays(part.columns, schema=out_schema))
        row += n
        kept_row += k
assert row == len(keys) and kept_row == int(keep.sum())
tmp_out.replace(synapse_table_out)
print(f"wrote {synapse_table_out} ({kept_row:,} rows)")
out_comp = feather.read_table(synapse_table_out, columns=["comp"]).column("comp")
print("comp:", {d["values"]: d["counts"] for d in pc.value_counts(out_comp).to_pylist()})
del keys, key_id, key_pre, key_post, comp, si_pre, si_post, out_comp
gc.collect()


#%%8 The neuron table (step 05's compartment counts on the new synapse table)
syn = feather.read_table(synapse_table_out, columns=["pre", "post", "comp"]).to_pandas()
cells = pd.read_feather(BUHMANN_NEURON_SFC_FTR)
cells["neuron"] = cells["neuron"].astype(np.int64)
dtypes = cells.dtypes
for part in ("A", "D", "L"):
    pre_counts = syn.loc[syn["comp"].str[0] == part].groupby("pre").size()
    post_counts = syn.loc[syn["comp"].str[1] == part].groupby("post").size()
    cells[f"pre_{part}"] = cells["neuron"].map(pre_counts).fillna(0).astype(dtypes[f"pre_{part}"])
    cells[f"post_{part}"] = cells["neuron"].map(post_counts).fillna(0).astype(dtypes[f"post_{part}"])
cells["axon_correct"] = cells["pre_A"] / (cells["pre_A"] + cells["post_A"])
cells["dend_correct"] = cells["post_D"] / (cells["pre_D"] + cells["post_D"])
cells["SI"] = cells["neuron"].map(si_by_cell).astype(dtypes["SI"])
cells["linker"] = np.zeros(len(cells)).astype(dtypes["linker"])
in_syn = cells["neuron"].isin(pd.unique(syn[["pre", "post"]].values.ravel()))
keep_cells = cells["neuron"].isin(summary["neuron"]) & in_syn
print(f"cells of {BUHMANN_NEURON_SFC_FTR.name}: {len(cells):,}; kept (labelled and in the new synapse table): "
      f"{int(keep_cells.sum()):,}")
cells = cells.loc[keep_cells].reset_index(drop=True)
assert list(cells.columns) == list(dtypes.index)
assert (cells.dtypes == dtypes).all(), cells.dtypes[cells.dtypes != dtypes]
# written with the published schema, so a column that is empty in every kept cell keeps its type
write_whole(pa.Table.from_pandas(cells, schema=feather.read_table(BUHMANN_NEURON_SFC_FTR).schema.remove_metadata(),
                                 preserve_index=False), neuron_table_out)
print(f"wrote {neuron_table_out} ({len(cells):,} rows)")
