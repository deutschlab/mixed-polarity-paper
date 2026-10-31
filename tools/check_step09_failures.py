"""Run step 09 (MaxSI synapse labels) on five local cells with planted failures, and check what it writes.

Step 09 has to record a cell that fails, leave it out of the tables, refuse to write the tables until the
number of failed cells is accepted, stop on a MemoryError, and refuse batches made from another skeleton set
or raw table. None of this happens on the local skeletons as they are, where every cell labels. This tool
makes it happen.

How it works. The tool chooses five cells whose skeleton is under SWC_DIR: the local cell with the most
local partners in the SFC synapse table (the hub), the two partners with the most synapses with the hub,
which label, and the two of its other partners with the smallest skeleton files, which fail. For the
full-run branch it adds the first cell of the neuron table, by ID, that has no local skeleton.
It copies the skeletons into its own folder and plants two failures there, never in data/raw:
- one skeleton file is overwritten with text navis cannot read (a cell with no usable skeleton);
- one cell's synapse ends are all moved onto one skeleton leaf, in the tool's copy of the raw synapse table,
  so every cut leaves one side without a synapse.
It reads step 09 (or --script), makes text substitutions (each must match exactly once, or the tool stops):
the repository root; SWC_DIR, the raw and SFC synapse tables (also in config, which check_batch_record and
upload_swc read) and MAXSI_SUBSET_DIR pointed at the tool's copies; BATCH_SIZE; ONLY_LOCAL_SKELETONS; the
two ACCEPT settings; and a line before the batch loop that runs a plant file. It then runs the patched copy:

1. default settings: must stop for the cell without a skeleton, with both failures in the failures file,
   labels for the other three cells, the tie columns, the skeleton fingerprint, navis version and the tool's
   raw table in batch_run.json, no temporary file and no table; with ACCEPT_FAILED_CELLS = 2 alone it must
   still stop for the cell without a skeleton;
2. ACCEPT_CELLS_WITHOUT_SKELETON = True: must stop because ACCEPT_FAILED_CELLS is None;
3. ACCEPT_FAILED_CELLS = 1 (the wrong count): must stop;
4. ACCEPT_FAILED_CELLS = 2: must write the three tables and print the counts of ends lost, and then
   - the synapse table has the SFC table's schema, no row with a failed cell at either end, comp equal to
     the batch labels of both ends, SI_pre / SI_post equal to MaxSI_SI;
   - the neuron table holds the three labelled cells and neither failed cell, with SI = MaxSI_SI,
     pre_A ... post_L, axon_correct and dend_correct equal to a recount from the written synapse table (made
     here, independently of step 09), linker 0, and every other column (the 14 cell-property columns among
     them) equal to the SFC table's;
   - the SI table holds root_id (int64) and SI = MaxSI_SI of the three labelled cells;
5. the same again: must stop at the start and keep the three tables as they are; with one of them moved
   away, must stop;
6. one more skeleton file, a skeleton file of another size, or a changed raw table: must refuse the batches;
7. one skeleton file in two subfolders: must stop before anything runs;
8. a MemoryError planted in one cell: must stop and save no batch;
9. one cell per batch: the batches of the failed cells hold labels files with no rows but string columns,
   and the tables equal those of step 4; with one cell's summary copied into another batch, must stop;
10. every cell failing: must stop with a message and write nothing;
11. the full-run branch (ONLY_LOCAL_SKELETONS = False) on a six-cell neuron table, so that a cell has no
    skeleton file at all: must stop for it, record a FileNotFoundError, and, once the three failures are
    accepted, write the same tables as step 4.
It also runs step 09's label_by_cut_node on a five-node chain whose two sides have the same presynaptic
fraction: it must raise, and label a chain whose sides differ. It does not test the temporary-name writing
of batch files under a stopped run.

Writes outputs/tools/check_step09_failures/report.txt (and its working copies in work/), prints PASS or
FAIL per check and exits with status 1 if any fails. Needs the SFC tables (steps 04 and 05), the raw
synapse table (processing/build_raw_synapse_table.py) and at least five local skeletons. About four minutes.
Run it with --script on a copy of step 09 with a planted bug to see a check fail.
"""
import argparse
import ast
import hashlib
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import navis
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.feather as feather

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from config import (OUTPUT_DIR, SWC_DIR, NEURON_TABLE_SFC_FTR, SYNAPSE_TABLE_SFC_FTR,  # noqa: E402
                    SYNAPSE_TABLE_RAW_FTR, MAXSI_SYNAPSE_TABLE_FTR, MAXSI_NEURON_TABLE_FTR,
                    MAXSI_SI_UPDATED_FTR)

STEP09 = REPO / "processing" / "09_maxsi_synapse_labels.py"
RECOUNTED = ["pre_A", "pre_D", "pre_L", "post_A", "post_D", "post_L", "axon_correct", "dend_correct"]
COPIED = ["out_synapses_count", "in_synapses_count", "max_out_degree", "max_in_degree", "mean_in_degree",
          "mean_out_degree", "out_partners", "in_partners", "partners_ratio", "max_degree_ratio",
          "mean_degree_ratio", "mean_out_synapses_per_partner", "mean_in_synapses_per_partner",
          "mean_synapses_per_partner_ratio"]


def choose_cells():
    """The hub (most local partners), its two partners with the most synapses with it, then two more partners,
    and the first cell of the neuron table, by ID, with no local skeleton. When every cell has one, the first cell
    not among those five: its file is not copied to the check's own skeleton folder, so it has none there."""
    neurons = set(pd.read_feather(NEURON_TABLE_SFC_FTR, columns=["neuron"])["neuron"])
    files = {np.int64(p.stem): p for p in SWC_DIR.glob("*/*.swc") if p.stem.isdigit()}
    local = sorted(set(files) & neurons)
    if len(local) < 5:
        raise SystemExit(f"needs at least five cells with a skeleton under {SWC_DIR}; found {len(local)}")
    rows = (ds.dataset(str(SYNAPSE_TABLE_SFC_FTR), format="feather")
            .to_table(columns=["pre", "post"], filter=ds.field("pre").isin(local) & ds.field("post").isin(local))
            .to_pandas())
    pairs = rows.drop_duplicates()
    partners = pd.concat([pairs.rename(columns={"pre": "a", "post": "b"}),
                          pairs.rename(columns={"post": "a", "pre": "b"})]).drop_duplicates()
    degree = partners.groupby("a").size().sort_index().sort_values(ascending=False, kind="stable")
    hub = np.int64(degree.index[0])
    with_hub = pd.concat([rows.loc[rows["pre"] == hub, "post"], rows.loc[rows["post"] == hub, "pre"]]).value_counts()
    if len(with_hub) < 4:
        raise SystemExit(f"the hub {hub} has {len(with_hub)} local partners; needs four")
    # the two partners with the most synapses with the hub label; the two smallest skeletons of the rest fail
    order = sorted(with_hub.index, key=lambda c: (-with_hub[c], c))
    labelled, rest = order[:2], sorted(order[2:], key=lambda c: (files[c].stat().st_size, c))
    chosen = [np.int64(c) for c in labelled + rest[:2]]
    without = neurons - set(files)
    no_skeleton = np.int64(min(without if without else neurons - {hub, *chosen}))
    return hub, chosen, no_skeleton, files


def make_inputs(work, hub, mates, no_skeleton, files):
    """Skeleton copies, raw and SFC synapse extracts, a small neuron table, and the planted failures."""
    cells = [hub] + mates
    ok_cells, unreadable, one_side = [hub, mates[0], mates[1]], mates[2], mates[3]
    swc = work / "swc"
    (swc / "local").mkdir(parents=True)
    for c in cells:
        shutil.copy(files[c], swc / "local" / files[c].name)
    (swc / "local" / f"{unreadable}.swc").write_text("this is not a skeleton\n")

    raw = (ds.dataset(str(SYNAPSE_TABLE_RAW_FTR), format="feather")
           .to_table(filter=ds.field("pre").isin(cells) | ds.field("post").isin(cells)).to_pandas())
    if not ((raw["pre"] == one_side).any() and (raw["post"] == one_side).any()):
        raise SystemExit(f"cell {one_side} needs both pre- and postsynapses for the one-side plant")
    skeleton = navis.read_swc(str(files[one_side]))
    leaf = skeleton.nodes.loc[skeleton.nodes["type"] == "end"].iloc[0]
    for end in ("pre", "post"):
        on = raw[end] == one_side
        for axis in "xyz":
            raw.loc[on, f"{end}_{axis}"] = raw[f"{end}_{axis}"].dtype.type(leaf[axis])
    raw.reset_index(drop=True).to_feather(work / "raw_synapses.ftr")

    sfc = (ds.dataset(str(SYNAPSE_TABLE_SFC_FTR), format="feather")
           .to_table(filter=ds.field("pre").isin(cells) | ds.field("post").isin(cells)))
    feather.write_feather(sfc, work / "sfc_synapses.ftr")

    # for the full-run branch: the five cells and one with no skeleton, in the SFC table's order
    table = pd.read_feather(NEURON_TABLE_SFC_FTR)
    table.loc[table["neuron"].isin(cells + [no_skeleton])].reset_index(drop=True).to_feather(work / "neuron_table.ftr")
    return ok_cells, unreadable, one_side


def patched_copy(script, work, run_dir, accept_skeleton, accept_failed, plant, full, batch_size):
    s = script.read_text(encoding="utf-8")
    import_end = "MAXSI_SYNAPSE_TABLE_FTR, MAXSI_NEURON_TABLE_FTR, MAXSI_SI_UPDATED_FTR)\n"
    overrides = (import_end
                 + f"SWC_DIR = Path(r'{work / 'swc'}')\n"
                 + f"SYNAPSE_TABLE_RAW_FTR = Path(r'{work / 'raw_synapses.ftr'}')\n"
                 + "import config   # methods_all.upload_swc and check_batch_record read these from config\n"
                 + "config.SWC_DIR, config.SYNAPSE_TABLE_RAW_FTR = SWC_DIR, SYNAPSE_TABLE_RAW_FTR\n"
                 + f"SYNAPSE_TABLE_SFC_FTR = Path(r'{work / 'sfc_synapses.ftr'}')\n"
                 + f"MAXSI_SUBSET_DIR = Path(r'{run_dir}')\n")
    if full:
        overrides += (f"NEURON_TABLE_SFC_FTR = Path(r'{work / 'neuron_table.ftr'}')\n"
                      f"MAXSI_LABEL_BATCH_DIR = Path(r'{run_dir / 'batches'}')\n"
                      f"MAXSI_SYNAPSE_TABLE_FTR = Path(r'{run_dir}') / MAXSI_SYNAPSE_TABLE_FTR.name\n"
                      f"MAXSI_NEURON_TABLE_FTR = Path(r'{run_dir}') / MAXSI_NEURON_TABLE_FTR.name\n"
                      f"MAXSI_SI_UPDATED_FTR = Path(r'{run_dir}') / MAXSI_SI_UPDATED_FTR.name\n")
    subs = [("sys.path.insert(0, str(Path(__file__).parent.parent))", f"sys.path.insert(0, r'{REPO}')"),
            (import_end, overrides),
            ("BATCH_SIZE = 500\n", f"BATCH_SIZE = {batch_size}\n"),
            ("ONLY_LOCAL_SKELETONS = False", f"ONLY_LOCAL_SKELETONS = {not full}"),
            ("ACCEPT_CELLS_WITHOUT_SKELETON = False", f"ACCEPT_CELLS_WITHOUT_SKELETON = {accept_skeleton}"),
            ("ACCEPT_FAILED_CELLS = None", f"ACCEPT_FAILED_CELLS = {accept_failed}"),
            ("#%%6 Run the batches\n", "#%%6 Run the batches\n" + f"exec(open(r'{plant}').read())\n")]
    for old, new in subs:
        if s.count(old) != 1:
            raise SystemExit(f"{script.name}: {old.strip()!r} found {s.count(old)} times, not once; not run")
        s = s.replace(old, new)
    out = work / "patched_step09.py"
    out.write_text(s, encoding="utf-8")
    return out


def run(script, work, run_dir, accept_skeleton=False, accept_failed=None, plant="", full=False, batch_size=500):
    plant_file = work / "plant.py"
    plant_file.write_text(plant)
    patched = patched_copy(script, work, run_dir, accept_skeleton, accept_failed, plant_file, full, batch_size)
    done = subprocess.run([sys.executable, str(patched)], cwd=work, capture_output=True, text=True)
    return done.returncode, done.stdout + done.stderr


def recount(syn, neuron_ids):
    """Step 05's compartment counts, written out again here (05:223-256)."""
    r = pd.DataFrame({"neuron": neuron_ids})
    for k, end in ((0, "pre"), (1, "post")):
        for c in "ADL":
            counts = syn.loc[syn["comp"].str[k] == c].groupby(end).size()
            r[f"{end}_{c}"] = r["neuron"].map(counts).fillna(0).astype(np.int64)
    r["axon_correct"] = r["pre_A"] / (r["pre_A"] + r["post_A"])
    r["dend_correct"] = r["post_D"] / (r["pre_D"] + r["post_D"])
    return r


def read_tables(run_dir):
    return (pd.read_feather(run_dir / MAXSI_SYNAPSE_TABLE_FTR.name), pd.read_feather(run_dir / MAXSI_NEURON_TABLE_FTR.name),
            pd.read_feather(run_dir / MAXSI_SI_UPDATED_FTR.name))


def tie_checks(script):
    """label_by_cut_node, taken from the script, on a five-node chain 1-2-3-4-5 cut at node 3."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "label_by_cut_node")
    sys.path.insert(0, str(REPO / "methods"))
    from node_cut_si import get_descendants_from_node
    space = {"np": np, "pd": pd, "get_descendants_from_node": get_descendants_from_node}
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(script), "exec"), space)

    def chain(types_at):
        nodes = pd.DataFrame({"node_id": [1, 2, 3, 4, 5], "parent_id": [-1, 1, 2, 3, 4],
                              "x": [0., 1, 2, 3, 4], "y": 0., "z": 0., "radius": 0.})
        n = navis.TreeNeuron(nodes, units="nm")
        rows = [(i, t, node) for i, (node, t) in enumerate(types_at)]
        n.connectors = pd.DataFrame({"connector_id": [r[0] for r in rows], "synapse_id": [100 + r[0] for r in rows],
                                     "type": [r[1] for r in rows], "node_id": [r[2] for r in rows],
                                     "x": 0., "y": 0., "z": 0.})
        return n

    results = []
    same = chain([(1, "pre"), (1, "post"), (5, "pre"), (5, "post")])
    row = {"node_id": 3, "side_1_pre": 1, "side_1_post": 1, "side_2_pre": 1, "side_2_post": 1}
    try:
        space["label_by_cut_node"](same, row)
        results.append((False, "equal presynaptic fractions on both sides: labelled, should fail"))
    except ValueError as e:
        results.append(("same presynaptic fraction" in str(e), f"equal presynaptic fractions on both sides: {e}"))
    differ = chain([(1, "post"), (1, "post"), (5, "pre"), (5, "pre")])
    row = {"node_id": 3, "side_1_pre": 2, "side_1_post": 0, "side_2_pre": 0, "side_2_post": 2}
    try:
        labelled, sides = space["label_by_cut_node"](differ, row)
        got = dict(zip(labelled["synapse_id"], labelled["label"]))
        results.append((got == {100: "D", 101: "D", 102: "A", 103: "A"} and sides["axon_side"] == 1,
                        f"sides that differ: labels {got}, axon side {sides['axon_side']}"))
    except Exception as e:
        results.append((False, f"sides that differ: raised {e!r}"))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--script", type=Path, default=STEP09, help="step 09 file (default: processing/09_maxsi_synapse_labels.py)")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "tools" / "check_step09_failures",
                    help="output folder (default outputs/tools/check_step09_failures)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.txt").unlink(missing_ok=True)   # never leave an earlier run's PASS behind
    work = args.out / "work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()

    header = [f"Checked {datetime.now():%Y-%m-%d %H:%M}",
              f"Step 09 code: {args.script} (sha256 {hashlib.sha256(args.script.read_bytes()).hexdigest()[:16]})"]
    lines, failures = [], 0

    def report(ok, text):
        nonlocal failures
        failures += not ok
        lines.append(f"{'PASS' if ok else 'FAIL'}  {text}")
        print(lines[-1])

    def logged(name, *a, **k):
        code, log = run(args.script, work, *a, **k)
        (work / f"{name}.log").write_text(log)
        return code, log

    hub, mates, no_skeleton, files = choose_cells()
    ok_cells, unreadable, one_side = make_inputs(work, hub, mates, no_skeleton, files)
    header.append(f"Cells: labelled {[int(c) for c in ok_cells]}, unreadable skeleton {unreadable}, "
                  f"all synapses on one leaf {one_side}; no skeleton (full-run branch) {no_skeleton}")
    print("\n".join(header))
    run_dir = work / "run"
    batches = run_dir / "batches"
    outs = [run_dir / MAXSI_SYNAPSE_TABLE_FTR.name, run_dir / MAXSI_NEURON_TABLE_FTR.name,
            run_dir / MAXSI_SI_UPDATED_FTR.name]
    failed = [unreadable, one_side]

    def none_exist(d=run_dir):
        return not any((d / p.name).exists() for p in outs)

    def tables_aside(d):
        """Move the three tables of folder d aside (a run stops at once when all three exist), and return a function
        that puts them back."""
        names = [MAXSI_SYNAPSE_TABLE_FTR.name, MAXSI_NEURON_TABLE_FTR.name, MAXSI_SI_UPDATED_FTR.name]
        moved = [(d / n, d / (n + ".aside")) for n in names if (d / n).exists()]
        for a, b in moved:
            a.rename(b)
        return lambda: [b.rename(a) for a, b in moved]

    def same_tables(a, b):
        ta, tb = read_tables(a), read_tables(b)
        return (ta[0].sort_values("synapse_id").reset_index(drop=True).equals(tb[0].sort_values("synapse_id").reset_index(drop=True))
                and ta[1].equals(tb[1])
                and ta[2].sort_values("root_id").reset_index(drop=True).equals(tb[2].sort_values("root_id").reset_index(drop=True)))

    # 1. default settings
    code, log = logged("run1", run_dir)
    report(code != 0 and "cells failed for want of a usable skeleton" in log and none_exist(),
           f"default settings: stops for the cell without a usable skeleton, no table (exit {code})")
    try:
        fails = pd.concat([pd.read_feather(p) for p in sorted(batches.glob("failures_batch_*.ftr"))])
        summary = pd.concat([pd.read_feather(p) for p in sorted(batches.glob("summary_batch_*.ftr"))])
        labels = pd.concat([pd.read_feather(p) for p in sorted(batches.glob("labels_batch_*.ftr"))])
        err = dict(zip(fails["neuron"], fails["error"]))
        report(set(err) == set(failed)
               and err.get(unreadable, "").startswith("OSError: No usable skeleton")
               and "contains no synapses" in err.get(one_side, ""),
               f"failures file: {err}")
        cols = ["n_tied_max_SI", "n_tied_max_SI_distinct_sides", "n_close_max_SI"]
        report(set(summary["neuron"]) == set(ok_cells) and set(labels["neuron"]) == set(ok_cells)
               and all(c in summary.columns for c in cols)
               and bool((summary["n_tied_max_SI_distinct_sides"] <= summary["n_tied_max_SI"]).all())
               and bool((summary["n_close_max_SI"] >= summary["n_tied_max_SI"]).all()),
               f"labels and summaries for exactly the three labelled cells; tie columns "
               f"{summary[cols].to_dict('list') if all(c in summary.columns for c in cols) else 'missing'}")
        record = (batches / "batch_run.json").read_text()
        report(all(k in record for k in ('"swc_files": 5', "swc_files_sha256", f'"navis_version": "{navis.__version__}"',
                                         '"synapse_table": "raw_synapses.ftr"')),
               "batch_run.json holds the skeleton count (5), their hash, the navis version and the tool's raw table")
        report(not list(batches.glob("*.tmp")), "no temporary batch file left behind")
    except Exception as e:
        report(False, f"batch files after run 1: {e!r}")
        summary = labels = None

    code, log = logged("run1b", run_dir, accept_failed=2)
    report(code != 0 and "cells failed for want of a usable skeleton" in log and none_exist(),
           f"ACCEPT_FAILED_CELLS = 2 but cells without a skeleton not accepted: stops, no table (exit {code})")

    # 2 and 3. the failed cells not accepted
    code, log = logged("run2", run_dir, accept_skeleton=True)
    report(code != 0 and "2 cells failed, and ACCEPT_FAILED_CELLS is None" in log and none_exist(),
           f"ACCEPT_FAILED_CELLS = None: stops before writing (exit {code})")
    code, log = logged("run3", run_dir, accept_skeleton=True, accept_failed=1)
    report(code != 0 and "ACCEPT_FAILED_CELLS is 1" in log and none_exist(),
           f"ACCEPT_FAILED_CELLS = 1, the wrong count: stops before writing (exit {code})")

    # 4. accepted: the three tables
    code, log = logged("run4", run_dir, accept_skeleton=True, accept_failed=2)
    written = code == 0 and all(p.exists() for p in outs)
    report(written, f"ACCEPT_FAILED_CELLS = 2: writes the three tables (exit {code})")
    report(all(t in log for t in ("dropped because the MaxSI split failed: 2", "Labelled ends with no row in",
                                  "Labelled ends lost because the other end", "on the cut node, so unlabelled",
                                  "empty here but not in the SFC table", "within numpy isclose")),
           "prints the failed cells dropped, the ends lost and with no SFC row, isclose ties and newly empty ratios")
    if written and summary is not None:
        syn, neu, si_table = read_tables(run_dir)
        si = summary.set_index("neuron")["MaxSI_SI"]
        sfc_schema = ds.dataset(str(work / "sfc_synapses.ftr"), format="feather").schema
        report(feather.read_table(outs[0]).schema.equals(sfc_schema, check_metadata=False),
               "synapse table: the SFC table's columns and types")
        sfc_rows = pd.read_feather(work / "sfc_synapses.ftr", columns=["pre", "post"])
        n_failed_rows = int((sfc_rows["pre"].isin(failed) & sfc_rows["post"].isin(ok_cells)
                             | sfc_rows["post"].isin(failed) & sfc_rows["pre"].isin(ok_cells)).sum())
        report(len(syn) > 0 and n_failed_rows > 0 and not (syn["pre"].isin(failed) | syn["post"].isin(failed)).any()
               and bool((syn["pre"].isin(ok_cells) & syn["post"].isin(ok_cells)).all()),
               f"synapse table: {len(syn)} rows, none of the {n_failed_rows} SFC rows between a failed and a "
               "labelled cell")
        lab = labels.set_index(["synapse_id", "neuron", "side"])["label"]
        want = [lab.get((s, p, "pre"), "?") + lab.get((s, q, "post"), "?")
                for s, p, q in zip(syn["synapse_id"], syn["pre"], syn["post"])]
        report(list(syn["comp"]) == want, "synapse table: comp = the batch labels of the two ends")
        report(np.array_equal(syn["SI_pre"].to_numpy(), syn["pre"].map(si).to_numpy())
               and np.array_equal(syn["SI_post"].to_numpy(), syn["post"].map(si).to_numpy()),
               "synapse table: SI_pre and SI_post = MaxSI_SI")
        report(sorted(neu["neuron"]) == sorted(int(c) for c in ok_cells) and not neu["neuron"].isin(failed).any(),
               f"neuron table: rows {sorted(neu['neuron'].tolist())}, the labelled cells only")
        report(np.array_equal(neu["SI"].to_numpy(), neu["neuron"].map(si).to_numpy()), "neuron table: SI = MaxSI_SI")
        mine = recount(syn, neu["neuron"].to_numpy())
        bad = [c for c in RECOUNTED if not np.allclose(neu[c].to_numpy(float), mine[c].to_numpy(float),
                                                        rtol=0, atol=0, equal_nan=True)]
        report(not bad, "neuron table: pre_A ... post_L, axon_correct, dend_correct equal a recount from the synapse table"
               + (f"; differ: {bad}" if bad else ""))
        sfc_neu = pd.read_feather(NEURON_TABLE_SFC_FTR)
        sfc_same = sfc_neu.set_index("neuron").loc[neu["neuron"]].reset_index()
        other = [c for c in sfc_neu.columns if c not in RECOUNTED + ["SI", "linker"]]
        bad = [c for c in other if not sfc_same[c].equals(neu[c])]
        report(not bad, f"neuron table: the {len(COPIED)} cell-property columns and every other column equal the SFC "
               "table's" + (f"; differ: {bad}" if bad else ""))
        report(neu["linker"].dtype == sfc_neu["linker"].dtype and bool((neu["linker"] == 0).all()),
               "neuron table: linker is 0, with the SFC table's dtype")
        report(list(neu.columns) == list(sfc_neu.columns) and (neu.dtypes == sfc_neu.dtypes).all(),
               "neuron table: the SFC table's columns and dtypes")
        report(list(si_table.columns) == ["root_id", "SI"] and si_table["root_id"].dtype == np.int64
               and sorted(si_table["root_id"]) == sorted(int(c) for c in ok_cells)
               and np.array_equal(si_table["SI"].to_numpy(), si_table["root_id"].map(si).to_numpy()),
               f"SI table: root_id (int64) and SI = MaxSI_SI for the {len(si_table)} labelled cells")

        # 5. never overwrite, and all three or none
        before = [p.read_bytes() for p in outs]
        code, log = logged("run5", run_dir, accept_skeleton=True, accept_failed=2)
        report(code != 0 and "Nothing to do" in log and [p.read_bytes() for p in outs] == before,
               f"run again: stops at the start, the three tables kept as they were (exit {code})")
        aside = outs[2].with_name(outs[2].name + ".aside")
        outs[2].rename(aside)
        code, log = logged("run5b", run_dir, accept_skeleton=True, accept_failed=2)
        report(code != 0 and "Only some of the three MaxSI tables exist" in log and not outs[2].exists(),
               f"two of the three tables: stops (exit {code})")
        aside.rename(outs[2])

    # 6. another skeleton set, another size, another raw table (tables aside, or the run stops at the start)
    put_back = tables_aside(run_dir)
    extra = work / "swc" / "local" / "1.swc"
    shutil.copy(files[ok_cells[1]], extra)
    code, log = logged("run6", run_dir, accept_skeleton=True, accept_failed=2)
    extra.unlink()
    report(code != 0 and "come from another run" in log and "MAXSI_SUBSET_DIR" in log,
           f"one more skeleton file: the batches are refused (exit {code})")
    grown = work / "swc" / "local" / files[ok_cells[1]].name
    kept_bytes = grown.read_bytes()
    grown.write_bytes(kept_bytes + b"# one more comment line\n")
    code, log = logged("run6b", run_dir, accept_skeleton=True, accept_failed=2)
    grown.write_bytes(kept_bytes)
    report(code != 0 and "come from another run" in log, f"a skeleton file of another size: the batches are refused (exit {code})")
    raw_file = work / "raw_synapses.ftr"
    raw_bytes = raw_file.read_bytes()
    pd.read_feather(raw_file).iloc[1:].reset_index(drop=True).to_feather(raw_file)
    code, log = logged("run6c", run_dir, accept_skeleton=True, accept_failed=2)
    raw_file.write_bytes(raw_bytes)
    report(code != 0 and "come from another run" in log, f"a changed raw synapse table: the batches are refused (exit {code})")

    # 7. one name in two subfolders
    (work / "swc" / "second").mkdir()
    shutil.copy(files[ok_cells[1]], work / "swc" / "second" / files[ok_cells[1]].name)
    code, log = logged("run7", run_dir, accept_skeleton=True, accept_failed=2)
    shutil.rmtree(work / "swc" / "second")
    report(code != 0 and "more than one subfolder" in log, f"a skeleton name in two subfolders: stops (exit {code})")
    put_back()

    # 8. MemoryError
    plant = ("_real_one_cell = maxsi_labels_one_cell\n"
             "def maxsi_labels_one_cell(neuron_id, neuron_synapses):\n"
             f"    if int(neuron_id) == {int(ok_cells[1])}:\n"
             "        raise MemoryError('planted')\n"
             "    return _real_one_cell(neuron_id, neuron_synapses)\n")
    mem_dir = work / "run_memory"
    code, log = logged("run8", mem_dir, accept_skeleton=True, accept_failed=2, plant=plant)
    report(code != 0 and "MemoryError" in log and not list((mem_dir / "batches").glob("*_batch_*.ftr")),
           f"a MemoryError in one cell: the run stops and saves no batch (exit {code})")

    # 9. one cell per batch, so that two batches hold only a failure
    one_dir = work / "run_one_per_batch"
    code, log = logged("run9", one_dir, accept_skeleton=True, accept_failed=2, batch_size=1)
    empty = [p for p in sorted((one_dir / "batches").glob("labels_batch_*.ftr")) if feather.read_table(p).num_rows == 0]
    types = {str(feather.read_table(p).schema.field(c).type) for p in empty for c in ("side", "label")}
    report(len(empty) == 2 and types == {"string"},
           f"batches with only a failed cell: {len(empty)} labels files with no rows, side and label typed {sorted(types)}")
    report(code == 0 and written and same_tables(one_dir, run_dir),
           f"one cell per batch: the same three tables as one batch (exit {code})")
    if code == 0:
        # a labelled cell's summary copied into a failed cell's batch: that cell is then in two batches
        summaries = sorted((one_dir / "batches").glob("summary_batch_*.ftr"))
        empty_summary = next(p for p in summaries if feather.read_table(p).num_rows == 0)
        full_summary = next(p for p in summaries if feather.read_table(p).num_rows == 1)
        kept_summary = empty_summary.read_bytes()
        empty_summary.write_bytes(full_summary.read_bytes())
        put_back = tables_aside(one_dir)
        code, log = logged("run9b", one_dir, accept_skeleton=True, accept_failed=2, batch_size=1)
        put_back()
        empty_summary.write_bytes(kept_summary)
        report(code != 0 and ("both labelled and failed" in log or "appear in two batches" in log),
               f"a cell in two batches: stops (exit {code})")

    # 10. every cell fails
    plant = ("def maxsi_labels_one_cell(neuron_id, neuron_synapses):\n"
             "    raise ValueError('planted')\n")
    none_dir = work / "run_none_labelled"
    code, log = logged("run10", none_dir, accept_skeleton=True, accept_failed=5, plant=plant)
    report(code != 0 and "No cell was labelled" in log and none_exist(none_dir),
           f"no cell labelled: stops with a message, no table (exit {code})")

    # 11. the full-run branch: a small neuron table, a sparse skeleton folder, syn_filter None
    full_dir = work / "run_full"
    code, log = logged("run11", full_dir, full=True)
    report(code != 0 and "cells failed for want of a usable skeleton" in log and none_exist(full_dir),
           f"full run, default settings: stops for the cells without a usable skeleton (exit {code})")
    try:
        fails = pd.concat([pd.read_feather(p) for p in sorted((full_dir / "batches").glob("failures_batch_*.ftr"))])
        err = dict(zip(fails["neuron"], fails["error"]))
        report(err.get(no_skeleton, "").startswith("FileNotFoundError: No usable skeleton") and len(err) == 3,
               f"full run: the cell with no file fails with FileNotFoundError ({err.get(no_skeleton)})")
    except Exception as e:
        report(False, f"full run batch files: {e!r}")
    code, log = logged("run11b", full_dir, accept_skeleton=True, accept_failed=3, full=True)
    report(code == 0 and "MAXSI_SUBSET_DIR" not in log and "Subset run" not in log and written
           and same_tables(full_dir, run_dir),
           f"full run, three failures accepted: the same three tables as the subset run (exit {code})")

    for ok, text in tie_checks(args.script):
        report(ok, "label_by_cut_node: " + text)

    summary_line = f"\n{sum(l.startswith('PASS') for l in lines)} passed, {failures} failed"
    print(summary_line)
    (args.out / "report.txt").write_text("\n".join(header + [""] + lines) + summary_line + "\n", encoding="utf-8")
    print(f"Report: {args.out / 'report.txt'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    # Needs the SFC tables, the raw synapse table and five local skeletons. Writes
    #   outputs/tools/check_step09_failures/report.txt
    # To check a changed copy of step 09, uncomment:
    # sys.argv[1:] = ["--script", "path/to/copy_of_09.py"]
    main()
