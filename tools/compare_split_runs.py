"""Compare two runs of step 08 on the same neurons: do they choose different cuts?

Step 08 (processing/08_alternative_split_methods.py) writes, per batch, a summary table
(one row per neuron: the published SFC split's SI and the node chosen by MaxSI and by
MinFisherP) and a node table (every candidate cut with its four side counts). Two runs that
differ in one input, for example the synapse table the cuts are chosen on, can pick
different nodes. A different node does not always mean a different cut: neighbouring
nodes with no synapses between them split the synapses the same way.

For each neuron in both runs and each method, this tool takes run A's chosen node, looks
up its side counts in run B's node table, and compares them with run B's own choice, so
both cuts are scored on run B's synapses. It reports
  - in how many neurons the node differs and in how many the cut differs (side counts
    differ, the two mirror orders counted as one), and neurons with a cut in only one run,
  - the score of each cut on run B's synapses (SI for MaxSI, |phi| for MinFisherP) and how
    many neurons cross a cutoff,
  - for MinFisherP, how many differing cuts are ties at p = 0 (the p-value underflows),
  - how many synapses change between axon and dendrite. From the node tables alone the
    two cuts can sit in up to three tree layouts (one below the other, either way, or
    neither), so the tool gives the smallest and largest change over the layouts the counts
    allow. On the 89 differing cuts of skeleton folder 7, raw against processed synapse
    table (3 Oct 2026), the exact value was the smallest in 87, and the largest can be far
    off. With --exact the tool rebuilds
    the neurons from the skeletons, as step 08 does, and counts exactly. The axon is the side
    with the higher presynaptic fraction (side 1 on a tie), step 08's rule for MaxSI; step 08
    names no axon for MinFisherP, and the tool applies the same rule there. Synapses on the
    two cut nodes are labelled under one cut only; they are not counted as changes, and their
    number is reported separately.
  - how many neurons' SFC SI differs between the runs.
Before comparing, it checks that each run's summary agrees with its own node table (the
side counts stored for each chosen node), so batch files from two different runs mixed in
one folder are caught, and that both runs have the same candidate nodes (same skeletons).

Run from the repository root, run A first:
    python tools/compare_split_runs.py RUN_A/batches RUN_B/batches
    python tools/compare_split_runs.py RUN/batches RUN/batches --expect-identical
RUN/batches holds step 08's summary_batch_*.ftr and nodes_batch_*.ftr files (for example
the batches folder written by tools/check_split_methods_on_subset.py). With
--expect-identical the tool exits with status 1 unless both runs have the same neurons, the
same chosen node for every neuron and method, the same side counts in the node tables,
and the same SFC SI.
    python tools/compare_split_runs.py RUN_A/batches RUN_B/batches --exact processed
--exact needs the skeletons in SWC_DIR and the synapse table run B used (raw =
SYNAPSE_TABLE_RAW_FTR, processed = SYNAPSE_TABLE_FTR); it checks that the rebuilt neurons
give run B's side counts before counting, and takes about 3 seconds per differing neuron. Output: summary.txt,
per_neuron.csv and, when the neuron sets differ, only_in_A.csv / only_in_B.csv, in
outputs/check/compare_split_runs/ (or --out). Node tables are read one batch file at a time;
1,144 neurons take about 2 seconds and 1 GB.
"""
import argparse
import hashlib
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.ipc as pa_ipc

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from config import OUTPUT_DIR, SWC_DIR, SYNAPSE_TABLE_FTR, SYNAPSE_TABLE_RAW_FTR  # noqa: E402

# The |phi| that matches SI = 0.1: the cutoff processing/phi_threshold.py recommends on the
# delivered SI_comparisons.ftr (its plot, outputs/phi_threshold/phi_threshold.svg, said 0.36 on
# 29 Sep 2026). Rerun that script if the SI or the split changes; set it here with --phi-cutoff.
PHI_CUTOFF = 0.36
SI_CUTOFF = 0.1
TABLE = ["side_1_pre", "side_1_post", "side_2_pre", "side_2_post"]
NODE_COLUMNS = ["neuron_id", "node_id", "parent_id", "child_id", *TABLE,
                "node_pre_excluded", "node_post_excluded", "fisher_p"]
METHODS = {"MaxSI": ("best_node_id", "best_side_"), "MinFisherP": ("best_fisher_node_id", "best_fisher_side_")}
SUMMARY_COLUMNS = ["neuron_id", "original_SI", "best_node_id", "best_fisher_node_id",
                   *[f"{prefix}{s}" for _, prefix in METHODS.values() for s in ("1_pre", "1_post", "2_pre", "2_post")]]


def batch_files(folder, kind):
    files = sorted(Path(folder).glob(f"{kind}_batch_*.ftr"))
    if not files:
        sys.exit(f"No {kind}_batch_*.ftr files in {folder}")
    return files


def read_columns(path, columns):
    """Read only these columns, with a clear message if the file lacks any of them."""
    present = set(pa_ipc.open_file(path).schema.names)
    missing = [c for c in columns if c not in present]
    if missing:
        sys.exit(f"{path} has no column(s) {missing}; is it a step 08 batch file?")
    return pd.read_feather(path, columns=columns)


def read_summaries(folder, name):
    s = pd.concat([read_columns(f, SUMMARY_COLUMNS) for f in batch_files(folder, "summary")],
                  ignore_index=True)
    if s["neuron_id"].duplicated().any():
        sys.exit(f"Run {name} lists a neuron twice; were two runs mixed in one folder?")
    return s.set_index("neuron_id")


def read_nodes(folder, name, keep):
    """The node rows of the neurons in keep, one batch file at a time."""
    parts = []
    for f in batch_files(folder, "nodes"):
        t = read_columns(f, NODE_COLUMNS)
        parts.append(t[t["neuron_id"].isin(keep)])
    n = pd.concat(parts, ignore_index=True).set_index(["neuron_id", "node_id"]).sort_index()
    if n.index.duplicated().any():
        sys.exit(f"Run {name}'s node tables list a (neuron, node) twice; were two runs mixed in one folder?")
    return n


def check_summary_matches_nodes(summary, nodes, name):
    """Each chosen node's side counts in the summary must equal that node's row in the node table."""
    bad = 0
    for col, prefix in METHODS.values():
        chosen = summary[summary[col].notna()]
        keys = list(zip(chosen.index, chosen[col].astype(np.int64)))
        missing = [k for k in keys if k not in nodes.index]
        if missing:
            sys.exit(f"Run {name}: {len(missing)} chosen nodes are not in its node table (e.g. {missing[0]}); "
                     "the summary and node files come from different runs.")
        stored = chosen[[f"{prefix}{s}" for s in ("1_pre", "1_post", "2_pre", "2_post")]].to_numpy(np.int64)
        bad += int((stored != nodes.loc[keys, TABLE].to_numpy(np.int64)).any(axis=1).sum())
    if bad:
        sys.exit(f"Run {name}: for {bad} chosen cuts the summary's side counts differ from the node table; "
                 "the summary and node files come from different runs.")


def entropy(pre, post):
    n = pre + post
    if n == 0 or pre == 0 or post == 0:
        return 0.0
    return -sum(x / n * math.log2(x / n) for x in (pre, post))


def si(a, b, c, d):
    n, snorm = a + b + c + d, entropy(a + c, b + d)
    if n == 0 or snorm == 0:
        return math.nan
    return 1 - ((a + b) * entropy(a, b) + (c + d) * entropy(c, d)) / n / snorm


def phi(a, b, c, d):
    den = math.sqrt(float((a + b) * (c + d) * (a + c) * (b + d)))
    return math.nan if den == 0 else abs(a * d - b * c) / den


def axon_side(a, b, c, d):
    """Step 08's rule: side 1 if its pre fraction is at least side 2's (a NaN compares False)."""
    f1 = a / (a + b) if a + b else math.nan
    f2 = c / (c + d) if c + d else math.nan
    return 1 if f1 >= f2 else 2


def label_change_range(u, v, total):
    """Smallest and largest number of synapses whose axon/dendrite label differs between cut u
    and cut v (two rows of the same neuron's node table), over the tree layouts the counts allow.

    For two cut nodes, each has one side that does not contain the other node (B_u, B_v), and
    M = total - B_u - B_v - the synapses on u and v lies between them. Within a layout the
    change is exact, since B_u, M and B_v each keep one label per cut; the counts only rule
    out layouts where M would be negative. Returns (min, max), or (None, None) if none fits."""
    side = lambda r, s: np.array([r[f"side_{s}_pre"], r[f"side_{s}_post"]], dtype=np.int64)  # noqa: E731
    on_u = np.array([u["node_pre_excluded"], u["node_post_excluded"]], dtype=np.int64)
    on_v = np.array([v["node_pre_excluded"], v["node_post_excluded"]], dtype=np.int64)
    axon_u, axon_v = axon_side(*side(u, 1), *side(u, 2)), axon_side(*side(v, 1), *side(v, 2))
    changes = []
    # v below u (in u's child side), u below v, or neither below the other
    for bu, bv in ((2, 1), (1, 2), (1, 1)):
        b_u, b_v = side(u, bu), side(v, bv)
        middle = total - b_u - b_v - on_u - on_v
        if (middle < 0).any():
            continue
        n = 0
        n += b_u.sum() if (bu == axon_u) != (3 - bv == axon_v) else 0
        n += middle.sum() if (3 - bu == axon_u) != (3 - bv == axon_v) else 0
        n += b_v.sum() if (3 - bu == axon_u) != (bv == axon_v) else 0
        changes.append(int(n))
    return (min(changes), max(changes)) if changes else (None, None)


def exact_label_changes(d, nodes_b, synapse_table):
    """Exact axon/dendrite label changes for the differing cuts, from the skeletons.

    Each neuron is rebuilt as step 08 does (methods_all: upload_swc, then
    heal_attach_princeton_non_process on its synapses from synapse_table). The rebuilt
    side counts of both cuts must equal run B's node table, otherwise that neuron is
    reported as not reproduced. Returns {(neuron, method): exact count or None}."""
    import pyarrow.dataset as pa_ds
    from config import METHODS_DIR
    sys.path.insert(0, str(METHODS_DIR))
    import methods_all as M  # noqa: E402  (only needed here)
    todo = {}
    for method in METHODS:
        if f"{method}_same_cut" not in d:
            continue
        rows = d[(d[f"{method}_status"] == "compared") & (~d[f"{method}_same_cut"].astype(bool))]
        for _, r in rows.iterrows():
            todo.setdefault(int(r["neuron_id"]), []).append((method, int(r[f"{method}_node_A"]), int(r[f"{method}_node_B"])))
    if not todo:
        return {}
    ids = list(todo)
    cols = ["pre_x", "pre_y", "pre_z", "post_x", "post_y", "post_z", "pre", "post"]
    synapses = pa_ds.dataset(str(synapse_table), format="feather").to_table(
        columns=cols, filter=pa_ds.field("pre").isin(ids) | pa_ds.field("post").isin(ids)).to_pandas()
    out = {}
    for nid, pairs in todo.items():
        swc = M.upload_swc(nid)
        neuron = None if swc is None else M.heal_attach_princeton_non_process(
            swc, synapses[(synapses["pre"] == nid) | (synapses["post"] == nid)].copy())
        if neuron is None or isinstance(neuron, str):
            out.update({(nid, m): None for m, _, _ in pairs})
            continue
        children = {}
        for node, parent in zip(neuron.nodes["node_id"], neuron.nodes["parent_id"]):
            children.setdefault(parent, []).append(node)
        con = neuron.connectors[["node_id", "type"]]

        def sides(cut):
            """0 on the cut node, 1 in the child's subtree, 2 elsewhere; checked against run B."""
            row = nodes_b.loc[(nid, cut)]
            below, stack = set(), [int(row["child_id"])]
            while stack:
                x = stack.pop()
                below.add(x)
                stack.extend(children.get(x, []))
            where = np.where(con["node_id"] == cut, 0, np.where(con["node_id"].isin(below), 1, 2))
            pre = (con["type"] == "pre").to_numpy()
            counts = (int((pre & (where == 1)).sum()), int((~pre & (where == 1)).sum()),
                      int((pre & (where == 2)).sum()), int((~pre & (where == 2)).sum()))
            if counts != tuple(int(row[c]) for c in TABLE):
                return None
            return np.where(where == 0, "", np.where(where == axon_side(*counts), "A", "D"))

        for method, u, v in pairs:
            lu, lv = sides(u), sides(v)
            out[(nid, method)] = None if lu is None or lv is None else int(((lu != "") & (lv != "") & (lu != lv)).sum())
    return out


def fingerprint(folder):
    h = hashlib.sha256()
    for f in batch_files(folder, "summary") + batch_files(folder, "nodes"):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def compare_neuron(nid, sa, sb, nodes_b, totals, cutoffs):
    row = {"neuron_id": int(nid), "SFC_SI_A": sa.at[nid, "original_SI"], "SFC_SI_B": sb.at[nid, "original_SI"]}
    for method, (col, _) in METHODS.items():
        node_a, node_b = sa.at[nid, col], sb.at[nid, col]
        if pd.isna(node_a) or pd.isna(node_b):
            row[f"{method}_status"] = "no cut in either run" if pd.isna(node_a) and pd.isna(node_b) else "cut in one run only"
            continue
        u, v = nodes_b.loc[(nid, int(node_a))], nodes_b.loc[(nid, int(node_b))]
        ta, tb = tuple(int(u[c]) for c in TABLE), tuple(int(v[c]) for c in TABLE)
        same_cut = ta == tb or ta == (tb[2], tb[3], tb[0], tb[1])
        score = si if method == "MaxSI" else phi
        sc_a, sc_b = score(*ta), score(*tb)
        above_a, above_b = sc_a >= cutoffs[method], sc_b >= cutoffs[method]
        lo, hi = (0, 0) if same_cut else label_change_range(u, v, totals.loc[nid].to_numpy(np.int64))
        row.update({f"{method}_status": "compared", f"{method}_same_node": int(node_a) == int(node_b),
                    f"{method}_same_cut": same_cut, f"{method}_score_A_cut": sc_a, f"{method}_score_B_cut": sc_b,
                    f"{method}_crosses_cutoff": above_a != above_b, f"{method}_above_cutoff": above_a or above_b,
                    f"{method}_label_changes_min": lo, f"{method}_label_changes_max": hi,
                    f"{method}_on_cut_nodes": 0 if same_cut else int(u["node_pre_excluded"] + u["node_post_excluded"]
                                                                       + v["node_pre_excluded"] + v["node_post_excluded"]),
                    f"{method}_node_A": int(node_a), f"{method}_node_B": int(node_b)})
        if method == "MinFisherP":
            row["MinFisherP_p0_tie"] = bool(u["fisher_p"] == 0 and v["fisher_p"] == 0)
    return row


def summarise(d, method, cutoff):
    status = d.get(f"{method}_status", pd.Series(dtype=object))
    ok = d[status == "compared"].copy()
    lines = ["", f"{method} ({len(ok):,} neurons compared)"]
    one_only = int((status == "cut in one run only").sum())
    if one_only:
        lines.append(f"  a cut in one run only: {one_only:,}")
    if ok.empty:
        return lines, one_only
    for c in ("same_node", "same_cut", "crosses_cutoff", "above_cutoff"):
        ok[f"{method}_{c}"] = ok[f"{method}_{c}"].astype(bool)
    cut = ok[~ok[f"{method}_same_cut"]]
    above = cut[cut[f"{method}_above_cutoff"]]
    lines += [f"  node differs: {int((~ok[f'{method}_same_node']).sum()):,}",
              f"  cut differs (side counts on run B's synapses): {len(cut):,}"]
    if len(cut):
        delta = cut[f"{method}_score_B_cut"] - cut[f"{method}_score_A_cut"]
        name = "SI" if method == "MaxSI" else "|phi|"
        lines += [f"  {name} of B's cut minus A's cut: min {delta.min():.4g}, median {delta.median():.4g}, "
                  f"max {delta.max():.4g}",
                  f"  neurons that cross the cutoff ({cutoff}): {int(cut[f'{method}_crosses_cutoff'].sum()):,}",
                  f"  differing cuts above the cutoff in either run: {len(above):,}"]
        for label, part in (("those above the cutoff", above), ("all differing cuts", cut)):
            lo = pd.to_numeric(part[f"{method}_label_changes_min"])
            hi = pd.to_numeric(part[f"{method}_label_changes_max"])
            unknown = int(lo.isna().sum())
            text = f"  synapses changing axon/dendrite, {label}: {int(lo.sum()):,} to {int(hi.sum()):,} (range from counts)"
            if f"{method}_exact" in part:
                exact = pd.to_numeric(part[f"{method}_exact"])
                text += f"; exact {int(exact.sum()):,}" + (f" ({int(exact.isna().sum())} neurons not reproduced)"
                                                          if exact.isna().any() else "")
            lines.append(text + (f" ({unknown} neurons with no layout that fits)" if unknown else ""))
        exact_col = f"{method}_exact" if f"{method}_exact" in cut else None
        per_neuron = pd.to_numeric(cut[exact_col or f"{method}_label_changes_min"]).dropna()
        if len(per_neuron):
            top = per_neuron.idxmax()
            lines.append(f"  largest single neuron: {int(cut.at[top, 'neuron_id'])} with "
                         f"{'' if exact_col else 'at least '}{int(per_neuron[top]):,} changes")
        lines.append(f"  synapses on the two cut nodes (labelled under one cut only, not counted): "
                     f"{int(pd.to_numeric(cut[f'{method}_on_cut_nodes']).sum()):,}")
        if method == "MinFisherP":
            lines.append(f"  of the differing cuts, ties at p = 0: {int(cut['MinFisherP_p0_tie'].astype(bool).sum()):,}")
    return lines, one_only + len(cut) + int((~ok[f"{method}_same_node"]).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_a", type=Path, help="batches folder of run A (whose choices are checked)")
    ap.add_argument("run_b", type=Path, help="batches folder of run B (whose synapses score both cuts)")
    ap.add_argument("--si-cutoff", type=float, default=SI_CUTOFF, help=f"SI cutoff for MaxSI (default {SI_CUTOFF})")
    ap.add_argument("--phi-cutoff", type=float, default=PHI_CUTOFF,
                    help=f"|phi| cutoff for MinFisherP (default {PHI_CUTOFF}, the value matched to SI 0.1)")
    ap.add_argument("--exact", choices=["raw", "processed"],
                    help="count label changes exactly from the skeletons, with the synapse table run B used")
    ap.add_argument("--expect-identical", action="store_true",
                    help="exit with status 1 unless the neuron sets, every cut and every SFC SI agree")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "check" / "compare_split_runs",
                    help="output folder (default outputs/check/compare_split_runs)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for old in ("summary.txt", "per_neuron.csv", "only_in_A.csv", "only_in_B.csv"):
        (args.out / old).unlink(missing_ok=True)  # never leave an earlier run's result behind

    sa, sb = read_summaries(args.run_a, "A"), read_summaries(args.run_b, "B")
    common = sa.index.intersection(sb.index)
    if len(common) == 0:
        sys.exit("No neuron has a result in both runs.")
    na, nb = read_nodes(args.run_a, "A", common), read_nodes(args.run_b, "B", common)
    check_summary_matches_nodes(sa.loc[common], na, "A")
    check_summary_matches_nodes(sb.loc[common], nb, "B")
    shape_a, shape_b = na[["parent_id", "child_id"]], nb[["parent_id", "child_id"]]
    if not (shape_a.index.equals(shape_b.index)
            and np.array_equal(shape_a.to_numpy(np.int64), shape_b.to_numpy(np.int64))):
        sys.exit("The two runs' candidate nodes (or their parent and child) differ: they were not run on the "
                 "same skeletons, so their cuts cannot be compared.")
    count_cols = TABLE + ["node_pre_excluded", "node_post_excluded"]
    counts_differ = int((na[count_cols].to_numpy(np.int64) != nb[count_cols].to_numpy(np.int64)).any(axis=1).sum())
    del na, shape_a, shape_b

    first = nb.groupby(level=0).first()
    totals = pd.DataFrame({"pre": first[["side_1_pre", "side_2_pre", "node_pre_excluded"]].sum(axis=1),
                           "post": first[["side_1_post", "side_2_post", "node_post_excluded"]].sum(axis=1)})
    cutoffs = {"MaxSI": args.si_cutoff, "MinFisherP": args.phi_cutoff}
    d = pd.DataFrame([compare_neuron(nid, sa, sb, nb, totals, cutoffs) for nid in common])
    if args.exact:
        table = SYNAPSE_TABLE_FTR if args.exact == "processed" else SYNAPSE_TABLE_RAW_FTR
        if not SWC_DIR.is_dir() or not Path(table).exists():
            sys.exit(f"--exact needs the skeletons in {SWC_DIR} and the synapse table {table}.")
        exact = exact_label_changes(d, nb, table)
        for method in METHODS:
            d[f"{method}_exact"] = [exact.get((int(n), method), 0 if s else None) if st == "compared" else None
                                    for n, s, st in zip(d["neuron_id"], d.get(f"{method}_same_cut", [True] * len(d)),
                                                        d.get(f"{method}_status", [None] * len(d)))]

    only_a, only_b = sa.index.difference(sb.index), sb.index.difference(sa.index)
    lines = [f"Compared {datetime.now():%Y-%m-%d %H:%M}",
             f"Run A: {args.run_a} (files {fingerprint(args.run_a)})",
             f"Run B: {args.run_b} (files {fingerprint(args.run_b)})", "",
             f"Neurons with a result: A {len(sa):,}, B {len(sb):,}, both {len(common):,}"
             + (f"; only in A {len(only_a):,}, only in B {len(only_b):,}" if len(only_a) or len(only_b) else "")]
    a_nan, b_nan = d["SFC_SI_A"].isna(), d["SFC_SI_B"].isna()
    gap = (d["SFC_SI_A"] - d["SFC_SI_B"]).abs()
    sfc_differ = int(((gap > 1e-12) | (a_nan != b_nan)).sum())
    lines.append(f"SFC split SI differs (> 1e-12, or missing in one run) in {sfc_differ:,} neurons, "
                 f"by > 0.01 in {int((gap > 0.01).sum()):,}")
    lines.append(f"Candidate cuts whose side counts differ between the two node tables: {counts_differ:,}")
    differ = 0
    for method in METHODS:
        method_lines, n_diff = summarise(d, method, cutoffs[method])
        lines += method_lines
        differ += n_diff
    lines += ["", "Counting: a cut differs when its four side counts on run B's synapses differ (mirror order "
                  "counted as the same); synapses on the two cut nodes themselves are not counted."]

    d.to_csv(args.out / "per_neuron.csv", index=False)
    if len(only_a):
        pd.Series(only_a, name="neuron_id").to_csv(args.out / "only_in_A.csv", index=False)
    if len(only_b):
        pd.Series(only_b, name="neuron_id").to_csv(args.out / "only_in_B.csv", index=False)
    if args.expect_identical:
        identical = differ == 0 and sfc_differ == 0 and counts_differ == 0 and not len(only_a) and not len(only_b)
        lines += ["", "PASS: the runs agree" if identical else "FAIL: the runs differ"]
    (args.out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWritten to {args.out}")
    if args.expect_identical:
        sys.exit(0 if identical else 1)


if __name__ == "__main__":
    # Compares the cuts of two step 08 runs on the same 1,144 neurons:
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/outputs/split_methods_check/run_2oct/batches                    (A: cuts chosen on the raw synapse table)
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/outputs/check/split_raw_vs_processed/run_processed_table/batches  (B: cuts chosen on the processed table)
    # --exact processed rebuilds the differing neurons from the skeletons under
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/data/raw/swc/783
    # with run B's synapse table
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/data/derived/synapses_783_article_princeton.ftr
    # and writes summary.txt and per_neuron.csv to
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/outputs/check/compare_split_runs
    # To run it, uncomment:
    # sys.argv[1:] = ["/Users/ohajyahia/PycharmProjects/mixed-polarity-paper/outputs/split_methods_check/run_2oct/batches",
    #                 "/Users/ohajyahia/PycharmProjects/mixed-polarity-paper/outputs/check/split_raw_vs_processed/run_processed_table/batches",
    #                 "--exact", "processed"]
    main()
