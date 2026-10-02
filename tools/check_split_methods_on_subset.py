"""Run step 08 (alternative split methods) on a subset of neurons, and compare the result with
a reference table.

Step 08 is written for every intrinsic neuron at once, which needs all ~139,000 skeletons
and many hours. This tool runs the same code on fewer neurons, by default every intrinsic
neuron whose skeleton is under SWC_DIR, so the method can be checked on a partial download.

How it works. The tool does not keep its own copy of step 08. It reads
processing/08_alternative_split_methods.py and makes five text substitutions:

1. the repository root is given as a path, because the patched copy runs from --out;
2. ALT_SPLIT_BATCH_DIR and SI_COMPARISONS_FTR are redirected into --out;
3. the neuron list is cut to the chosen neurons, after step 08's own super-class filter;
4. only the synapses that touch a chosen neuron are read from SYNAPSE_TABLE_RAW_FTR.
   Step 08 selects each neuron's synapses as pre == neuron or post == neuron, so this
   changes no neuron's synapses, only the memory and time needed;
5. the batch range of the combine step is set to the number of batches written.

Each substitution must match exactly once, or the tool stops before running anything, so
a changed step 08 cannot be run half-patched. The patched copy is saved as
--out/patched_step08.py. Everything else, including the cut search, SI, Fisher test and
Phi, is step 08's own code. Step 08 computes every score per neuron, so which other
neurons share a batch does not change a neuron's result.

After the run (or with --compare-only, on an existing --out) the tool writes
--out/per_neuron.csv: step 08's four scores per neuron (SFC_SI, SFC_Phi, MaxSI_SI,
MinFisherP_Phi), the neuron's status, and, from the per-node tables step 08 saves:

- MinFisherP_Phi_first: the Phi of the first node in table order with the lowest
  fisher_p (pandas idxmin). Step 08 instead takes the first row after an unstable sort
  on fisher_p. The two can differ only when nodes that give different cuts tie for the
  lowest p, which happens mostly where Fisher's p underflows to 0 on well-separated
  neurons;
- MaxPhi: the highest |Phi| over the candidate nodes, a fourth split method (step
  08 does not compute MaxPhi, so comparing it checks this reimplementation, not step 08);
- the number of candidate nodes, of nodes with fisher_p == 0, of nodes tied for the
  lowest p, and of nodes tied for the highest SI.

A neuron's status is "result" (step 08 wrote its scores; a score can still be empty, for
example the SFC SI when every synapse of the SFC split is an input), "failed" (listed in
step 08's failures table and no scores), "result, also listed as failed" (step 08 can do
both when an error comes after the scores were written) or "missing" (neither).

With --compare REF.ftr (one or more), every score column the reference shares with this
run (SFC_SI, also accepted as original_SI; SFC_Phi; MaxSI_SI; MinFisherP_Phi; MaxPhi) is
compared on neuron_id, which must be an integer column without duplicates. A value counts
as different when the absolute difference is above --tol or when it is empty in one table
only. A reference MinFisherP_Phi is also compared with MinFisherP_Phi_first, for
information. These count as problems: a score that differs; a neuron with a result that
the reference lacks; a neuron that failed here but has a reference row; a missing neuron.
A neuron that failed here and is also absent from the reference agrees with it, and is
counted separately. A reference also fails if no neuron could be compared, or if it has a
score column this run has no values for (for example MaxPhi without per-node tables).
Every problem neuron goes to --out/differences_<n>_<reference name>.csv. The tool ends with
PASS or FAIL per reference (MinFisherP_Phi_first does not count towards it) and exits with
status 1 if any reference fails, or if step 08 itself stops.

Usage:
    python tools/check_split_methods_on_subset.py --out outputs/split_methods_check/run1
    python tools/check_split_methods_on_subset.py --out outputs/split_methods_check/run1 \\
        --compare data/derived/SI_comparisons.ftr
    python tools/check_split_methods_on_subset.py --out outputs/split_methods_check/run1 \\
        --compare-only --compare data/derived/SI_comparisons.ftr other_summary.ftr

Options: --ids FILE takes neuron IDs from a text file, one integer per line, instead of the
skeleton folder; --limit N (N >= 1) keeps the first N of the chosen neurons, in ascending
ID order; --resume continues an interrupted run in the same --out (step 08 skips batches
already written). --resume refuses if the neuron list, step 08, methods/methods_all.py,
config.py, this tool, or the raw synapse table or neuron table (path, size, date) changed
since the run started. It cannot tell whether a batch file was cut short when the run was
stopped; if in doubt, delete the last batch's three files first. A run whose tables hold a
neuron twice stops with a message, since that means two runs were mixed. Step 08 needs SYNAPSE_TABLE_RAW_FTR
(processing/build_raw_synapse_table.py) and the neuron table. Time and memory are given
in the README.
"""
import argparse
import hashlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from config import NEURON_TABLE_FTR, SWC_DIR, SYNAPSE_TABLE_RAW_FTR  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.feather as feather  # noqa: E402
import pyarrow.ipc as ipc  # noqa: E402

STEP08 = REPO_ROOT / "processing" / "08_alternative_split_methods.py"
INPUT_FILES = {"step 08": STEP08, "methods_all.py": REPO_ROOT / "methods" / "methods_all.py",
               "config.py": REPO_ROOT / "config.py", "this tool": Path(__file__).resolve()}
INTRINSIC = ["central", "optic", "visual_centrifugal", "visual_projection"]
SCORE_COLUMNS = ["SFC_SI", "SFC_Phi", "MaxSI_SI", "MinFisherP_Phi", "MaxPhi"]
NODE_COLUMNS = ["neuron_id", "side_1_pre", "side_1_post", "side_2_pre", "side_2_post", "SI", "fisher_p"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive_int(text):
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return n


def read_ids(path):
    ids = set()
    for i, line in enumerate(Path(path).read_text().splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        if not text.isdigit() or int(text) > np.iinfo(np.int64).max:
            sys.exit(f"{path}, line {i}: '{text}' is not a neuron ID (one 64-bit integer per line)")
        ids.add(np.int64(text))
    return ids


def choose_neurons(ids_file, limit):
    """Intrinsic neurons (step 08's own filter) that have a skeleton, or those listed in a file."""
    table = pd.read_feather(NEURON_TABLE_FTR, columns=["neuron", "super_class"])
    intrinsic = set(table.loc[table["super_class"].isin(INTRINSIC), "neuron"].astype(np.int64))
    if ids_file:
        wanted = read_ids(ids_file)
        source = f"file {ids_file}"
    else:
        wanted = {np.int64(p.stem) for p in SWC_DIR.glob("*/*.swc") if p.stem.isdigit()}
        source = f"skeletons under {SWC_DIR}"
    chosen = sorted(wanted & intrinsic)
    print(f"{len(wanted):,} neurons from {source}; {len(chosen):,} of them are intrinsic "
          f"(super_class {', '.join(INTRINSIC)})")
    if limit is not None:
        chosen = chosen[:limit]
        print(f"--limit {limit}: keeping the first {len(chosen):,} by ID")
    if not chosen:
        sys.exit("No neurons to run.")
    return chosen


def input_fingerprint():
    """What a resumed run must share with the run it continues: the code (step 08, the shared
    helpers, config.py and this tool, which holds the patches) and the two tables it reads."""
    lines = [f"input {name} sha256: {sha256(path)}" for name, path in INPUT_FILES.items()]
    for name, table in (("synapse table", SYNAPSE_TABLE_RAW_FTR), ("neuron table", NEURON_TABLE_FTR)):
        if not table.exists():
            sys.exit(f"{table} not found; step 08 needs it (see the README).")
        stat = table.stat()
        lines.append(f"input {name}: {table} size {stat.st_size} modified {stat.st_mtime_ns}")
    return lines


def patched_step08(out, ids_path):
    src = STEP08.read_text()
    patches = [
        ("repository root",
         "sys.path.insert(0, str(Path(__file__).parent.parent))",
         f"sys.path.insert(0, {str(REPO_ROOT)!r})"),
        ("output paths",
         "from config import (METHODS_DIR, NEURON_TABLE_FTR, SYNAPSE_TABLE_RAW_FTR,\n"
         "                    ALT_SPLIT_BATCH_DIR, SI_COMPARISONS_FTR)",
         "from config import (METHODS_DIR, NEURON_TABLE_FTR, SYNAPSE_TABLE_RAW_FTR,\n"
         "                    ALT_SPLIT_BATCH_DIR, SI_COMPARISONS_FTR)\n"
         "# --- subset run (tools/check_split_methods_on_subset.py) ---\n"
         f"ALT_SPLIT_BATCH_DIR = Path({str(out / 'batches')!r})\n"
         f"SI_COMPARISONS_FTR = Path({str(out / 'SI_comparisons_subset.ftr')!r})\n"
         f"_SUBSET_IDS = [int(x) for x in open({str(ids_path)!r}).read().split()]"),
        ("neuron subset",
         "nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_centrifugal','visual_projection'])]",
         "nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_centrifugal','visual_projection'])]\n"
         "nodesG = nodesG[nodesG['neuron'].astype('int64').isin(_SUBSET_IDS)]  # subset run"),
        ("synapse read",
         "allsynapses = pd.read_feather(SYNAPSE_TABLE_RAW_FTR)",
         "import pyarrow.dataset as _ds  # subset run: read only the chosen neurons' synapses\n"
         "_dataset = _ds.dataset(str(SYNAPSE_TABLE_RAW_FTR), format='feather')\n"
         "allsynapses = _dataset.to_table(filter=_ds.field('pre').isin(_SUBSET_IDS)\n"
         "                                | _ds.field('post').isin(_SUBSET_IDS)).to_pandas()"),
        ("combine range",
         "Y = 236    # last batch will be X + Y",
         "Y = n_batches - 1    # subset run: every batch written"),
    ]
    for name, old, new in patches:
        n = src.count(old)
        if n != 1:
            sys.exit(f"Cannot patch step 08 ({name}): the text to replace occurs {n} times, "
                     "expected once. Step 08 has changed; update this tool.")
        src = src.replace(old, new)
    return src


def check_resume(out, chosen, fingerprint):
    ids_path, provenance = out / "neuron_ids.txt", out / "provenance.txt"
    if not ids_path.exists() or not provenance.exists():
        sys.exit(f"--resume: {out} has no neuron_ids.txt or provenance.txt, so it is not a run "
                 "this tool started; use a new --out.")
    if ids_path.read_text().split() != [str(i) for i in chosen]:
        sys.exit("--resume: the chosen neurons differ from the ones in neuron_ids.txt.")
    recorded = [ln for ln in provenance.read_text().splitlines() if ln.startswith("input ")]
    first_run = recorded[:len(fingerprint)]
    changed = [now for now, then in zip(fingerprint, first_run) if now != then]
    if len(first_run) != len(fingerprint) or changed:
        sys.exit("--resume: an input changed since this run started, so the batches would mix two "
                 "versions; use a new --out." + "".join(f"\n  now: {c}" for c in changed))


def failure_summary(out):
    """Print what step 08's failures tables say, if anything."""
    frames = [pd.read_feather(p) for p in sorted((out / "batches").glob("failures_batch_*.ftr"))]
    frames = [f for f in frames if len(f) and "error" in f.columns]
    if not frames:
        return
    fails = pd.concat(frames, ignore_index=True)
    print(f"{len(fails):,} neurons listed as failed by step 08; most common errors:")
    for error, n in fails["error"].astype(str).str[:100].value_counts().head(5).items():
        print(f"  {n:>6,}  {error}")


def run(args, out):
    if out.exists() and any(out.iterdir()) and not args.resume:
        sys.exit(f"{out} is not empty. Use a new folder, or --resume to continue a run there.")
    chosen = choose_neurons(args.ids, args.limit)
    fingerprint = input_fingerprint()
    if args.resume and out.exists() and any(out.iterdir()):
        check_resume(out, chosen, fingerprint)
    out.mkdir(parents=True, exist_ok=True)
    ids_path = out / "neuron_ids.txt"
    ids_path.write_text("\n".join(str(i) for i in chosen) + "\n")

    script = out / "patched_step08.py"
    script.write_text(patched_step08(out, ids_path))

    def git(*a):
        try:
            return subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True,
                                  text=True).stdout.strip()
        except FileNotFoundError:
            return "unknown (git not found)"
    dirty = git("status", "--porcelain", "--", str(STEP08), "config.py", "methods")
    dirty = dirty and not dirty.startswith("unknown")
    with open(out / "provenance.txt", "a") as f:
        f.write("\n".join([f"{'resumed' if args.resume else 'started'}: "
                           f"{datetime.now().isoformat(timespec='seconds')}",
                           f"git commit: {git('rev-parse', '--short', 'HEAD')}"
                           f"{' (step 08, config or methods modified)' if dirty else ''}",
                           *fingerprint,
                           f"neurons: {len(chosen)}",
                           f"python: {sys.executable}; pandas {pd.__version__}; numpy {np.__version__}"])
                + "\n")

    env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    print(f"Running step 08 on {len(chosen):,} neurons; log: {out / 'run.log'}")
    with open(out / "run.log", "a") as log:
        proc = subprocess.Popen([sys.executable, str(script)], cwd=out, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line)
            if line.startswith(("Batch", "Saved batch", "Total neurons", "Finished")):
                print(line, end="", flush=True)
        status = proc.wait()
    if status != 0:
        print(f"Step 08 stopped with status {status}; see {out / 'run.log'}.")
        failure_summary(out)
        sys.exit(1)


def absolute_phi(a, b, c, d):
    """|Phi| of the 2x2 table side x (pre, post); NaN when a margin is empty (as step 08)."""
    a, b, c, d = (np.asarray(x, dtype=float) for x in (a, b, c, d))
    den = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, np.abs((a * d - b * c) / den), np.nan)


def node_summaries(batches):
    """Per-neuron values from the per-node tables, one batch file at a time, reading only the
    columns needed. Step 08 writes an empty table for a batch in which every neuron failed.
    A neuron must appear in one batch only; the caller checks."""
    rows = []
    for path in sorted(batches.glob("nodes_batch_*.ftr")):
        schema = ipc.open_file(path).schema
        if not set(NODE_COLUMNS) <= set(schema.names):
            continue                                  # an empty table: every neuron of the batch failed
        nodes = feather.read_table(path, columns=NODE_COLUMNS).to_pandas()
        if not len(nodes):
            continue
        nodes["neuron_id"] = nodes["neuron_id"].astype(np.int64)
        nodes["Phi"] = absolute_phi(nodes["side_1_pre"], nodes["side_1_post"],
                                    nodes["side_2_pre"], nodes["side_2_post"])
        for nid, g in nodes.groupby("neuron_id", sort=False):
            r = {"neuron_id": nid, "n_candidates": len(g)}
            p = g["fisher_p"].dropna()
            if len(p):
                r["n_fisher_p_zero"] = int((p == 0).sum())
                r["n_tied_lowest_p"] = int((p == p.min()).sum())
                r["MinFisherP_Phi_first"] = g.loc[p.idxmin(), "Phi"]
            si = g["SI"].dropna()
            if len(si):
                r["n_tied_highest_SI"] = int((si == si.max()).sum())
            phi = g["Phi"].dropna()
            if len(phi):
                r["MaxPhi"] = phi.max()
            rows.append(r)
        del nodes
    return pd.DataFrame(rows)


def per_neuron(out):
    batches = out / "batches"
    chosen = [np.int64(x) for x in (out / "neuron_ids.txt").read_text().split()]
    table = pd.DataFrame({"neuron_id": pd.Series(chosen, dtype=np.int64)})

    scores_path = out / "SI_comparisons_subset.ftr"
    if scores_path.exists():
        ours = pd.read_feather(scores_path)
        ours["neuron_id"] = ours["neuron_id"].astype(np.int64)
    else:
        print(f"No {scores_path.name}: step 08 wrote no combined table.")
        ours = pd.DataFrame({"neuron_id": pd.Series([], dtype=np.int64)})
    if ours["neuron_id"].duplicated().any():
        sys.exit(f"{scores_path.name} holds {int(ours['neuron_id'].duplicated().sum()):,} neurons twice; the "
                 "run mixes two runs (for example a --resume after the neuron table changed). Use a new --out.")
    has_result = table["neuron_id"].isin(ours["neuron_id"])
    table = table.merge(ours, on="neuron_id", how="left")
    for c in ("SFC_SI", "SFC_Phi", "MaxSI_SI", "MinFisherP_Phi"):
        if c not in table:
            table[c] = np.nan

    extra = node_summaries(batches)
    if len(extra) and extra["neuron_id"].duplicated().any():
        sys.exit(f"the per-node tables hold {int(extra['neuron_id'].duplicated().sum()):,} neurons in more "
                 "than one batch; the run mixes two runs. Use a new --out.")
    if len(extra):
        table = table.merge(extra, on="neuron_id", how="left")

    frames = [pd.read_feather(p) for p in sorted(batches.glob("failures_batch_*.ftr"))]
    frames = [f for f in frames if len(f) and "neuron_id" in f.columns]
    failures = (pd.concat(frames, ignore_index=True) if frames
                else pd.DataFrame({"neuron_id": pd.Series([], dtype=np.int64), "error": []}))
    failures["neuron_id"] = failures["neuron_id"].astype(np.int64)
    table = table.merge(failures[["neuron_id", "error"]].drop_duplicates("neuron_id"),
                        on="neuron_id", how="left")
    listed_failed = table["error"].notna()
    table["status"] = np.select(
        [has_result & listed_failed, has_result, listed_failed],
        ["result, also listed as failed", "result", "failed"], default="missing")

    counts = table["status"].value_counts()
    print(f"\n{len(chosen):,} neurons chosen: " + ", ".join(
        f"{counts.get(s, 0):,} {s}" for s in ["result", "result, also listed as failed", "failed", "missing"]))
    print(f"With a result but an empty SFC_SI: "
          f"{int((table['status'].str.startswith('result') & table['SFC_SI'].isna()).sum()):,}")
    if "MinFisherP_Phi_first" in table:
        tied = table["n_tied_lowest_p"].fillna(0) > 1
        print(f"Lowest Fisher p shared by more than one node: {int(tied.sum()):,} neurons "
              f"({int((table['n_fisher_p_zero'].fillna(0) > 0).sum()):,} with at least one p == 0).")
        a, b = table["MinFisherP_Phi"], table["MinFisherP_Phi_first"]
        differs = table["status"].str.startswith("result") & (((a - b).abs() > 1e-9) | (a.isna() != b.isna()))
        print(f"The two MinFisherP choices give a different Phi for {int(differs.sum()):,} neurons.")
    below = table["MaxSI_SI"] < table["SFC_SI"] - 1e-12
    print(f"MaxSI_SI below SFC_SI: {int(below.sum()):,} neurons.")
    table.to_csv(out / "per_neuron.csv", index=False)
    return table


def compare(table, refs, tol, out):
    """Compare with each reference; return True if every reference passes."""
    all_ok = True
    for k, ref_path in enumerate(refs, start=1):
        ref = pd.read_feather(ref_path)
        if "original_SI" in ref.columns and "SFC_SI" not in ref.columns:
            ref = ref.rename(columns={"original_SI": "SFC_SI"})
        reason = None
        if "neuron_id" not in ref:
            reason = "no neuron_id column"
        elif not pd.api.types.is_integer_dtype(ref["neuron_id"]):
            reason = (f"neuron_id is {ref['neuron_id'].dtype}, not an integer column "
                      "(18-digit IDs do not survive a float)")
        elif ref["neuron_id"].isna().any():
            reason = f"neuron_id has {int(ref['neuron_id'].isna().sum()):,} empty values"
        elif ref["neuron_id"].duplicated().any():
            reason = f"neuron_id has {int(ref['neuron_id'].duplicated().sum()):,} duplicates"
        if reason:
            print(f"\n{ref_path}: {reason}; FAIL.")
            all_ok = False
            continue
        ref["neuron_id"] = ref["neuron_id"].astype(np.int64)
        m = table.merge(ref, on="neuron_id", how="left", suffixes=("", "_ref"), indicator=True)
        in_ref = m["_merge"] == "both"
        status = m["status"]
        problems = {}
        for nid in m.loc[status.str.startswith("result") & ~in_ref, "neuron_id"]:
            problems.setdefault(nid, []).append("result here, not in reference")
        for nid in m.loc[(status == "failed") & in_ref, "neuron_id"]:
            problems.setdefault(nid, []).append("failed here, in reference")
        for nid in m.loc[status == "missing", "neuron_id"]:
            problems.setdefault(nid, []).append("no result and not listed as failed")
        both_failed = int(((status == "failed") & ~in_ref).sum())
        compared = m[in_ref & status.str.startswith("result")]
        print(f"\nAgainst {ref_path}: {int(in_ref.sum()):,} of the {len(table):,} chosen neurons are in it; "
              f"{len(compared):,} with a result here are compared; {both_failed:,} failed here and are "
              "absent from it too (agree).")
        print(f"  {'column':<50}{'compared':>9}{'equal':>9}{'differ':>9}{'max |diff|':>13}")
        pairs = [(c, c, True) for c in SCORE_COLUMNS if c in ref.columns and c in table.columns]
        lacking = [c for c in SCORE_COLUMNS if c in ref.columns and c not in table.columns]
        for c in lacking:
            print(f"  {c} is in the reference but this run has no values for it (no per-node tables?)")
        if "MinFisherP_Phi" in ref.columns and "MinFisherP_Phi_first" in table.columns:
            pairs.append(("MinFisherP_Phi_first", "MinFisherP_Phi", False))
        if not pairs:
            print("  no score column in common; FAIL.")
            all_ok = False
        for ours_col, ref_col, counts in pairs:
            ref_name = ref_col + "_ref" if ref_col in table.columns else ref_col
            a, b = compared[ours_col].astype(float), compared[ref_name].astype(float)
            d = (a - b).abs()
            bad = ~(d <= tol) & ~(a.isna() & b.isna())       # empty in one table only, or too far apart
            label = ours_col if ours_col == ref_col else f"{ours_col} vs {ref_col} (info)"
            print(f"  {label:<50}{len(compared):>9,}{int((~bad).sum()):>9,}{int(bad.sum()):>9,}"
                  f"{(d.max() if d.notna().any() else float('nan')):>13.3g}")
            if counts:
                for nid in compared.loc[bad, "neuron_id"]:
                    problems.setdefault(nid, []).append(ours_col)
        if len(compared) == 0:
            print("  no neuron could be compared")
        ok = not problems and bool(pairs) and not lacking and len(compared) > 0
        all_ok &= ok
        csv = out / f"differences_{k}_{Path(ref_path).stem}.csv"
        pd.DataFrame([{"neuron_id": n, "problems": "; ".join(v)} for n, v in problems.items()],
                     columns=["neuron_id", "problems"]).to_csv(csv, index=False)
        print(f"  {'PASS' if ok else 'FAIL'}: {len(problems):,} neurons with a problem ({csv.name})")
    return all_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="output folder (new or empty)")
    ap.add_argument("--ids", help="text file of neuron IDs to run, one integer per line")
    ap.add_argument("--limit", type=positive_int, help="keep only the first N neurons, by ID (N >= 1)")
    ap.add_argument("--resume", action="store_true", help="continue a run in a non-empty --out")
    ap.add_argument("--compare", nargs="+", default=[], metavar="REF.ftr",
                    help="reference tables to compare with, by neuron_id")
    ap.add_argument("--compare-only", action="store_true",
                    help="do not run step 08; analyse the run already in --out")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="largest absolute difference counted as equal (default 1e-9)")
    args = ap.parse_args()
    out = args.out.resolve()
    if not args.compare_only:
        run(args, out)
    else:
        for name in ("neuron_ids.txt", "batches"):
            if not (out / name).exists():
                sys.exit(f"--compare-only: no {name} in {out}; is it a run this tool made?")
    table = per_neuron(out)
    if args.compare and not compare(table, args.compare, args.tol, out):
        sys.exit(1)


if __name__ == "__main__":
    main()
