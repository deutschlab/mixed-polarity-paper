"""Check that processing/05 writes root_id as exact whole-number cell IDs.

Step 05 builds its cell list from the synapse table (every cell seen as pre or
post), left-merges it onto `nodesG`, the annotated cells that have an SI, and
keeps the rows where neuron == root_id. A synapse-table cell that is not in
`nodesG` (no annotation, or no SI after the dropna on SI) gets a missing root_id
in that merge, which turns the column into float64; a float64 holds only every
128th integer at the size of these IDs, so the IDs are rounded. Since 4 Oct the
line after the filter sets root_id from neuron.

This tool runs step 05's own lines for that part, taken from the file unchanged
and found by their exact text (a reworded line is reported as a failure):
  - the cell list (`unique_neurons = ...` and `result_df = pd.DataFrame(...)`),
  - the merge (`df=result_df.merge(nodesG, ...)`),
  - everything from the filter (`df=df[df['neuron']==df['root_id']]`) up to the
    first `df.to_feather(NEURON_TABLE_FTR)`, without the mkdir line before it.
`nodesG` is stood in by the annotated cells that have an SI in SI_UPDATED_FTR.
That is enough for root_id: its type depends only on which synapse-table cells
find no match. The real `nodesG` also has the cell_stats, linker and swc_data
columns merged in; those are not modelled, so rows they might duplicate would
not show here. The lines between the merge and the filter are skipped; they
compute other columns, merging on neuron.

Checks:
  - no live line between the merge and the filter mentions root_id, and no
    skipped live line assigns, renames or recasts columns in a way that could
    touch the key columns;
  - after the block, root_id is int64;
  - root_id equals neuron on every row (compared as exact Python numbers);
  - apart from root_id, the block leaves every column, row, the index and the
    column order as the filter line alone does;
  - the stand-in keeps exactly the delivered neuron table's cells, each once
    (NEURON_TABLE_FTR; this checks the stand-in, not the fix).

Usage:
    python tools/check_step05_root_id.py
    python tools/check_step05_root_id.py --script path/to/another_copy_of_05.py

Needs step 05's outputs (SI_updated.ftr and the neuron table), the annotation
file and the pre and post columns of the synapse table. Takes a few seconds with
about 5 GB of peak memory. Writes outputs/check/check_step05_root_id/report.txt.
Exits with status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import (NEURON_ANNOTATIONS_CSV, NEURON_TABLE_FTR, OUTPUT_DIR,  # noqa: E402
                    SI_UPDATED_FTR, SYNAPSE_TABLE_FTR)

STEP05 = REPO_ROOT / "processing" / "05_build_neuron_metadata_table.py"
OUT_DIR = OUTPUT_DIR / "check" / "check_step05_root_id"

CELL_LIST = ("unique_neurons = ", "result_df = pd.DataFrame(")
MERGE = "df=result_df.merge(nodesG"
FILTER = "df=df[df['neuron']==df['root_id']]"
MKDIR = "NEURON_TABLE_FTR.parent.mkdir"
SAVE = "df.to_feather(NEURON_TABLE_FTR)"
MAX_BLOCK = 10  # the save follows the filter within a few lines; more means the lookup went astray
TARGETS_FRAME = re.compile(r"^(result_)?df\b")  # a statement that writes df or result_df
# reassigning the key column, relabelling columns, or recasting the whole frame
KEY_RISK = re.compile(r"\[['\"]neuron['\"]\]\s*=|\.rename\(|\.columns\s*=|^(result_)?df\s*=\s*(result_)?df\.astype\(")


def dead_lines(source: str) -> set[int]:
    """Line numbers inside bare string statements (the blocks that never run)."""
    dead = set()
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            dead.update(range(node.lineno, node.end_lineno + 1))
    return dead


def find(lines: list[str], dead: set[int], start: str, after: int = 0) -> int:
    """1-based number of the first live line after `after` that starts with `start`."""
    for i, line in enumerate(lines, 1):
        if i > after and i not in dead and line.strip().startswith(start):
            return i
    raise LookupError(f"no live line starting with {start!r} after line {after}")


def code(line: str) -> str:
    return line.split("#")[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--script", type=Path, default=STEP05,
                        help="the copy of step 05 to test (default: the repository's)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "report.txt"
    report_path.unlink(missing_ok=True)
    out: list[str] = []
    results: list[tuple[bool, str]] = []

    def say(text: str = "") -> None:
        print(text)
        out.append(text)

    def finish() -> None:
        say()
        for ok, text in results:
            say(("PASS  " if ok else "FAIL  ") + text)
        say(f"{sum(ok for ok, _ in results)} passed, {sum(not ok for ok, _ in results)} failed")
        report_path.write_text("\n".join(out) + "\n")
        sys.exit(0 if results and all(ok for ok, _ in results) else 1)

    raw = args.script.read_bytes()
    source = raw.decode("utf-8-sig")
    lines = source.splitlines()
    dead = dead_lines(source)
    say(f"Checked {datetime.datetime.now():%Y-%m-%d %H:%M}")
    say(f"Step 05 code: {args.script} (sha256 {hashlib.sha256(raw).hexdigest()[:16]})")
    say()

    try:
        a = find(lines, dead, CELL_LIST[0])
        b = find(lines, dead, CELL_LIST[1], a)
        m = find(lines, dead, MERGE, b)
        f = find(lines, dead, FILTER, m)
        s = find(lines, dead, SAVE, f)
    except LookupError as error:
        results.append((False, f"step 05's lines were found: {error}"))
        finish()
    if s - f > MAX_BLOCK:
        results.append((False, f"the save follows the filter within {MAX_BLOCK} lines (it is {s - f} lines on)"))
        finish()
    block_lines = [i for i in range(f, s) if not lines[i - 1].strip().startswith(MKDIR)]
    block = "\n".join(lines[i - 1] for i in block_lines)
    say(f"lines run: {a}, {b} (cell list), {m} (merge), {f}-{s - 1} without the mkdir (filter up to the save at {s})")
    for i in block_lines:
        if i not in dead and code(lines[i - 1]):
            say(f"   {i}: {lines[i - 1].strip()}")

    between = [i for i in range(m + 1, f) if i not in dead and "root_id" in code(lines[i - 1])]
    risky = [i for i in list(range(b + 1, m)) + list(range(m + 1, f))
             if i not in dead and TARGETS_FRAME.search(code(lines[i - 1]))
             and KEY_RISK.search(code(lines[i - 1]))]
    results.append((not between and not risky,
                    "no skipped live line mentions root_id or renames, recasts or reassigns the key columns"
                    + (f" (mentions root_id: {between})" if between else "")
                    + (f" (renames or recasts: {risky})" if risky else "")))

    syn = feather.read_table(SYNAPSE_TABLE_FTR, columns=["pre", "post"], memory_map=True)
    allsynapses = syn.to_pandas()
    del syn
    annot = pd.read_csv(NEURON_ANNOTATIONS_CSV, usecols=["root_id"])
    si = pd.read_feather(SI_UPDATED_FTR, columns=["root_id", "SI"])
    nodesG = annot.merge(si, on="root_id", how="left").dropna(subset=["SI"])

    ns = {"pd": pd, "np": np, "allsynapses": allsynapses, "nodesG": nodesG}
    exec(lines[a - 1].strip() + "\n" + lines[b - 1].strip(), ns)
    exec(lines[m - 1].strip(), ns)
    unmatched = int(ns["df"]["root_id"].isna().sum())
    say(f"cells in the synapse table: {len(ns['result_df']):,}; with no match in the merge: {unmatched:,} "
        f"(root_id {ns['df']['root_id'].dtype} after the merge)")

    base_ns = dict(ns)
    base_ns["df"] = ns["df"].copy()
    exec(lines[f - 1].strip(), base_ns)
    base = base_ns["df"]
    exec(block, ns)
    df = ns["df"]

    results.append((df["root_id"].dtype == np.int64, f"root_id is int64 after the block (it is {df['root_id'].dtype})"))
    exact = int((df["root_id"].to_numpy(dtype=object) == df["neuron"].to_numpy(dtype=object)).sum())
    results.append((exact == len(df), f"root_id equals neuron on every row ({exact:,} of {len(df):,})"))
    same_rest = (list(df.columns) == list(base.columns) and df.index.equals(base.index)
                 and df.drop(columns="root_id").equals(base.drop(columns="root_id")))
    results.append((same_rest, "apart from root_id, the block leaves every column, row, the index and the "
                    "column order as the filter line alone does"))
    delivered = pd.read_feather(NEURON_TABLE_FTR, columns=["neuron"])["neuron"].to_numpy()
    same_cells = len(df) == len(delivered) and np.array_equal(np.sort(df["neuron"].to_numpy()), np.sort(delivered))
    results.append((same_cells, f"the stand-in keeps exactly the delivered neuron table's {len(delivered):,} cells, "
                    f"each once ({len(df):,} rows here)"))
    finish()


if __name__ == "__main__":
    # No inputs needed: it runs the step 05 lines from
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/processing/05_build_neuron_metadata_table.py
    # on the tables named in config.py, on this machine
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/data/derived/synapses_783_article_princeton.ftr       (pre and post only)
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/data/raw/Supplemental_file1_neuron_annotations.csv      (root_id)
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/data/derived/SI_updated.ftr                            (root_id, SI)
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/data/derived/neuron_data_full_article_princeton.ftr    (neuron, the delivered cell set)
    # and writes its report to
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/outputs/check/check_step05_root_id/report.txt
    # To test a copy of step 05 from before the 4 Oct fix (made with
    #   git show 3cb03a6:processing/05_build_neuron_metadata_table.py > outputs/check/check_step05_root_id/step05_before_fix.py)
    # uncomment:
    # sys.argv[1:] = [
    #     "--script", "/Users/ohajyahia/PycharmProjects/mixed-polarity-paper/outputs/check/check_step05_root_id/step05_before_fix.py",
    # ]
    main()
