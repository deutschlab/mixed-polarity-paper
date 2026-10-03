"""Check the raw synapse tables that processing/build_raw_synapse_table.py writes.

For each version that exists (without self-synapses, SYNAPSE_TABLE_RAW_NO_SELF_FTR,
and with them, SYNAPSE_TABLE_RAW_WITH_SELF_FTR) it checks:

- the 11 columns, in order; synapse_id, pre and post int64, the coordinates and
  size integers;
- synapse_id strictly increasing (so unique, and still in the Codex CSV's row order),
  and, in the version with self-synapses, equal to the row number 0, 1, 2, ...;
- the number of self-synapses (none in the version without them);
- that every synapse of the processed synapse table (SYNAPSE_TABLE_FTR, from step 04)
  is in it, under the same synapse_id, with the same pre and post cell and the same
  six coordinates (which would catch the pre and post columns being swapped).

If both versions exist, it also checks that the version without self-synapses holds
exactly the synapse_ids of the other version whose pre and post differ. With --csv it
counts the Codex CSV's rows and checks that the version with self-synapses has one
row per CSV row.

Usage:
    python tools/check_raw_synapse_table.py
    python tools/check_raw_synapse_table.py --csv

Reads the columns through a memory map; it needs SYNAPSE_TABLE_FTR and about 10 GB of
free memory, and takes about a minute.
Exits with status 1 if a check fails or no version exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import (PRINCETON_SYNAPSE_CSV, SYNAPSE_TABLE_FTR,  # noqa: E402
                    SYNAPSE_TABLE_RAW_NO_SELF_FTR, SYNAPSE_TABLE_RAW_WITH_SELF_FTR)

COLUMNS = ["synapse_id", "pre_x", "pre_y", "pre_z", "post_x", "post_y", "post_z",
           "size", "pre", "post", "neuropil"]
ID_COLUMNS = ["synapse_id", "pre", "post"]
XYZ = ["pre_x", "pre_y", "pre_z", "post_x", "post_y", "post_z"]


def check_version(path: Path, ref, expect_no_self: bool) -> tuple[bool, np.ndarray | None]:
    """Check one raw table; return (passed, its synapse_ids)."""
    ok = True
    schema = feather.read_table(path, memory_map=True).schema
    if schema.names != COLUMNS:
        print(f"  columns differ: {schema.names}")
        return False, None
    for name in COLUMNS[:-1]:
        kind = schema.field(name).type
        wrong = kind != pa.int64() if name in ID_COLUMNS else not pa.types.is_integer(kind)
        if wrong:
            print(f"  {name} is {kind}, not {'int64' if name in ID_COLUMNS else 'an integer'}")
            ok = False
    table = feather.read_table(path, columns=ID_COLUMNS + XYZ, memory_map=True)
    ids = table["synapse_id"].to_numpy()
    pre, post = table["pre"].to_numpy(), table["post"].to_numpy()
    rows = len(ids)
    if rows > 1 and not np.all(np.diff(ids) > 0):
        print("  synapse_id is not strictly increasing")
        return False, None
    if not expect_no_self and not np.array_equal(ids, np.arange(rows)):
        print("  synapse_id is not the row number 0, 1, 2, ...")
        ok = False
    n_self = int(np.count_nonzero(pre == post))
    if expect_no_self and n_self:
        print(f"  {n_self:,} self-synapses in the version that should have none")
        ok = False
    pos = np.searchsorted(ids, ref["synapse_id"])
    found = pos < rows
    found[found] = ids[pos[found]] == ref["synapse_id"][found]
    missing = int(np.count_nonzero(~found))
    at = pos[found]
    other = np.zeros(int(found.sum()), dtype=bool)
    for name in ["pre", "post"] + XYZ:
        other |= table[name].to_numpy()[at] != ref[name][found]
    differing = int(np.count_nonzero(other))
    print(f"  {rows:,} rows, {n_self:,} self-synapses; of the {len(ref['synapse_id']):,} synapses in "
          f"{SYNAPSE_TABLE_FTR.name}: {missing:,} missing, {differing:,} with a different pre, post "
          "or coordinate")
    ok = ok and missing == 0 and differing == 0
    no_self_ids = ids[pre != post] if not expect_no_self else ids
    return ok, (ids, no_self_ids, rows, n_self)


def count_csv_rows(path: Path) -> int:
    with open(path, "rb") as handle:
        lines = sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1 << 24), b""))
    return lines - 1   # the header


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", action="store_true",
                        help="also count the Codex CSV's rows (reads the whole 7 GB file)")
    args = parser.parse_args()

    versions = [(SYNAPSE_TABLE_RAW_NO_SELF_FTR, True), (SYNAPSE_TABLE_RAW_WITH_SELF_FTR, False)]
    present = [(path, no_self) for path, no_self in versions if path.exists()]
    if not present:
        print("no raw synapse table found; run processing/build_raw_synapse_table.py first")
        sys.exit(1)

    ref_table = feather.read_table(SYNAPSE_TABLE_FTR, columns=ID_COLUMNS + XYZ, memory_map=True)
    ref = {name: ref_table[name].to_numpy() for name in ID_COLUMNS + XYZ}

    ok, counts = True, {}
    for path, no_self in present:
        print(path.name)
        passed, info = check_version(path, ref, no_self)
        ok = ok and passed
        if info is not None:
            counts[no_self] = info
    if True in counts and False in counts:
        same = np.array_equal(counts[True][0], counts[False][1])
        print(f"versions: the {counts[True][2]:,} synapse_ids without self-synapses "
              f"{'are exactly' if same else 'are NOT'} the {counts[False][2] - counts[False][3]:,} "
              "of the other version whose pre and post differ")
        ok = ok and same
    if args.csv and False in counts:
        csv_rows = count_csv_rows(PRINCETON_SYNAPSE_CSV)
        same = csv_rows == counts[False][2]
        print(f"Codex CSV: {csv_rows:,} rows, {'equal to' if same else 'NOT equal to'} the version "
              "with self-synapses")
        ok = ok and same
    print("all checks passed" if ok else "a check failed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    # No inputs needed: it reads the tables named in config.py, on this machine
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/data/derived/synapses_783_article_princeton_raw.ftr            (without self-synapses)
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/data/derived/synapses_783_article_princeton_raw_with_self.ftr  (with self-synapses)
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/data/derived/synapses_783_article_princeton.ftr                (processed, from step 04)
    # and prints its checks; it writes no file. To also count the rows of the Codex CSV
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/data/raw/fafb_v783_princeton_synapse_table.csv  (7 GB)
    # uncomment:
    # sys.argv[1:] = ["--csv"]
    main()
