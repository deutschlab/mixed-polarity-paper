"""Check the Figure 7 input tables that config.py names against a recorded reference.

No script in this repository writes these tables (apart from NEURONS_NT_BWF_FTR, which
figures/fig7/fig7i_filopodia_fraction_x_nt.py writes), so a figure can only be trusted to come
from the same data if the files themselves are the same. For each table in TABLES below this
checks, against tools/fig7_tables_reference.json:
  - the file is there (a table in NOT_YET_SUPPLIED is reported NOT SUPPLIED until someone provides it,
    which is not a failure; every table was supplied to the maintainers by 18 Oct, and none is deposited);
  - its sha256 (for NEURONS_NT_BWF_FTR, which is regenerated, a hash of its content instead,
    since feather bytes can differ for the same table);
  - its columns and their types, and the number of rows;
  - each ID column is a 64-bit integer in the file, and the pandas dtype it reads back as
    (nullable Int64 is recorded where the file says so, and printed as a note, not a failure);
  - the key columns hold no duplicate;
  - the cross-table counts made by cross_counts() below (for example how many rows of the bouton table with a
    postsynaptic terminal have no bouton in the bouton table) are unchanged.
A table supplied for the first time has no reference: run --record to add it.

--record measures every table that is there and writes the reference (check the printed values
before committing it). --selftest shows the checks can fail: it writes altered copies of the
smaller tables to a temporary folder (a row dropped, one value changed, an ID column made float,
an ID column that reads back as nullable Int64, a column dropped, a key duplicated, the rows
reordered, the file missing) and requires each to FAIL through the check meant for it (a reason
line naming that check), so a check that stopped working cannot hide behind a changed checksum.

It does not test what the tables mean (how they were made, or whether a value is right), the
content of the authors' tables beyond their checksums, or tables a reader saves under other names.

Run:
    .venv/bin/python tools/check_fig7_tables.py
    .venv/bin/python tools/check_fig7_tables.py --selftest
    .venv/bin/python tools/check_fig7_tables.py --record
About a minute (it reads about 1.7 GB). Writes outputs/tools/check_fig7_tables/report.txt.
Exit code 0 on PASS, 1 on FAIL.
"""
import argparse
import datetime
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import config  # noqa: E402

OUT_DIR = config.OUTPUT_DIR / "tools" / "check_fig7_tables"
REFERENCE = REPO_ROOT / "tools" / "fig7_tables_reference.json"

# config constant -> (ID columns, key columns that must be unique)
TABLES = {
    "BOUTON_TABLE_FTR": (["neuron", "cluster"], ["neuron", "cluster"]),
    "BOUTON_FILOPODIA_TABLE_FTR": (["neuron", "bwf"], ["neuron", "bwf"]),
    "FILOPODIA_RECIPROCAL_FTR": (["pre", "post"], ["synapse_id"]),
    "FILOPODIA_FILTERED_FTR": (["pre"], []),
    "BOUTON_INERTIA_FTR": (["neuron"], ["neuron"]),
    "NEURON_DATA_FTR": (["root_id"], ["root_id"]),
    "SYNAPSES_NO_SELF_FTR": (["pre", "post"], []),
    "CONSOLIDATED_CELL_TYPES_CSV": (["root_id"], ["root_id"]),
    "SYN_BOUTON_FTR": (["synapse_id"], ["synapse_id"]),
    "SYN_BOUTON_NEW_FTR": (["synapse_id"], ["synapse_id"]),
    "NEURONS_NT_BWF_FTR": (["neuron"], ["neuron"]),
    "NEURON_TABLE_FTR": (["neuron", "root_id"], ["neuron"]),
}
CONTENT_HASH = {"NEURONS_NT_BWF_FTR"}
NOT_YET_SUPPLIED = set()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def content_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df.reset_index(drop=True), index=True).values.tobytes())
    h.update(json.dumps([list(map(str, df.columns)), [str(t) for t in df.dtypes]]).encode())
    return h.hexdigest()


def read_columns(path: Path, columns: list) -> pa.Table:
    if path.suffix == ".gz" or path.suffix == ".csv":
        return pa.Table.from_pandas(pd.read_csv(path, usecols=columns), preserve_index=False)
    return pf.read_table(path, columns=columns)


def measure(name: str, path: Path) -> dict:
    """Everything the reference records about one table."""
    ids, key = TABLES[name]
    m = {"file": path.name}
    if path.suffix in (".gz", ".csv"):
        df = pd.read_csv(path)
        m["columns"] = {c: str(t) for c, t in df.dtypes.items()}
        m["rows"] = len(df)
        m["pandas_id_dtypes"] = {c: str(df[c].dtype) for c in ids if c in df}
        m["file_id_types"] = {c: str(df[c].dtype) for c in ids if c in df}
    else:
        with pa.ipc.open_file(path) as r:
            schema = r.schema
        m["columns"] = {f.name: str(f.type) for f in schema}
        with pa.ipc.open_file(path) as r:
            m["rows"] = sum(r.get_batch(i).num_rows for i in range(r.num_record_batches))
        pandas_meta = json.loads((schema.metadata or {}).get(b"pandas", b"{}"))
        numpy_types = {c.get("name"): c.get("numpy_type") for c in pandas_meta.get("columns", [])}
        m["pandas_id_dtypes"] = {c: numpy_types.get(c, m["columns"].get(c)) for c in ids}
        m["file_id_types"] = {c: m["columns"].get(c) for c in ids}
    if name in CONTENT_HASH:
        m["content_hash"] = content_hash(pd.read_feather(path))
    else:
        m["sha256"] = file_sha256(path)
    if key and all(k in m["columns"] for k in key):
        t = read_columns(path, key)
        if len(key) == 1:
            m["duplicate_keys"] = int(t.num_rows - pc.count_distinct(t.column(0)).as_py())
        else:
            df = t.to_pandas()
            m["duplicate_keys"] = int(df.duplicated().sum())
    return m


def cross_counts(paths: dict) -> dict:
    """Counts across tables; each one names its counting rule."""
    out = {}
    have = {k for k, p in paths.items() if p.exists()}
    if {"BOUTON_TABLE_FTR", "BOUTON_FILOPODIA_TABLE_FTR"} <= have:
        b = pf.read_table(paths["BOUTON_TABLE_FTR"], columns=["neuron", "cluster"]).to_pandas()
        bwf = pf.read_table(paths["BOUTON_FILOPODIA_TABLE_FTR"], columns=["neuron", "bwf"]).to_pandas()
        o = b.astype("int64").merge(bwf.astype("int64").rename(columns={"bwf": "cluster"}), how="outer", indicator=True)
        out["bwf rows (neuron, bwf) with no (neuron, cluster) in the bouton table"] = int((o["_merge"] == "right_only").sum())
        out["bouton neurons (distinct neuron in the bouton table)"] = int(b["neuron"].nunique())
        if "NEURONS_NT_BWF_FTR" in have:
            nb = pf.read_table(paths["NEURONS_NT_BWF_FTR"], columns=["neuron"]).to_pandas()["neuron"].astype("int64")
            out["NEURONS_NT_BWF neurons not in the bouton table"] = int((~nb.isin(b["neuron"].astype("int64"))).sum())
        if "BOUTON_INERTIA_FTR" in have:
            ine = pf.read_table(paths["BOUTON_INERTIA_FTR"], columns=["neuron"]).to_pandas()["neuron"].astype("int64")
            out["inertia neurons not in the bouton table"] = int((~ine.isin(b["neuron"].astype("int64"))).sum())
    return out


def compare(name: str, ref: dict, got: dict) -> list:
    problems = []
    for field in ("sha256", "content_hash", "rows", "columns", "duplicate_keys", "pandas_id_dtypes", "file_id_types"):
        if field in ref and ref.get(field) != got.get(field):
            problems.append("%s: %s is %s, reference %s" % (name, field, str(got.get(field))[:120], str(ref.get(field))[:120]))
    for c, t in got.get("file_id_types", {}).items():
        if t not in ("int64",):
            problems.append("%s: ID column %s is %s in the file, not int64" % (name, c, t))
    if got.get("duplicate_keys"):
        problems.append("%s: %d duplicate keys %s" % (name, got["duplicate_keys"], TABLES[name][1]))
    return problems


def run_checks(paths: dict, reference: dict) -> tuple:
    lines, failed = [], 0
    for name in TABLES:
        path = paths[name]
        if not path.exists():
            if name in NOT_YET_SUPPLIED and name not in reference["tables"]:
                lines.append("NOT SUPPLIED %-28s %s" % (name, path.name))
            else:
                lines.append("FAIL     %-28s missing: %s" % (name, path))
                failed += 1
            continue
        if name not in reference["tables"]:
            lines.append("NO REF   %-28s %s is there but has no reference; run --record" % (name, path.name))
            failed += 1
            continue
        try:
            got = measure(name, path)
            problems = compare(name, reference["tables"][name], got)
        except Exception as e:
            problems = ["%s: could not be read: %r" % (name, e)]
        if problems:
            failed += 1
            lines.append("FAIL     %-28s %s" % (name, path.name))
            lines += ["           " + p for p in problems]
        else:
            notes = [c for c, t in got["pandas_id_dtypes"].items() if t not in ("int64",)]
            lines.append("PASS     %-28s %s: %d rows, %d columns%s" % (
                name, path.name, got["rows"], len(got["columns"]),
                "; note: %s read back as pandas %s" % (", ".join(notes), got["pandas_id_dtypes"][notes[0]]) if notes else ""))
    try:
        cross = cross_counts(paths)
        for k, v in reference.get("cross", {}).items():
            if cross.get(k) != v:
                lines.append("FAIL     cross: %s is %s, reference %s" % (k, cross.get(k), v))
                failed += 1
            else:
                lines.append("PASS     cross: %s = %d" % (k, v))
    except Exception as e:
        lines.append("FAIL     cross counts could not be made: %r" % e)
        failed += 1
    return lines, failed


def selftest(paths: dict, reference: dict) -> list:
    """Altered copies of the smaller tables; each must make the check FAIL."""
    lines, missed = [], 0
    nb = pd.read_feather(paths["NEURONS_NT_BWF_FTR"])
    ine = pd.read_feather(paths["BOUTON_INERTIA_FTR"])
    nd = pd.read_feather(paths["NEURON_DATA_FTR"])
    FRAC = next(c for c in nb.columns if c != "neuron" and pd.api.types.is_float_dtype(nb[c]))
    # (label, table, altered frame or None for a missing file, text the specific check's reason must contain)
    cases = [
        ("row dropped (generated table)", "NEURONS_NT_BWF_FTR", nb.iloc[1:].reset_index(drop=True), "rows is"),
        ("one value changed (generated table)", "NEURONS_NT_BWF_FTR",
         nb.assign(**{FRAC: nb[FRAC].where(nb.index != 0, nb[FRAC].iloc[0] + 0.001)}), "content_hash is"),
        ("ID column made float", "BOUTON_INERTIA_FTR", ine.assign(neuron=ine["neuron"].astype("float64")), "not int64"),
        ("ID column read back nullable", "BOUTON_INERTIA_FTR", ine.assign(neuron=ine["neuron"].astype("Int64")), "pandas_id_dtypes is"),
        ("column dropped", "BOUTON_INERTIA_FTR", ine.drop(columns=ine.columns[-1]), "columns is"),
        ("key duplicated", "NEURON_DATA_FTR", pd.concat([nd, nd.iloc[:1]], ignore_index=True), "duplicate keys"),
        ("same content, rows reordered", "BOUTON_INERTIA_FTR", ine.iloc[::-1].reset_index(drop=True), "sha256 is"),
        ("file missing", "BOUTON_INERTIA_FTR", None, "missing"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for label, name, df, must in cases:
            altered = dict(paths)
            target = Path(tmp) / (name + ".ftr")
            if df is not None:
                df.to_feather(target)
            altered[name] = target
            report, failed = run_checks(altered, reference)
            reasons = [x.strip() for x in report if x.startswith("           " + name)
                       or (x.startswith("FAIL") and name in x and "missing" in x)]
            caught = failed > 0 and any(must in r for r in reasons)
            missed += not caught
            lines.append("%-7s %-38s %s" % ("caught" if caught else "MISSED", label, " | ".join(reasons)[:300]))
            if target.exists():
                target.unlink()
    lines.append("PASS: all %d altered tables caught, each by its own check" % len(cases) if not missed
                 else "FAIL: %d of %d altered tables not caught by their own check" % (missed, len(cases)))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--record", action="store_true", help="measure the tables that are there and write the reference")
    parser.add_argument("--selftest", action="store_true", help="require altered copies of the tables to FAIL")
    args = parser.parse_args()
    paths = {name: Path(getattr(config, name)) for name in TABLES}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = OUT_DIR / ("selftest_report.txt" if args.selftest else "report.txt")
    if report.exists():
        report.unlink()
    head = ["Figure 7 table check, %s; reference %s" % (datetime.datetime.now().isoformat(timespec="seconds"), REFERENCE.name),
            "pandas %s, pyarrow %s" % (pd.__version__, pa.__version__),
            "tool sha256 %s" % file_sha256(Path(__file__))]
    if REFERENCE.exists():
        head.append("reference sha256 %s" % file_sha256(REFERENCE))
    try:
        if args.record:
            ref = {"recorded": datetime.date.today().isoformat(),
                   "tables": {n: measure(n, p) for n, p in paths.items() if p.exists()},
                   "cross": cross_counts(paths)}
            REFERENCE.write_text(json.dumps(ref, indent=1) + "\n", encoding="utf-8")
            lines = ["%s: %s" % (n, json.dumps(m)) for n, m in ref["tables"].items()]
            lines += ["cross: %s = %d" % kv for kv in ref["cross"].items()]
            lines.append("RECORDED %d tables into %s; check these values before committing it" % (len(ref["tables"]), REFERENCE))
            ok = True
        else:
            reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
            if args.selftest:
                lines = selftest(paths, reference)
                ok = lines[-1].startswith("PASS")
            else:
                lines, failed = run_checks(paths, reference)
                ok = not failed
                lines.append("PASS" if ok else "FAIL")
    except Exception as e:
        lines, ok = ["FAIL: the check stopped: %r" % e], False
    report.write_text("\n".join(head + lines) + "\n", encoding="utf-8")
    print("\n".join(head + lines))
    print("Report:", report)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    # No inputs needed: it reads the Figure 7 tables named in config.py:
    #   data/derived/b_table_full_prince(new)_filtered.ftr  (BOUTON_TABLE_FTR)
    #   data/derived/bwf_table_full_prince(new)_filtered.ftr  (BOUTON_FILOPODIA_TABLE_FTR)
    #   data/derived/filopodia_reciprocal.ftr  (FILOPODIA_RECIPROCAL_FTR)
    #   data/derived/inertia_all.ftr  (BOUTON_INERTIA_FTR)
    #   data/derived/neuron_data.ftr  (NEURON_DATA_FTR)
    #   data/derived/neurons_nt_bwf_frac.ftr  (NEURONS_NT_BWF_FTR, written by fig7i)
    #   data/raw/consolidated_cell_types.csv.gz  (CONSOLIDATED_CELL_TYPES_CSV)
    #   data/raw/syn_bouton_filopodia.ftr  (SYN_BOUTON_FTR)
    #   data/raw/syn_bouton_filopodia(new).ftr  (SYN_BOUTON_NEW_FTR)
    #   data/derived/filopodia_filtered.ftr  (FILOPODIA_FILTERED_FTR)
    #   data/derived/t2.ftr  (SYNAPSES_NO_SELF_FTR)
    # It compares them with
    #   tools/fig7_tables_reference.json  (recorded 18 Oct 2026)
    # and writes
    #   outputs/tools/check_fig7_tables/report.txt
    # To run the self-test instead, uncomment:
    # sys.argv[1:] = [
    #     "--selftest",
    # ]
    main()
