"""Check the Fig 7 MSB cell-type exclusion list (MSB_EXCLUDED_TYPES_CSV) against the bouton tables.

The list has one row per excluded cell type, with its kind: pre_clustering (left out before the boutons were
clustered), rms_inspection (left out after clustering, by inspection) or uniglomerular (left out before
clustering, cell by cell). The first two
kinds remove every cell of the type. The uniglomerular kind removes a cell only if its own sub_class in
NEURON_DATA_FTR is "uniglomerular"; the rows of that kind name the types such cells belong to, so a type listed there
can keep cells of another sub_class (M_lvPNm46 and M_vPNml80 do in the bouton tables).

The check:
  1. the list: its columns, the three kinds and nothing else, no type listed twice, a whole number of cells per row,
     and every type name present as a primary_type in the Codex cell-type file, CONSOLIDATED_CELL_TYPES_CSV (whose
     names the bouton tables use; a name typed with an en dash, or misspelt, matches no cell and would exclude nothing);
  2. BOUTON_TABLE_FTR and BOUTON_FILOPODIA_TABLE_FTR: no cell of a pre_clustering or rms_inspection type, and no cell
     whose sub_class is uniglomerular. The listed uniglomerular types that still have cells there are printed, with
     those cells' sub_class, as a note;
  3. with --unfiltered and --unfiltered-bwf (the two bouton tables before any type was removed): the rule above,
     applied to them, must give the two filtered tables exactly, row for row and value for value, and every
     uniglomerular cell it removes must be of a type the list gives as uniglomerular.

--mutants (needs the two unfiltered tables) runs the same check on wrong lists and requires each to FAIL: one
rms_inspection type left out, a type the tables keep added, R1-6 written with an en dash, a type given the wrong kind,
a kind spelt differently.

It does not test whether the list is the paper's Table 4 or why a type was excluded, and the unfiltered tables are not
deposited. Run from the repository root (the unfiltered tables wherever they are kept):
    .venv/bin/python tools/check_msb_exclusion.py
    .venv/bin/python tools/check_msb_exclusion.py --unfiltered B_FULL.ftr --unfiltered-bwf BWF_FULL.ftr
    .venv/bin/python tools/check_msb_exclusion.py --unfiltered B_FULL.ftr --unfiltered-bwf BWF_FULL.ftr --mutants
Writes outputs/tools/check_msb_exclusion/report.txt (mutants_report.txt with --mutants). Exit code 0 on PASS, 1 on FAIL.
"""
import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import config  # noqa: E402

OUT_DIR = config.OUTPUT_DIR / "tools" / "check_msb_exclusion"
COLUMNS = ["primary_type", "exclusion_kind", "reason", "n_cells_sheet"]
KINDS = {"pre_clustering", "rms_inspection", "uniglomerular"}
BY_TYPE = {"pre_clustering", "rms_inspection"}


def read_list(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_table(path):
    df = pd.read_feather(path)
    if df["neuron"].isna().any():
        raise SystemExit(f"{path.name}: {int(df['neuron'].isna().sum())} rows have no neuron")
    return df


def neuron_ids(df):
    return df["neuron"].astype(np.int64)


def run(lst, tables, nd, names, unfiltered, say):
    """Every check on one list; returns the reasons it fails (empty on PASS)."""
    fails = []
    # 1. the list
    if list(lst.columns) != COLUMNS:
        fails.append(f"list: columns {list(lst.columns)}, expected {COLUMNS}")
        return fails
    bad_kind = sorted(set(lst.exclusion_kind) - KINDS)
    if bad_kind:
        fails.append(f"list: kinds not allowed {bad_kind}")
    dup = sorted(lst.primary_type[lst.primary_type.duplicated()])
    if dup:
        fails.append(f"list: types listed twice {dup}")
    if not lst.n_cells_sheet.str.fullmatch(r"\d+").all():
        fails.append("list: n_cells_sheet not a whole number in some rows")
    unknown = sorted(set(lst.primary_type) - names)
    if unknown:
        fails.append(f"list: types not in CONSOLIDATED_CELL_TYPES_CSV {unknown}")
    by_type = set(lst.primary_type[lst.exclusion_kind.isin(BY_TYPE)])
    uni_listed = set(lst.primary_type[lst.exclusion_kind == "uniglomerular"])
    uni_cells = set(nd.root_id[nd.sub_class == "uniglomerular"])
    say(f"list: {len(lst)} rows; " + ", ".join(f"{k} {int((lst.exclusion_kind == k).sum())}" for k in sorted(KINDS)))

    # 2. the filtered tables
    for name, df in tables.items():
        ids = neuron_ids(df)
        cells = ids.nunique()
        n_type = ids[df.primary_type.isin(by_type)].nunique()
        n_uni = ids[ids.isin(uni_cells)].nunique()
        say(f"{name}: {len(df):,} rows, {cells:,} cells; cells of a pre_clustering or rms_inspection type {n_type}, "
            f"cells of sub_class uniglomerular {n_uni}")
        if n_type:
            fails.append(f"{name}: {n_type} cells of a pre_clustering or rms_inspection type, types "
                         f"{sorted(set(df.primary_type[df.primary_type.isin(by_type)]))}")
        if n_uni:
            fails.append(f"{name}: {n_uni} cells of sub_class uniglomerular")
        kept = df.loc[df.primary_type.isin(uni_listed), ["primary_type"]].assign(neuron=ids).drop_duplicates("neuron")
        kept = kept.merge(nd.rename(columns={"root_id": "neuron", "primary_type": "nd_type"}), on="neuron", how="left")
        say(f"  note: listed uniglomerular types with cells here: {kept.groupby('primary_type').size().to_dict()}; "
            f"those cells' sub_class: {kept.sub_class.value_counts(dropna=False).to_dict()}")

    # 3. the unfiltered tables, with the rule applied, against the filtered ones
    for name, full in (unfiltered or {}).items():
        ids = neuron_ids(full)
        drop = full.primary_type.isin(by_type) | ids.isin(uni_cells)
        kept = full[~drop].reset_index(drop=True)
        filt = tables[name].reset_index(drop=True)
        say(f"{name} unfiltered: {ids.nunique():,} cells; the rule drops {ids[drop].nunique():,} cells "
            f"({int(drop.sum()):,} rows), the filtered table has {ids.nunique() - neuron_ids(filt).nunique():,} "
            f"fewer cells")
        try:
            pd.testing.assert_frame_equal(kept, filt, check_exact=True)
            say(f"  the rule gives the filtered table exactly: {len(kept):,} rows, {kept.shape[1]} columns")
        except AssertionError as e:
            sym = set(neuron_ids(kept)) ^ set(neuron_ids(filt))
            fails.append(f"{name}: the rule does not give the filtered table ({len(kept):,} rows against "
                         f"{len(filt):,}; {len(sym)} cells in one only): {str(e).splitlines()[0:3]}")
        uni_types = set(full.primary_type[ids.isin(uni_cells)].dropna())
        stray = sorted(uni_types - uni_listed)
        if stray:
            fails.append(f"{name}: uniglomerular cells of types not listed as uniglomerular {stray}")
    return fails


def mutants(lst, tables, unfiltered):
    by_rows = {}
    full_b = next(iter(unfiltered.values()))
    rms = lst[lst.exclusion_kind == "rms_inspection"].primary_type
    biggest_rms = full_b[full_b.primary_type.isin(rms)].primary_type.value_counts().index[0]
    kept_type = next(iter(tables.values())).primary_type.value_counts().index[0]
    m = lst[lst.primary_type != biggest_rms]
    by_rows[f"rms_inspection type {biggest_rms} left out"] = m
    by_rows[f"kept type {kept_type} added as rms_inspection"] = pd.concat(
        [lst, pd.DataFrame([[kept_type, "rms_inspection", "", "1"]], columns=COLUMNS)], ignore_index=True)
    m = lst.copy()
    m.loc[m.primary_type == "R1-6", "primary_type"] = "R1–6"
    by_rows["R1-6 written with an en dash"] = m
    m = lst.copy()
    m.loc[m.primary_type == biggest_rms, "exclusion_kind"] = "uniglomerular"
    by_rows[f"{biggest_rms} given the kind uniglomerular"] = m
    m = lst.copy()
    m.loc[m.exclusion_kind == "rms_inspection", "exclusion_kind"] = "RMS excluded"
    by_rows["rms_inspection spelt as the sheet's 'RMS excluded'"] = m
    return by_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--unfiltered", type=Path, help="the bouton table before any type was removed")
    ap.add_argument("--unfiltered-bwf", type=Path, help="the bouton-with-terminal table before any type was removed")
    ap.add_argument("--mutants", action="store_true", help="require wrong lists to FAIL (needs both unfiltered tables)")
    a = ap.parse_args()
    if bool(a.unfiltered) != bool(a.unfiltered_bwf):
        ap.error("give both --unfiltered and --unfiltered-bwf, or neither")
    if a.mutants and not a.unfiltered:
        ap.error("--mutants needs --unfiltered and --unfiltered-bwf")
    lines = []

    def say(s=""):
        print(s)
        lines.append(str(s))

    say(f"check_msb_exclusion {datetime.datetime.now():%Y-%m-%d %H:%M}; PIPELINE_RUN={config.PIPELINE_RUN}, "
        f"SPLIT_METHOD={config.SPLIT_METHOD}")
    say(f"list {config.MSB_EXCLUDED_TYPES_CSV.name}; tables {config.BOUTON_TABLE_FTR.name}, "
        f"{config.BOUTON_FILOPODIA_TABLE_FTR.name}; sub_class from {config.NEURON_DATA_FTR.name}; type names from "
        f"{config.CONSOLIDATED_CELL_TYPES_CSV.name}")
    lst = read_list(config.MSB_EXCLUDED_TYPES_CSV)
    nd = pd.read_feather(config.NEURON_DATA_FTR, columns=["root_id", "primary_type", "sub_class"])
    nd["root_id"] = nd.root_id.astype(np.int64)
    names = set(pd.read_csv(config.CONSOLIDATED_CELL_TYPES_CSV, usecols=["primary_type"]).primary_type.dropna())
    tables = {"BOUTON_TABLE_FTR": read_table(config.BOUTON_TABLE_FTR),
              "BOUTON_FILOPODIA_TABLE_FTR": read_table(config.BOUTON_FILOPODIA_TABLE_FTR)}
    unfiltered = None
    if a.unfiltered:
        unfiltered = {"BOUTON_TABLE_FTR": read_table(a.unfiltered),
                      "BOUTON_FILOPODIA_TABLE_FTR": read_table(a.unfiltered_bwf)}
        say(f"unfiltered tables {a.unfiltered.name}, {a.unfiltered_bwf.name}")
    else:
        say("no unfiltered tables given: part 3 not run")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not a.mutants:
        fails = run(lst, tables, nd, names, unfiltered, say)
        for f in fails:
            say(f"FAIL: {f}")
        say("PASS" if not fails else "FAIL")
        (OUT_DIR / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"written {OUT_DIR / 'report.txt'}")
        return 0 if not fails else 1

    ok = True
    base = run(lst, tables, nd, names, unfiltered, lambda s: None)
    say(f"the deposited list: {'PASS' if not base else 'FAIL ' + str(base)}")
    ok &= not base
    for label, m in mutants(lst, tables, unfiltered).items():
        fails = run(m, tables, nd, names, unfiltered, lambda s: None)
        say(f"mutant '{label}': {'FAIL (as required)' if fails else 'PASS (the check missed it)'}")
        for f in fails:
            say(f"    {f[:300]}")
        ok &= bool(fails)
    say("MUTANTS PASS: the list passes and every wrong list fails" if ok else "MUTANTS FAIL")
    (OUT_DIR / "mutants_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written {OUT_DIR / 'mutants_report.txt'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
