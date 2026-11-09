# -*- coding: utf-8 -*-
"""
The two Fig 7 bouton tables on the MaxSI split, for the panels that read them (fig7e, fig7f, fig7g, fig7i).

The authors' per-synapse bouton table (CLUSTER_INFO_FILTERED_FTR) gives each presynaptic synapse its bouton (neuron =
the presynaptic cell, cluster = the bouton's number in that cell). Their filopodia table (FILOPODIA_FILTERED_PUBLISHED_FTR)
gives each postsynaptic synapse on a bouton the bouton it sits on (post = the bouton's cell, cluster = the bouton).
Neither uses the split. From these and a synapse table this builds:

b_table, one row per bouton (neuron, cluster):
    count       the bouton's rows whose synapse is in the synapse table; a bouton is kept if count >= 1
    comp        A if the presynaptic letter of every such synapse is A, D if every one is D, missing otherwise
    SI          the cell's SI_pre (the same on every synapse of a cell; the script stops if it is not)
    npil        the npil with its _L/_R ending removed, if every synapse gives the same one and none is missing
    filopodia   the bouton has at least one filopodia row whose synapse is in the synapse table
bwf_table, one row per bouton that has such a filopodia row:
    count       those filopodia rows (rows that repeat a synapse are counted each time, as in the published table)
    comp        A if the postsynaptic letter of every row is A, D if every one is D, missing otherwise
    npil        the npil with its ending kept, if every row gives the same one, then with the ending removed
                (so a bouton on SMP_L and SMP_R is missing here, but not in b_table)
    SI          the cell's SI in b_table
primary_type and super_class are copied per cell from the published b_table (BOUTON_TABLE_PUBLISHED_FTR); cells it
does not have take primary_type from CONSOLIDATED_CELL_TYPES_CSV and super_class from CLASSIFICATION_CSV.

First a check that the rules, on the published synapse table, give the published bouton tables: the same rules on
SYNAPSE_TABLE_PUBLISHED_FTR, each row with its own synapse_id and rows with no synapse_id left out, must give the
published b_table and bwf_table (BOUTON_TABLE_PUBLISHED_FTR, BOUTON_FILOPODIA_TABLE_PUBLISHED_FTR; pandas
assert_frame_equal, check_exact=False), and six planted wrong rules must each make that comparison fail. The rows with
no synapse_id are then also matched to the raw table (as below) and added: none of the matched ids is in the published
synapse table, so the tables must come out exactly the same, which runs the matching on real data. If any of this does
not hold, the script stops and writes nothing.

Then the MaxSI tables: the same rules on the MaxSI synapse table (SYNAPSE_TABLE_FTR under SPLIT_METHOD = "maxsi"). Here
the rows with no synapse_id also get one, by matching them to the raw synapse table SYNAPSE_TABLE_RAW_NO_SELF_FTR
(the bouton rows on neuron, partner, x, y, z; the filopodia rows on pre, post, pre_x, pre_y, pre_z); rows that match
more than one raw synapse stay out. The two tables are written to BOUTON_TABLE_FTR and BOUTON_FILOPODIA_TABLE_FTR, the
names the Fig 7 panel scripts read under "maxsi" (data/derived_rebuild_maxsi/, under the published file names). The
script then prints how many cells take their labels from the two other files and which super_class values among
them the published b_table does not have, how the tables differ from the published ones, and compares each cell's SI
with SI_UPDATED_FTR.

Counts are boutons (one (neuron, cluster) pair), cells or rows, as each line says. IDs are int64 on both sides of every
join. Every input is read only; the script stops at start-up if an input file is missing, if either output exists
already, or if an output path is one of its inputs or lies in data/derived or data/raw. Each table is written under a
temporary name and then renamed, so neither output is ever half written; if the second write fails, the first output
stays and must be removed by hand before the next run. Once the settings check has passed, everything the script
prints, including the message it stops with, also goes to the plain-text log BOUTON_MAXSI_LOG_TXT
(outputs_rebuild_maxsi/tools/bouton_maxsi_tables/), each run appended under a line with its start time.

Needs config.py on PIPELINE_RUN = "rebuild" and SPLIT_METHOD = "maxsi", after step 09 under "maxsi". Under
SPLIT_METHOD = "sfc" it stops: the SFC tables are the published ones, which the check above shows these rules
reproduce, so there is nothing to build. Under "minfisherp" and "maxphi" it stops too, as
processing/buhmann_maxsi_labels.py does, so BOUTON_TABLE_FTR and BOUTON_FILOPODIA_TABLE_FTR have no file there and the
panels that read them stop.
"""

#%%1 Imports and settings
import datetime
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (PIPELINE_RUN, SPLIT_METHOD, PUBLISHED_DERIVED_DIR, RAW_DATA_DIR, CLUSTER_INFO_FILTERED_FTR,
                    FILOPODIA_FILTERED_PUBLISHED_FTR, BOUTON_TABLE_PUBLISHED_FTR,
                    BOUTON_FILOPODIA_TABLE_PUBLISHED_FTR, SYNAPSE_TABLE_PUBLISHED_FTR, SYNAPSE_TABLE_FTR,
                    SYNAPSE_TABLE_RAW_NO_SELF_FTR, CONSOLIDATED_CELL_TYPES_CSV, CLASSIFICATION_CSV, SI_UPDATED_FTR,
                    BOUTON_TABLE_FTR, BOUTON_FILOPODIA_TABLE_FTR, BOUTON_MAXSI_LOG_TXT)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

if PIPELINE_RUN != "rebuild" or SPLIT_METHOD != "maxsi":
    raise SystemExit('Set PIPELINE_RUN = "rebuild" and SPLIT_METHOD = "maxsi" in config.py.')


class Tee:
    """Writes to the terminal and to the log file at once."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


LOG = Path(BOUTON_MAXSI_LOG_TXT)
LOG.parent.mkdir(parents=True, exist_ok=True)
_log = open(LOG, "a", encoding="utf-8", buffering=1)
sys.stdout, sys.stderr = Tee(sys.stdout, _log), Tee(sys.stderr, _log)
print(f"\n=== processing/bouton_maxsi_tables.py, started {datetime.datetime.now().isoformat(timespec='seconds')}; "
      f"log {LOG} ===", flush=True)

CI = CLUSTER_INFO_FILTERED_FTR
FIL = FILOPODIA_FILTERED_PUBLISHED_FTR
SYN = {"published": SYNAPSE_TABLE_PUBLISHED_FTR, "maxsi": SYNAPSE_TABLE_FTR}
RAW = SYNAPSE_TABLE_RAW_NO_SELF_FTR
TYPES = CONSOLIDATED_CELL_TYPES_CSV
CLASSES = CLASSIFICATION_CSV
B_OUT = Path(BOUTON_TABLE_FTR)
BWF_OUT = Path(BOUTON_FILOPODIA_TABLE_FTR)
KEY = ["neuron", "cluster"]
LETTER = {"A": 0, "D": 1, "L": 2}

inputs = [CI, FIL, BOUTON_TABLE_PUBLISHED_FTR, BOUTON_FILOPODIA_TABLE_PUBLISHED_FTR, SYN["published"], SYN["maxsi"],
          RAW, TYPES, CLASSES, SI_UPDATED_FTR]
missing = [str(p) for p in inputs if not Path(p).is_file()]
if missing:
    raise SystemExit("input files not found, nothing done: " + "; ".join(missing))
for out in (B_OUT, BWF_OUT):
    if (out.resolve() in {Path(p).resolve() for p in inputs}
            or any(out.resolve().is_relative_to(d.resolve()) for d in (PUBLISHED_DERIVED_DIR, RAW_DATA_DIR))):
        raise SystemExit(f"{out} is an input, or lies in data/derived or data/raw; no table written.")
if B_OUT.exists() or BWF_OUT.exists():
    raise SystemExit(f"{B_OUT.name} or {BWF_OUT.name} already exists in {B_OUT.parent}; move it away to build new ones.")


def read_synapses(path):
    """The synapse table as sorted numpy arrays: ids, pre and post letters (0 A, 1 D, 2 L), npil codes, SI_pre."""
    ids = feather.read_table(path, columns=["synapse_id"])["synapse_id"].to_numpy()
    comp = feather.read_table(path, columns=["comp"])["comp"]
    assert comp.null_count == 0, "the synapse table has a synapse with no comp"
    comp = comp.combine_chunks().dictionary_encode()
    words = comp.dictionary.to_pylist()
    codes = comp.indices.to_numpy()
    pre = np.array([LETTER[w[0]] for w in words], dtype="int8")[codes]
    post = np.array([LETTER[w[1]] for w in words], dtype="int8")[codes]
    del comp, codes
    npil = feather.read_table(path, columns=["npil"])["npil"].combine_chunks().dictionary_encode()
    npil_words = npil.dictionary.to_pylist()
    npil_codes = pc.fill_null(npil.indices, -1).to_numpy().astype("int32")
    del npil
    si = feather.read_table(path, columns=["SI_pre"])["SI_pre"].to_numpy()
    if not (np.diff(ids) > 0).all():
        order = np.argsort(ids, kind="stable")
        ids, pre, post, npil_codes, si = ids[order], pre[order], post[order], npil_codes[order], si[order]
        assert (np.diff(ids) > 0).all(), "the synapse table repeats a synapse_id"
    return {"ids": ids, "pre": pre, "post": post, "npil": npil_codes, "npil_words": npil_words, "SI_pre": si}


def read_columns(path, columns):
    """Columns of a feather file, one at a time, without the whole table in memory at once."""
    return pd.DataFrame({col: feather.read_table(path, columns=[col])[col].to_numpy() for col in columns})


def read_majd(path):
    """One of the published bouton tables, as pandas reads it, with each distinct text stored once."""
    t = feather.read_table(path)
    for i, field in enumerate(t.schema):
        if pa.types.is_string(field.type):
            t = t.set_column(i, field.name, t.column(i).dictionary_encode())
    d = t.to_pandas().reset_index(drop=True)
    del t
    for col in d.columns:
        if isinstance(d[col].dtype, pd.CategoricalDtype):
            d[col] = d[col].astype(object).where(d[col].notna(), None)
    return d.drop(columns=[c for c in ["__index_level_0__"] if c in d])


def read_rows(path, columns, rows):
    """The given columns of a feather file at the given row positions, read one column at a time."""
    idx = pa.array(rows)
    return pd.DataFrame({col: feather.read_table(path, columns=[col])[col].take(idx).to_numpy() for col in columns})


def look_up(syn, q):
    """Positions in the synapse table of the ids q, and whether each is there."""
    pos = np.searchsorted(syn["ids"], q)
    pos = np.minimum(pos, len(syn["ids"]) - 1)
    return pos, syn["ids"][pos] == q


def npil_codes(words, strip):
    """Codes for npil words, with the _L/_R ending removed or kept; returns (code per word, word per code)."""
    w = pd.Series(words, dtype=object)
    if strip:
        w = w.str.replace(r"_[LR]$", "", regex=True)
    uniq = pd.unique(w)
    return pd.Index(uniq).get_indexer(w).astype("int32"), list(uniq)


def one_npil(codes, keys, names):
    """Per group: the single npil if every row has the same one and none is missing, else None."""
    d = pd.DataFrame({"c": codes}).groupby(keys).c.agg(["min", "max"])
    ok = (d["min"] == d["max"]) & (d["min"] >= 0)
    return pd.Series(np.where(ok, np.array(names + [None], dtype=object)[np.where(ok, d["min"], -1)], None),
                     index=d.index, dtype=object)


def all_same(letters, keys):
    """Per group: 'A' if every letter is A, 'D' if every one is D, else None."""
    d = pd.DataFrame({"l": letters}).groupby(keys).l.agg(["min", "max"])
    out = np.full(len(d), None, dtype=object)
    out[((d["min"] == 0) & (d["max"] == 0)).to_numpy()] = "A"
    out[((d["min"] == 1) & (d["max"] == 1)).to_numpy()] = "D"
    return pd.Series(out, index=d.index, dtype=object)


#%%2 The per-synapse rows and the published tables (the cell labels, and the check on the published tables)
ci = read_columns(CI, ["neuron", "cluster", "synapse_id"])
assert ci.neuron.dtype == "int64" and ci.cluster.dtype == "int64"
cell_order = pd.unique(ci.neuron)
print(f"{CI.name}: {len(ci):,} rows, {len(cell_order):,} cells; "
      f"rows with no synapse_id: {int(ci.synapse_id.isna().sum()):,}", flush=True)
ci_noid = read_rows(CI, ["neuron", "cluster", "partner_id", "x", "y", "z"], np.flatnonzero(ci.synapse_id.isna()))
ci_noid = ci_noid.astype("int64")
ci = ci.dropna(subset=["synapse_id"])[KEY + ["synapse_id"]].astype({"synapse_id": "int64"}).reset_index(drop=True)
assert ci.synapse_id.is_unique, "a synapse_id is in two bouton rows"

fil = read_columns(FIL, ["pre", "post", "cluster", "synapse_id", "pre_x", "pre_y", "pre_z"])
assert (fil[["pre", "post", "cluster"]].dtypes == "int64").all()
print(f"{FIL.name}: {len(fil):,} rows; with no synapse_id: {int(fil.synapse_id.isna().sum()):,}", flush=True)
fil_noid = fil[fil.synapse_id.isna()].drop(columns="synapse_id").reset_index(drop=True)
for c in ["pre_x", "pre_y", "pre_z"]:
    assert (fil_noid[c] == fil_noid[c].round()).all(), f"a filopodia {c} is not a whole number"
    fil_noid[c] = fil_noid[c].astype("int64")
fil = fil.dropna(subset=["synapse_id"])[["post", "cluster", "synapse_id"]].astype({"synapse_id": "int64"})
fil = fil.reset_index(drop=True)

majd_b = read_majd(BOUTON_TABLE_PUBLISHED_FTR)
majd_bwf = read_majd(BOUTON_FILOPODIA_TABLE_PUBLISHED_FTR)
labels = majd_b.groupby("neuron")[["primary_type", "super_class"]].first()
labels.index = labels.index.astype("int64")
assert (majd_b.groupby("neuron")[["primary_type", "super_class"]].nunique(dropna=False) <= 1).all().all()
types = pd.read_csv(TYPES, usecols=["root_id", "primary_type"]).astype({"root_id": "int64"})
assert types.root_id.is_unique, f"{TYPES.name} repeats a cell"
classes = pd.read_csv(CLASSES, usecols=["root_id", "super_class"]).astype({"root_id": "int64"})
assert classes.root_id.is_unique, f"{CLASSES.name} repeats a cell"
types = types.set_index("root_id").primary_type
classes = classes.set_index("root_id").super_class
print(f"published b_table: {len(majd_b):,} boutons, {majd_b.neuron.nunique():,} cells; bwf_table: {len(majd_bwf):,} "
      f"boutons, {majd_bwf.neuron.nunique():,} cells", flush=True)


def label_cells(cells):
    """primary_type and super_class per cell: the published b_table first, then the two other files."""
    cells = pd.Index(np.unique(cells), dtype="int64")
    out = labels.reindex(cells)
    new = ~cells.isin(labels.index)
    out.loc[new, "primary_type"] = cells[new].map(types)
    out.loc[new, "super_class"] = cells[new].map(classes)
    return out.astype(object).where(out.notna(), None), int(new.sum())


#%%3 The rows joined to a synapse table, and the two tables built from them
def join(syn, ci_rows, fil_rows):
    """Bouton rows and filopodia rows whose synapse is in the table, with what the rules need from it."""
    pos, found = look_up(syn, ci_rows.synapse_id.to_numpy())
    p = pos[found]
    del pos
    c = {col: ci_rows[col].to_numpy()[found] for col in ci_rows.columns if col != "synapse_id"}
    c.update(letter=syn["pre"][p], npil=syn["npil"][p], SI_pre=syn["SI_pre"][p])
    c = pd.DataFrame(c)
    pos, found = look_up(syn, fil_rows.synapse_id.to_numpy())
    p = pos[found]
    f = {col: fil_rows[col].to_numpy()[found] for col in fil_rows.columns}
    f.update(letter=syn["post"][p], pre_letter=syn["pre"][p], npil=syn["npil"][p])
    return c, pd.DataFrame(f)


def build(c, f, words, variant=None):
    """b_table and bwf_table from the joined rows, with the published rules (or one planted wrong rule)."""
    kept_code, kept_names = npil_codes(words, strip=False)
    bare_code, bare_names = npil_codes(words, strip=True)
    bare_of_kept = dict(zip(kept_names, pd.Series(kept_names, dtype=object).str.replace(r"_[LR]$", "", regex=True)))

    # b_table
    keys = [c.neuron, c.cluster]
    count = c.groupby(KEY).size()
    if variant == "b count of all rows, id-less included":
        count = count.add(ci_noid.groupby(KEY).size(), fill_value=0).astype("int64").reindex(count.index)
    comp = all_same(c.letter.to_numpy(), keys)
    cg = c.groupby("neuron").SI_pre
    spread = cg.max() - cg.min()
    if (spread > 0).any():
        raise SystemExit(f"SI_pre differs within {int((spread > 0).sum())} cells, so a cell has no single SI")
    si = pd.Series(count.index.get_level_values("neuron").map(cg.first()).to_numpy(), index=count.index)
    nc = np.where(c.npil.to_numpy() >= 0, bare_code[np.maximum(c.npil.to_numpy(), 0)], -1)
    npil = one_npil(nc, keys, bare_names)
    if variant == "b npil ending removed after the one-value test":
        nk = np.where(c.npil.to_numpy() >= 0, kept_code[np.maximum(c.npil.to_numpy(), 0)], -1)
        npil = one_npil(nk, keys, kept_names).map(lambda v: None if v is None else bare_of_kept[v])

    # bwf_table
    fkeys = [f.post, f.cluster]
    fg = f.groupby(["post", "cluster"])
    fcount = fg.size()
    if variant == "bwf count without repeated rows":
        fcount = f.drop_duplicates(["synapse_id", "post", "cluster"]).groupby(["post", "cluster"]).size()
    fcomp = all_same(f.letter.to_numpy(), fkeys)
    if variant == "bwf comp from the presynaptic letter":
        fcomp = all_same(f.pre_letter.to_numpy(), fkeys)
    nk = np.where(f.npil.to_numpy() >= 0, kept_code[np.maximum(f.npil.to_numpy(), 0)], -1)
    fnpil = one_npil(nk, fkeys, kept_names).map(lambda v: None if v is None else bare_of_kept[v])
    if variant == "bwf npil ending removed first":
        nb = np.where(f.npil.to_numpy() >= 0, bare_code[np.maximum(f.npil.to_numpy(), 0)], -1)
        fnpil = one_npil(nb, fkeys, bare_names)

    b = pd.DataFrame({"count": count, "SI": si, "npil": npil, "comp": comp}).reset_index()
    b["filopodia"] = pd.MultiIndex.from_frame(b[KEY]).isin(fcount.index)
    if variant == "bwf comp from b's comp":
        fcomp = b.set_index(KEY).comp.reindex(fcount.index.rename(KEY)).set_axis(fcount.index)
    rank = pd.Series(np.arange(len(cell_order)), index=cell_order)
    b["rank"] = b.neuron.map(rank)
    b = b.sort_values(["rank", "cluster"], kind="stable").drop(columns="rank").reset_index(drop=True)
    bwf = pd.DataFrame({"count": fcount, "npil": fnpil, "comp": fcomp}).reset_index()
    bwf = bwf.rename(columns={"post": "neuron", "cluster": "bwf"}).sort_values(["neuron", "bwf"]).reset_index(drop=True)
    cell_si = b.groupby("neuron").SI.first()
    bwf["SI"] = bwf.neuron.map(cell_si).astype("float64")

    lab, n_new = label_cells(np.concatenate([b.neuron.to_numpy(), bwf.neuron.to_numpy()]))
    for t in (b, bwf):
        t["primary_type"] = t.neuron.map(lab.primary_type).astype(object)
        t["super_class"] = t.neuron.map(lab.super_class).astype(object)
        for col in ["primary_type", "super_class", "npil", "comp"]:
            t[col] = t[col].astype(object).where(t[col].notna(), None)
    b = b[["neuron", "cluster", "count", "SI", "npil", "comp", "primary_type", "super_class", "filopodia"]]
    b = b.astype({"neuron": "Int64", "cluster": "int64", "count": "int64", "SI": "float64", "filopodia": "bool"})
    bwf = bwf[["neuron", "bwf", "count", "npil", "comp", "primary_type", "super_class", "SI"]]
    bwf = bwf.astype({"neuron": "Int64", "bwf": "Int64", "count": "Int64", "SI": "float64"})
    info = {"cells not in the published b_table": n_new,
            "bwf boutons with no SI": int(bwf.SI.isna().sum()),
            "bwf boutons with no b_table row": int((~pd.MultiIndex.from_frame(bwf[["neuron", "bwf"]])
                                                    .isin(pd.MultiIndex.from_frame(b[KEY]))).sum())}
    return b, bwf, info


def same(a, m):
    """Raises AssertionError if the rebuilt table is not the published one."""
    pd.testing.assert_frame_equal(a.reset_index(drop=True), m.reset_index(drop=True), check_exact=False)


def per_column(a, m, key):
    """Rows equal per column, matched on the table's key (missing equals missing; SI within 1e-12)."""
    j = a.merge(m, on=key, how="outer", suffixes=("", "_m"), indicator=True)
    print(f"  rows rebuilt {len(a):,}, published {len(m):,}, both {int((j._merge == 'both').sum()):,}")
    j = j[j._merge == "both"]
    for col in [c for c in a.columns if c not in key]:
        x, y = j[col], j[col + "_m"]
        if col == "SI":
            eq = ((x - y).abs() <= 1e-12) | (x.isna() & y.isna())
        else:
            eq = (x.astype(object) == y.astype(object)) | (x.isna() & y.isna())
        print(f"  {col:13s} {int(eq.sum()):>10,} of {len(j):,}")


#%%4 The id-less rows matched to the raw table (used in the published check and in the MaxSI tables)
print("\n== id-less rows matched to the raw table ==", flush=True)
raw_pre = feather.read_table(RAW, columns=["pre"])["pre"]
wanted = pa.array(np.union1d(ci_noid.neuron.unique(), fil_noid.pre.unique()))
raw_rows = np.flatnonzero(pc.is_in(raw_pre, value_set=wanted).to_numpy(zero_copy_only=False))
del raw_pre
raw = read_rows(RAW, ["synapse_id", "pre", "post", "pre_x", "pre_y", "pre_z"], raw_rows).astype("int64")


def match(rows, on_rows, name):
    """A synapse_id for each id-less row from the raw table; rows matching none or more than one stay out."""
    r = rows.assign(row=np.arange(len(rows)))
    m = r.merge(raw.rename(columns=dict(zip(["pre", "post", "pre_x", "pre_y", "pre_z"], on_rows))), on=on_rows,
                how="left")
    n = m.groupby("row").synapse_id.count()
    print(f"{name}: id-less rows {len(rows):,}; matched to one raw synapse {int((n == 1).sum()):,}, to more than one "
          f"{int((n > 1).sum()):,} (left out), to none {int((n == 0).sum()):,}")
    m = m.dropna(subset=["synapse_id"])
    m = m[~m.row.duplicated(keep=False)].astype({"synapse_id": "int64"})
    return m


ci_m = match(ci_noid, ["neuron", "partner_id", "x", "y", "z"], "cluster_info")[KEY + ["synapse_id"]]
fil_m = match(fil_noid, ["pre", "post", "pre_x", "pre_y", "pre_z"], "filopodia")[["post", "cluster", "synapse_id"]]
del raw
ci_all = pd.concat([ci.assign(src=np.int8(0)), ci_m.assign(src=np.int8(1))], ignore_index=True)
assert ci_all.synapse_id.is_unique, "a matched synapse_id is in two bouton rows"
fil_all = pd.concat([fil.assign(src=np.int8(0)), fil_m.assign(src=np.int8(1))], ignore_index=True)
n_matched = (len(ci_m), len(fil_m))
del ci_m, fil_m


#%%5 The check on the published tables: the published synapse table must give the published tables
print("\n== published: article table, each row's own synapse_id ==", flush=True)
syn = read_synapses(SYN["published"])
c, f = join(syn, ci, fil)
words = syn["npil_words"]
in_pub = int(look_up(syn, ci_all.synapse_id.to_numpy()[ci_all.src.to_numpy() == 1])[1].sum()
             + look_up(syn, fil_all.synapse_id.to_numpy()[fil_all.src.to_numpy() == 1])[1].sum())
cm, fm = join(syn, ci_all, fil_all)
del syn
print(f"bouton rows in the table: {len(c):,} of {len(ci):,} with an id; filopodia rows: {len(f):,} of {len(fil):,} "
      f"with an id; filopodia rows that repeat a synapse_id, post and cluster: "
      f"{int(f.duplicated(['synapse_id', 'post', 'cluster']).sum()):,}", flush=True)
pub_b, pub_bwf, info = build(c, f, words)
print(f"built: b_table {len(pub_b):,} boutons, bwf_table {len(pub_bwf):,} boutons; {info}")
print("b_table, rows equal per column:")
per_column(pub_b, majd_b, KEY)
print("bwf_table, rows equal per column:")
per_column(pub_bwf, majd_bwf, ["neuron", "bwf"])
golden = True
for name, a, m in [("b_table", pub_b, majd_b), ("bwf_table", pub_bwf, majd_bwf)]:
    try:
        same(a, m)
        print(f"golden master {name}: equal to the published table (assert_frame_equal, check_exact=False)")
    except AssertionError as e:
        golden = False
        print(f"golden master {name}: DIFFERS: {str(e).splitlines()[0:6]}")

print("planted wrong rules (each must make the comparison fail):")
planted_ok = True
for v in ["b count of all rows, id-less included", "b npil ending removed after the one-value test",
          "bwf npil ending removed first", "bwf comp from the presynaptic letter",
          "bwf comp from b's comp", "bwf count without repeated rows"]:
    vb, vbwf, _ = build(c, f, words, variant=v)
    failed = []
    for name, a, m in [("b_table", vb, majd_b), ("bwf_table", vbwf, majd_bwf)]:
        try:
            same(a, m)
        except AssertionError:
            failed.append(name)
    planted_ok &= bool(failed)
    print(f"  {v:48s} {'FAILS on ' + ', '.join(failed) if failed else 'PASSES: the check cannot tell'}", flush=True)
    del vb, vbwf
del c, f, ci, fil, majd_b, majd_bwf
print(f"id-less rows matched to the raw table: {n_matched[0]:,} bouton rows, {n_matched[1]:,} filopodia rows; "
      f"of their synapse_ids, in the published synapse table: {in_pub:,} (must be 0)")
mb, mbwf, minfo = build(cm, fm, words)
del cm, fm
matched_same = in_pub == 0
for name, a, m in [("b_table", mb, pub_b), ("bwf_table", mbwf, pub_bwf)]:
    try:
        pd.testing.assert_frame_equal(a, m, check_exact=True)
        print(f"published, with the matched id-less rows added: {name} exactly equal to the one built without them")
    except AssertionError as e:
        matched_same = False
        print(f"published, with the matched id-less rows added: {name} DIFFERS: {str(e).splitlines()[0:6]}")
del mb, mbwf
if not (golden and planted_ok and matched_same):
    raise SystemExit("the check on the published tables failed, a planted rule passed, or the matched id-less rows "
                     "changed the published-mode tables: the MaxSI tables are not written")
pub_b = pub_b[KEY + ["count", "SI", "npil", "comp", "filopodia"]].astype({"npil": "category"})
pub_bwf = pub_bwf[["neuron", "bwf", "SI", "npil", "comp"]].astype({"npil": "category"})


#%%6 The MaxSI tables: the id-less rows matched to the raw table
print("\n== maxsi: MaxSI article table; id-less rows matched to the raw table ==", flush=True)
syn = read_synapses(SYN["maxsi"])
c, f = join(syn, ci_all, fil_all)
words = syn["npil_words"]
del syn
print(f"bouton rows in the MaxSI table: {len(c):,} of {len(ci_all):,} (id-less rows entering b_table: "
      f"{int(c.src.sum()):,}); filopodia rows: {len(f):,} of {len(fil_all):,} (id-less rows entering bwf_table: "
      f"{int(f.src.sum()):,})", flush=True)
mx_b, mx_bwf, info = build(c, f, words)
del c, f
print(f"built: {info}")
new_cells = pd.Index(np.union1d(mx_b.neuron.to_numpy("int64"), mx_bwf.neuron.to_numpy("int64")))
new_cells = new_cells[~new_cells.isin(labels.index)]
new_sc = pd.concat([mx_b, mx_bwf]).astype({"neuron": "int64"}).drop_duplicates("neuron").set_index("neuron")
new_sc = new_sc.super_class.reindex(new_cells)
print(f"cells labelled from {Path(TYPES).name} and {Path(CLASSES).name} (not in the published b_table): "
      f"{len(new_cells):,}; their super_class, cells: "
      + "; ".join(f"{k if isinstance(k, str) else 'missing'} {v:,}" for k, v in new_sc.value_counts(dropna=False).items()))
print(f"  super_class values among them that the published b_table does not have: "
      f"{sorted(set(new_sc.dropna()) - set(labels.super_class.dropna()))}")


def write_whole(df, path):
    """Written under a temporary name and renamed, so this table is never half written. The two tables are written
    one after the other: if the second write fails, the first output stays and must be removed by hand."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    df.to_feather(tmp)
    tmp.replace(path)


B_OUT.parent.mkdir(parents=True, exist_ok=True)
write_whole(mx_b, B_OUT)
write_whole(mx_bwf, BWF_OUT)
print(f"wrote {B_OUT} ({len(mx_b):,} boutons) and {BWF_OUT} ({len(mx_bwf):,} boutons)", flush=True)


#%%7 The MaxSI tables against the published ones
print("\nMaxSI against published (boutons are (neuron, cluster) rows; cells are distinct neuron values):")
for name, p, x, key in [("b_table", pub_b, mx_b, KEY), ("bwf_table", pub_bwf, mx_bwf, ["neuron", "bwf"])]:
    pk, xk = pd.MultiIndex.from_frame(p[key]), pd.MultiIndex.from_frame(x[key])
    p_cells, x_cells = set(p.neuron), set(x.neuron)
    print(f"{name}: boutons published {len(p):,}, MaxSI {len(x):,} (leave {int((~pk.isin(xk)).sum()):,}, "
          f"enter {int((~xk.isin(pk)).sum()):,}); cells published {len(p_cells):,}, MaxSI {len(x_cells):,} "
          f"(leave {len(p_cells - x_cells):,}, enter {len(x_cells - p_cells):,})")
    pv, xv = p.comp.value_counts(dropna=False), x.comp.value_counts(dropna=False)
    print(f"  comp, boutons: " + "; ".join(f"{k if isinstance(k, str) else 'missing'} {int(pv.get(k, 0)):,} -> "
                                           f"{int(xv.get(k, 0)):,}" for k in ["A", "D", None]))
    print(f"  npil missing, boutons: {int(p.npil.isna().sum()):,} -> {int(x.npil.isna().sum()):,}")
    print(f"  SI: published {p.SI.min():.6g} to {p.SI.max():.6g}, MaxSI {x.SI.min():.6g} to {x.SI.max():.6g}; "
          f"missing {int(p.SI.isna().sum()):,} -> {int(x.SI.isna().sum()):,}")
print(f"b_table filopodia True, boutons: {int(pub_b.filopodia.sum()):,} -> {int(mx_b.filopodia.sum()):,}")
print(f"b_table boutons with count below 3: published {int((pub_b['count'] < 3).sum()):,} of {len(pub_b):,}, "
      f"MaxSI {int((mx_b['count'] < 3).sum()):,} of {len(mx_b):,}")
step09 = pd.read_feather(SI_UPDATED_FTR, columns=["root_id", "SI"])
step09 = step09.astype({"root_id": "int64"}).set_index("root_id").SI
cell_si = mx_b.groupby("neuron").SI.first()
cell_si.index = cell_si.index.astype("int64")
s09 = step09.reindex(cell_si.index)
print(f"MaxSI b_table SI against step 09's {Path(SI_UPDATED_FTR).name} (cells): {len(cell_si):,} cells, equal bit for "
      f"bit {int((s09.to_numpy() == cell_si.to_numpy()).sum()):,}, missing from step 09 {int(s09.isna().sum()):,}; "
      f"cells at SI 1: {int((cell_si == 1).sum()):,} (step 09 has {int((step09 == 1).sum()):,} of {len(step09):,})")
