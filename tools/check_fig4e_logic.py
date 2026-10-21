"""Check what figures/fig4/syntype_x_pathway_layers.py computes, on a small made-up table.

The revised Fig 4E script reads a per-type table this repository does not deposit
(ex_table_npil_comp_pre_post_hiSI_PerTypeSummary.csv), so it cannot be run from a clean clone.
This tool runs the script's own text on 27 made-up rows of that table, and on a made-up
neuron_data table for the olfactory classes, written to a temporary folder together with a
stand-in `config` module, and compares the tables it writes and the lines it plots with values
worked out by hand (written below, with their arithmetic, separately from the script).

The script is run as it is, except for its DATA_SOURCE and SI_SOURCE settings (replaced by exact
text for the variants below) and matplotlib drawing to no screen (the Agg backend), with the
outputs going to the temporary folder.

Checks, on the default setting (DATA_SOURCE = 'csv'):
  - the neuropil's _L/_R suffix is dropped and the two sides pooled; 'LAL' is not 'LA', and
    'MB_CA_R' is 'MB_CA'; a row with no neuropil is in no layer;
  - the shares are of AD + AA + DD + DA only (a link-compartment row is left out of the total,
    though its types count in nTypes);
  - the VPN row is every row with a visual projection cell on either side, wherever it is, so it
    overlaps the LO row; a layer with no synapses gives NaN shares, not 0;
  - within-type means the same pre and post type (two empty types count as the same type, as
    in MATLAB); nTypes counts the distinct types on either side;
  - the collapsed groups LA+ME and LO+LOP (Tables 3 and 4) and their two line plots;
  - the olfactory types: class olfactory, ALPN, Kenyon_Cell or MBON in the neuron_data table,
    primary_type or, when it is empty, cell_type; a row counts only when both types are
    olfactory; the four olfactory layers, and all, within- and between-type tables;
  - the two olfactory line plots leave out LH+MB-CA (which stays in the tables);
  - each line plot: DD and AA in that order, their colours, the x labels, y from 0 to 100;
then:
  - DATA_SOURCE = 'princeton' rebuilds the table from made-up synapse and neuron tables: cells
    with SI exactly 0.1 are kept, below it left out, cells with no type or an empty type left out,
    synapses summed per pre type, post type, comp and neuropil; adding SI_SOURCE = 'SI_updated'
    takes the SI from SI_UPDATED_FTR; DATA_SOURCE = 'buhmann' reads the other two tables and
    ignores SI_SOURCE;
  - a missing per-type table stops the script with a message that names it, and a misspelled
    DATA_SOURCE stops it.

It does not test the stacked-bar plots beyond their bar heights, the layout, or anything about
the real tables. A PASS means the script computes what its code says; it says nothing about
whether the figure is the right analysis.

Usage:
    python tools/check_fig4e_logic.py
    python tools/check_fig4e_logic.py --script path/to/another_copy.py

Needs no data. Takes a few seconds. Writes outputs/tools/check_fig4e_logic/report.txt.
Exits with status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import io
import math
import sys
import tempfile
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import OUTPUT_DIR  # noqa: E402

warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

SCRIPT = REPO_ROOT / "figures" / "fig4" / "syntype_x_pathway_layers.py"
OUT_DIR = OUTPUT_DIR / "tools" / "check_fig4e_logic"
TOL = 1e-6
SETTING_SOURCE = "DATA_SOURCE = 'csv'\n"
SETTING_SI = "SI_SOURCE = 'neuron_table'\n"

# ---------------------------------------------------------------- the made-up per-type table
# preType, postType, comp, npil, pre_SuperClass, post_SuperClass, synapses (preSI = postSI = 0.5)
OP, VP, CE = "optic", "visual_projection", "central"
ROWS = [
    ("T1", "T1", "DD", "ME_L", OP, OP, 10),    # r1  ME (left)
    ("T1", "T1", "DD", "ME_R", OP, OP, 6),     # r2  ME (right): pooled with r1
    ("T1", "T2", "AD", "LA_R", OP, OP, 4),     # r3  LA
    ("T2", "T1", "AA", "LA_L", OP, OP, 2),     # r4  LA
    ("T1", "T1", "AL", "ME_L", OP, OP, 50),    # r5  ME, link compartment: not in the shares
    ("T3", "T3", "AA", "LAL_R", CE, CE, 7),    # r6  LAL, not LA; not olfactory
    ("T2", "T2", "DA", "LO_L", OP, OP, 3),     # r7  LO
    ("T2", "T4", "DD", "LOP_R", OP, OP, 5),    # r8  LOP
    ("V1", "V1", "AA", "LO_R", VP, VP, 8),     # r9  LO and VPN
    ("V1", "T2", "AD", "LO_L", VP, OP, 4),     # r10 LO and VPN (pre)
    ("T2", "V2", "DD", "PLP_L", OP, VP, 6),    # r11 VPN (post), in no visual layer
    ("V1", "V1", "DD", "", VP, VP, 2),         # r12 VPN, no neuropil
    ("C1", "C2", "AA", "ME_L", CE, CE, 1),     # r13 ME: any cell types count in the ALL tables
    ("", "", "DD", "ME_R", OP, OP, 3),         # r14 ME, two empty types: within-type, as in MATLAB
    ("ORN_a", "PN1", "DD", "AL_R", "sensory", CE, 20),   # o1  AL
    ("ORN_a", "ORN_a", "DD", "AL_L", "sensory", "sensory", 5),   # o2  AL, within
    ("PN1", "PN1", "AA", "AL_L", CE, CE, 2),             # o3  AL, within
    ("ORN_a", "LH1", "DD", "AL_R", "sensory", CE, 100),  # o4  LH1 is not olfactory: left out
    ("PN2", "KCab", "AD", "MB_CA_R", CE, CE, 9),         # o5  LH+MB-CA (PN2 is a cell_type)
    ("PN1", "LH1", "AD", "LH_R", CE, CE, 30),            # o6  left out
    ("KCab", "KCab", "AA", "MB_ML_L", CE, CE, 12),       # o7  MB lobes, within
    ("KCab", "MBON01", "AD", "MB_VL_R", CE, CE, 18),     # o8  MB lobes
    ("KCab", "KCab", "DD", "MB_PED_L", CE, CE, 6),       # o9  MB lobes, within
    ("MBON01", "MBON01", "AA", "SMP_R", CE, CE, 3),      # o10 CRE+SMP+SIP+LAL, within
    ("MBON01", "PN1", "AD", "CRE_L", CE, CE, 6),         # o11 CRE+...
    ("MBON01", "MBON01", "DD", "LAL_R", CE, CE, 2),      # o12 CRE+... (LAL), within
    ("KCab", "MBON01", "LD", "MB_VL_L", CE, CE, 40),     # o13 MB lobes, link: not in the shares
]
# class, primary_type, cell_type (the neuron_data stand-in)
NEURON_DATA = [
    ("olfactory", "ORN_a", ""), ("ALPN", "PN1", ""), ("ALPN", "", "PN2"), ("Kenyon_Cell", "KCab", ""),
    ("MBON", "MBON01", ""), ("LHN", "LH1", ""), ("olfactory", "", ""), ("Kenyon_Cell", None, None),
    ("optic", "T1", ""), ("ALPN", "PN1", "PNx"),   # a cell_type that differs from a non-empty primary_type
]
N = float("nan")
# ---------------------------------------------------------------- expected values, by hand
# columns: synapses, nTypes, AD, AA, DD, DA (%)
# Table 1, ALL:
#   LA  r3 AD 4, r4 AA 2 = 6; types T1, T2
#   ME  r1 + r2 + r14 DD 19, r13 AA 1 = 20 (r5 AL left out of the total); types T1, C1, C2, '' (r5's T1 counts)
#   LO  r7 DA 3, r9 AA 8, r10 AD 4 = 15; types T2, V1
#   LOP r8 DD 5; types T2, T4
#   VPN r9 AA 8, r10 AD 4, r11 DD 6, r12 DD 2 = 20; types V1, T2, V2
EXPECTED = {
    "LayerComposition_AllSynapses.csv": ("layer", [
        ("LA", 6, 2, 400 / 6, 200 / 6, 0, 0), ("ME", 20, 4, 0, 5, 95, 0), ("LO", 15, 2, 400 / 15, 800 / 15, 0, 20),
        ("LOP", 5, 2, 0, 0, 100, 0), ("VPN", 20, 3, 20, 40, 40, 0)]),
    # Table 2, WITHIN-TYPE: LA none (r3, r4 cross types): 0 synapses, NaN, 0 types;
    #   ME r1 + r2 + r14 DD 19 (r5 in the rows, not the total); types T1, ''
    #   LO r7 DA 3, r9 AA 8 = 11; LOP none; VPN r9 AA 8, r12 DD 2 = 10; type V1
    "LayerComposition_WithinType.csv": ("layer", [
        ("LA", 0, 0, N, N, N, N), ("ME", 19, 2, 0, 0, 100, 0), ("LO", 11, 2, 0, 800 / 11, 0, 300 / 11),
        ("LOP", 0, 0, N, N, N, N), ("VPN", 10, 1, 0, 80, 20, 0)]),
    # Table 3: LA+ME AD 4, AA 3, DD 19 = 26; types T1, T2, C1, C2, ''.  LO+LOP DA 3, AA 8, AD 4, DD 5 = 20
    "LayerComposition_AllSynapses_collapsed.csv": ("group", [
        ("LA+ME", 26, 5, 400 / 26, 300 / 26, 1900 / 26, 0), ("LO+LOP", 20, 3, 20, 40, 25, 15),
        ("VPN", 20, 3, 20, 40, 40, 0)]),
    # Table 4: LA+ME DD 19; LO+LOP DA 3, AA 8 = 11; VPN AA 8, DD 2
    "LayerComposition_WithinType_collapsed.csv": ("group", [
        ("LA+ME", 19, 2, 0, 0, 100, 0), ("LO+LOP", 11, 2, 0, 800 / 11, 0, 300 / 11), ("VPN", 10, 1, 0, 80, 20, 0)]),
    # Olfactory types: ORN_a, PN1, PN2 (cell_type), KCab, MBON01 (5); LH1 (class LHN) and T1 are not.
    # Olfactory Table 1: AL o1 DD 20, o2 DD 5, o3 AA 2 = 27 (o4 left out); types ORN_a, PN1
    #   LH+MB-CA o5 AD 9; types PN2, KCab.  MB lobes o7 AA 12, o8 AD 18, o9 DD 6 = 36 (o13 LD left out)
    #   CRE+SMP+SIP+LAL o10 AA 3, o11 AD 6, o12 DD 2 = 11 (r6 T3 in LAL is not olfactory); types MBON01, PN1
    "LayerComposition_Olfactory_AllSynapses.csv": ("layer", [
        ("AL", 27, 2, 0, 200 / 27, 2500 / 27, 0), ("LH+MB-CA", 9, 2, 100, 0, 0, 0),
        ("MB lobes (ML/PED/VL)", 36, 2, 50, 100 / 3, 50 / 3, 0), ("CRE+SMP+SIP+LAL", 11, 2, 600 / 11, 300 / 11, 200 / 11, 0)]),
    # Table 2 within: AL o2 DD 5, o3 AA 2 = 7; LH+MB-CA none; MB lobes o7 AA 12, o9 DD 6 = 18; CRE+ o10 AA 3, o12 DD 2 = 5
    "LayerComposition_Olfactory_WithinType.csv": ("layer", [
        ("AL", 7, 2, 0, 200 / 7, 500 / 7, 0), ("LH+MB-CA", 0, 0, N, N, N, N),
        ("MB lobes (ML/PED/VL)", 18, 1, 0, 200 / 3, 100 / 3, 0), ("CRE+SMP+SIP+LAL", 5, 1, 0, 60, 40, 0)]),
    # Table 3 between: AL o1 DD 20; LH+MB-CA o5 AD 9; MB lobes o8 AD 18 (o13 LD in the rows, not the total);
    #   CRE+ o11 AD 6
    "LayerComposition_Olfactory_BetweenType.csv": ("layer", [
        ("AL", 20, 2, 0, 0, 100, 0), ("LH+MB-CA", 9, 2, 100, 0, 0, 0),
        ("MB lobes (ML/PED/VL)", 18, 2, 100, 0, 0, 0), ("CRE+SMP+SIP+LAL", 6, 2, 100, 0, 0, 0)]),
}
# the four line plots of Fig 4E: (figure number, x labels, DD, AA)
LINES = {
    3: (["LA+ME", "LO+LOP", "VPN"], [1900 / 26, 25, 40], [300 / 26, 40, 40]),
    4: (["LA+ME", "LO+LOP", "VPN"], [100, 0, 20], [0, 800 / 11, 80]),
    5: (["AL", "MB lobes (ML/PED/VL)", "CRE+SMP+SIP+LAL"], [2500 / 27, 50 / 3, 200 / 11], [200 / 27, 100 / 3, 300 / 11]),
    6: (["AL", "MB lobes (ML/PED/VL)", "CRE+SMP+SIP+LAL"], [500 / 7, 100 / 3, 40], [200 / 7, 200 / 3, 60]),
}
COLOR_DD, COLOR_AA = np.array([243, 185, 91]) / 255, np.array([120, 80, 161]) / 255
# plot 1, stacked bars: per layer AD, AA, DD, DA heights (NaN drawn as nothing)
BARS_1 = [(400 / 6, 200 / 6, 0, 0), (0, 5, 95, 0), (400 / 15, 800 / 15, 0, 20), (0, 0, 100, 0), (20, 40, 40, 0)]

# ---------------------------------------------------------------- the stand-in rebuild
# root_id, primary_type, super_class, SI (Princeton neuron table)
ID_BASE = 720575940600000000
CELLS = [(1, "T1", OP, 0.5), (2, "T1", OP, 0.1),       # SI exactly 0.1: kept
         (3, "T2", OP, 0.09),                         # below 0.1: left out (0.5 in SI_updated)
         (4, None, OP, 0.5), (5, "", OP, 0.5),       # no type, empty type: left out
         (6, "V1", VP, 0.5),
         (7, "T1", None, 0.5),                        # no super_class: its rows must not be dropped
         (8, "KCx", CE, 0.5)]                         # NEURON_DATA_FTR calls this cell KCab, class Kenyon_Cell
# pre, post, npil, comp, number of synapses (Princeton synapse table)
SYNAPSES = [(1, 2, "ME_L", "DD", 3), (2, 1, "ME_R", "DD", 1), (1, 3, "ME_L", "AA", 5), (1, 4, "ME_L", "AA", 5),
            (1, 5, "ME_L", "AA", 5), (1, 6, "LO_R", "AD", 2), (6, 6, "LO_L", "AA", 4),
            (7, 1, "ME_L", "DD", 2), (8, 8, "MB_ML_L", "AA", 3)]
# Princeton: ME DD 3 + 1 + 2 (7 -> 1, no super_class) = 6 (types T1); LO AD 2, AA 4 = 6; VPN the same two rows.
#   Within: ME DD 6 (T1 -> T1); LO AA 4 (V1 -> V1)
#   Olfactory: cell 8's type in the neuron table is KCx, and NEURON_DATA_FTR puts that cell in Kenyon_Cell, so
#   KCx is an olfactory type of the stand-in: MB lobes AA 3 (all and within)
# SI_updated (cell 3 at 0.5): ME also AA 5 -> 11: DD 6/11, AA 5/11; within unchanged
# Buhmann (no 2 -> 1 synapse, and cell 3 at 0.5 in its own table, which SI_SOURCE does not change): ME DD 5, AA 5 -> 10
STANDIN = {
    "princeton": {"ME": (6, 0, 0, 100, 0), "LO": (6, 200 / 6, 400 / 6, 0, 0), "VPN": (6, 200 / 6, 400 / 6, 0, 0),
                  "within ME": (6, 0, 0, 100, 0), "within LO": (4, 0, 100, 0, 0),
                  "olf MB lobes (ML/PED/VL)": (3, 0, 100, 0, 0), "olf within MB lobes (ML/PED/VL)": (3, 0, 100, 0, 0)},
    "SI_updated": {"ME": (11, 0, 500 / 11, 600 / 11, 0), "LO": (6, 200 / 6, 400 / 6, 0, 0),
                   "within ME": (6, 0, 0, 100, 0)},
    "buhmann": {"ME": (10, 0, 50, 50, 0), "within ME": (5, 0, 0, 100, 0)},
}

# A second, small per-type table for text MATLAB's readtable keeps as text or trims:
#   e1 'NA' -> 'None', DD 2 in ' ME_L ' (spaces trimmed: ME); two different types (pandas' default would read
#      both as missing, i.e. the same empty type); e2 'None' -> 'None', AA 1 in ME_R (within-type);
#   e3 a missing synapse count in LO: LO's synapses and shares are NaN, as MATLAB's sum gives
EDGE_ROWS = [("NA", "None", "DD", " ME_L ", OP, OP, 2), ("None", "None", "AA", "ME_R", OP, OP, 1),
             ("T5", "T5", "DD", "LO_L", OP, OP, None)]
EDGE = {"ME": (3, 2, 0, 100 / 3, 200 / 3, 0), "within ME": (1, 1, 0, 100, 0, 0), "LO": (N, 1, N, N, N, N)}
SVG_NAMES = {1: "LayerComposition_AllSynapses.svg", 2: "LayerComposition_WithinType.svg",
             3: "LayerComposition_AllSynapses_collapsed_lines.svg", 4: "LayerComposition_WithinType_collapsed_lines.svg",
             5: "LayerComposition_Olfactory_AllSynapses_lines.svg", 6: "LayerComposition_Olfactory_WithinType_lines.svg"}
# plot 2, stacked bars of the WITHIN-TYPE table (NaN rows draw NaN-height bars)
BARS_2 = [(N, N, N, N), (0, 0, 100, 0), (0, 800 / 11, 0, 300 / 11), (N, N, N, N), (0, 80, 20, 0)]

results: list[tuple[bool, str]] = []
_saved: list[tuple[int, str]] = []


def close(a, b):
    if isinstance(a, float) and math.isnan(a) or isinstance(b, float) and math.isnan(b):
        return isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)
    return abs(float(a) - float(b)) <= TOL


def write_inputs(tmp: Path, rows=ROWS):
    (tmp / "data").mkdir()
    pd.DataFrame([dict(preType=a, postType=b, comp=c, npil=d, pre_SuperClass=e, post_SuperClass=f,
                       preSI=0.5, postSI=0.5, pre_nt="ACH", post_nt="ACH", synapses=s)
                  for a, b, c, d, e, f, s in rows]).to_csv(tmp / "data" / "per_type.csv", index=False)
    nd = pd.DataFrame(NEURON_DATA, columns=["class", "primary_type", "cell_type"]).assign(
        root_id=np.arange(len(NEURON_DATA), dtype=np.int64) + ID_BASE + 100)
    cell8 = pd.DataFrame([dict(primary_type="KCab", cell_type="", root_id=ID_BASE + 8)]).assign(**{"class": "Kenyon_Cell"})
    pd.concat([nd, cell8], ignore_index=True).to_feather(tmp / "data" / "neuron_data.ftr")
    cells = pd.DataFrame(CELLS, columns=["root_id", "primary_type", "super_class", "SI"])
    cells["root_id"] = cells["root_id"].astype(np.int64) + ID_BASE
    cells.to_feather(tmp / "data" / "neurons_p.ftr")
    upd = cells[["root_id", "SI"]].copy()
    upd.loc[upd["root_id"] == ID_BASE + 3, "SI"] = 0.5
    upd.to_feather(tmp / "data" / "si_updated.ftr")
    cells.assign(SI=upd["SI"]).to_feather(tmp / "data" / "neurons_b.ftr")
    syn = pd.DataFrame([(ID_BASE + p, ID_BASE + q, n, c) for p, q, n, c, k in SYNAPSES for _ in range(k)],
                       columns=["pre", "post", "npil", "comp"])
    syn.to_feather(tmp / "data" / "synapses_p.ftr")
    syn[~((syn["pre"] == ID_BASE + 2) & (syn["post"] == ID_BASE + 1))].reset_index(drop=True).to_feather(
        tmp / "data" / "synapses_b.ftr")
    (tmp / "config.py").write_text(
        "from pathlib import Path\n"
        f"D = Path({str(tmp / 'data')!r})\n"
        f"OUTPUT_DIR = Path({str(tmp / 'out')!r})\n"
        "PER_TYPE_SUMMARY_CSV = D / 'per_type.csv'\n"
        "OLFACTORY_CLASS_NEUROPIL_CSV = D / 'absent.csv'\n"
        "NEURON_DATA_FTR = D / 'neuron_data.ftr'\n"
        "NEURON_TABLE_FTR = D / 'neurons_p.ftr'\nSYNAPSE_TABLE_FTR = D / 'synapses_p.ftr'\n"
        "NEURON_TABLE_NONP_FTR = D / 'neurons_b.ftr'\nSYNAPSE_TABLE_NONP_FTR = D / 'synapses_b.ftr'\n"
        "SI_UPDATED_FTR = D / 'si_updated.ftr'\n", encoding="utf-8")


def write_inputs_csv_only(tmp: Path):
    pd.DataFrame([dict(preType=a, postType=b, comp=c, npil=d, pre_SuperClass=e, post_SuperClass=f,
                       preSI=0.5, postSI=0.5, pre_nt="ACH", post_nt="ACH", synapses=s)
                  for a, b, c, d, e, f, s in ROWS]).to_csv(tmp / "data" / "per_type.csv", index=False)


def run(source: str, tmp: Path, settings: dict[str, str]):
    """Run the script text with the given setting replacements; return (namespace, stdout, error)."""
    for old, new in settings.items():
        if source.count(old) != 1:
            return None, "", f"setting line {old.strip()!r} found {source.count(old)} times"
        source = source.replace(old, new)
    fake = tmp / "figures" / "fig4" / "script.py"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text(source, encoding="utf-8")
    sys.modules.pop("config", None)
    saved_path = list(sys.path)
    plt.close("all")
    ns = {"__name__": "__main__", "__file__": str(fake)}
    out = io.StringIO()
    del _saved[:]
    real_savefig = matplotlib.figure.Figure.savefig

    def recording_savefig(fig, fname, *a, **k):
        _saved.append((fig.number, Path(str(fname)).name))
        return real_savefig(fig, fname, *a, **k)
    matplotlib.figure.Figure.savefig = recording_savefig
    try:
        with contextlib.redirect_stdout(out):
            exec(compile(source, str(fake), "exec"), ns)
        return ns, out.getvalue(), None
    except Exception as error:  # noqa: BLE001  a stop is a result here
        return ns, out.getvalue(), f"{type(error).__name__}: {error}"
    finally:
        matplotlib.figure.Figure.savefig = real_savefig
        sys.path[:] = saved_path
        sys.modules.pop("config", None)


def check_tables(out_dir: Path):
    for name, (key, rows) in EXPECTED.items():
        path = out_dir / name
        if not path.exists():
            results.append((False, f"{name}: written"))
            continue
        got = pd.read_csv(path)
        want = pd.DataFrame(rows, columns=[key, "synapses", "nTypes", "AD_pct", "AA_pct", "DD_pct", "DA_pct"])
        ok = list(got.columns) == list(want.columns) and len(got) == len(want) and \
            list(got[key]) == list(want[key]) and all(
                close(float(got.iloc[i][c]), float(want.iloc[i][c]))
                for i in range(len(want)) for c in want.columns[1:])
        detail = "" if ok else f"\n      got  {got.round(3).values.tolist()}\n      want {want.round(3).values.tolist()}"
        results.append((ok, f"{name}: rows, synapses, nTypes and shares as worked out by hand{detail}"))


def check_lines():
    for num, (labels, dd, aa) in LINES.items():
        if not plt.fignum_exists(num):
            results.append((False, f"line plot {num}: drawn"))
            continue
        ax = plt.figure(num).axes[0]
        lines = ax.get_lines()
        got_labels = [t.get_text() for t in ax.get_xticklabels()]
        ok = (len(lines) == 2 and [ln.get_label() for ln in lines] == ["DD", "AA"]
              and all(close(a, b) for a, b in zip(lines[0].get_ydata(), dd)) and len(lines[0].get_ydata()) == len(dd)
              and all(close(a, b) for a, b in zip(lines[1].get_ydata(), aa)) and len(lines[1].get_ydata()) == len(aa)
              and np.allclose(matplotlib.colors.to_rgb(lines[0].get_color()), COLOR_DD)
              and np.allclose(matplotlib.colors.to_rgb(lines[1].get_color()), COLOR_AA)
              and got_labels == labels and tuple(ax.get_ylim()) == (0, 100)
              and tuple(ax.get_xlim()) == (0.5, len(labels) + 0.5)
              and all(t.get_rotation() == (20 if num in (5, 6) else 0) for t in ax.get_xticklabels()))
        detail = "" if ok else (f"\n      got labels {got_labels}, lines {[ln.get_label() for ln in lines]}, "
                                f"y {[list(np.round(ln.get_ydata(), 3)) for ln in lines]}, ylim {ax.get_ylim()}")
        results.append((ok, f"line plot {num}: DD then AA, values, colours, x labels (rotated 20 in the "
                            f"olfactory plots), x and y limits{detail}"))
    for num, bars in ((1, BARS_1), (2, BARS_2)):
        if not plt.fignum_exists(num):
            results.append((False, f"stacked bars (plot {num}): drawn"))
            continue
        ax = plt.figure(num).axes[0]
        heights = [p.get_height() for p in ax.patches]
        bottoms = [p.get_y() for p in ax.patches]
        want = [h for col in zip(*bars) for h in col]
        want_b = [b for col in zip(*[np.concatenate([[0], np.cumsum(np.nan_to_num(r))[:-1]]) for r in bars])
                  for b in col]
        ok = (len(heights) == len(want) and all(close(float(a), float(b)) for a, b in zip(heights, want))
              and all(close(float(a), float(b)) for a, b in zip(bottoms, want_b))
              and [t.get_text() for t in ax.get_xticklabels()] == ["LA", "ME", "LO", "LOP", "VPN"])
        results.append((ok, f"stacked bars (plot {num}): AD, AA, DD, DA heights and where each starts, per layer"
                        + ("" if ok else f"\n      got {np.round(heights, 3).tolist()} from {np.round(bottoms, 3).tolist()}")))


def check_standin(ns, label, want):
    for row, values in want.items():
        olf = row.startswith("olf ")
        name = row[4:] if olf else row
        within = name.startswith("within ")
        layer = name[7:] if within else name
        tab = ns.get(("T_olf_within" if within else "T_olf_all") if olf else ("T_within" if within else "T_all"))
        if tab is None or layer not in list(tab["layer"]):
            results.append((False, f"{label}: {row} row present"))
            continue
        r = tab[tab["layer"] == layer].iloc[0]
        got = (r["synapses"], r["AD_pct"], r["AA_pct"], r["DD_pct"], r["DA_pct"])
        ok = all(close(float(a), float(b)) for a, b in zip(got, values))
        results.append((ok, f"{label}: {row} = {values[0]} synapses with shares as worked out by hand"
                        + ("" if ok else f" (got {[round(float(x), 3) for x in got]})")))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--script", type=Path, default=SCRIPT, help="the copy of the Fig 4E script to check")
    args = ap.parse_args()
    source = args.script.read_text(encoding="utf-8")
    report = []

    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        write_inputs(tmp)
        ns, out, err = run(source, tmp, {})
        results.append((err is None, "the script runs on the made-up per-type table" + (f": {err}" if err else "")))
        if err is None:
            check_tables(tmp / "out" / "fig4" / "syntype_x_pathway_layers")
            check_lines()
            results.append(("5 distinct cell types belong to ORN/ALPN/Kenyon_Cell/MBON" in out,
                            "olfactory types: 5 (primary_type, or cell_type when empty; blanks dropped)"))
            out_dir = tmp / "out" / "fig4" / "syntype_x_pathway_layers"
            ok = sorted(_saved) == sorted(SVG_NAMES.items()) and all((out_dir / n).exists() for n in SVG_NAMES.values())
            results.append((ok, "each plot is saved once, under its own SVG name" + ("" if ok else f" (got {_saved})")))
            info = (out_dir / "run_info.txt").read_text() if (out_dir / "run_info.txt").exists() else ""
            results.append(("DATA_SOURCE = 'csv'" in info, "run_info.txt records the input the run used"))

        # the second, small table: text kept as text, spaces trimmed, a missing count
        pd.DataFrame([dict(preType=a, postType=b, comp=c, npil=d, pre_SuperClass=e, post_SuperClass=f,
                           preSI=0.5, postSI=0.5, pre_nt="ACH", post_nt="ACH", synapses=s)
                      for a, b, c, d, e, f, s in EDGE_ROWS]).to_csv(tmp / "data" / "per_type.csv", index=False)
        ns, out, err = run(source, tmp, {})
        results.append((err is None, "the script runs on the second table" + (f": {err}" if err else "")))
        if err is None:
            for row, values in EDGE.items():
                tab = ns["T_within"] if row.startswith("within") else ns["T_all"]
                r = tab[tab["layer"] == row.split()[-1]].iloc[0]
                got = (r["synapses"], r["nTypes"], r["AD_pct"], r["AA_pct"], r["DD_pct"], r["DA_pct"])
                ok = all(close(float(a), float(b)) for a, b in zip(got, values))
                results.append((ok, f"second table, {row}: 'NA' and 'None' are type names, ' ME_L ' is ME, a missing "
                                    f"count gives NaN" + ("" if ok else f" (got {[round(float(x), 3) for x in got]})")))
        write_inputs_csv_only(tmp)

        for label, settings, want in (
                ("princeton", {SETTING_SOURCE: "DATA_SOURCE = 'princeton'\n"}, STANDIN["princeton"]),
                ("princeton + SI_updated", {SETTING_SOURCE: "DATA_SOURCE = 'princeton'\n",
                                            SETTING_SI: "SI_SOURCE = 'SI_updated'\n"}, STANDIN["SI_updated"]),
                ("buhmann", {SETTING_SOURCE: "DATA_SOURCE = 'buhmann'\n",
                             SETTING_SI: "SI_SOURCE = 'SI_updated'\n"}, STANDIN["buhmann"])):
            ns, out, err = run(source, tmp, settings)
            results.append((err is None, f"{label}: the stand-in rebuild runs" + (f": {err}" if err else "")))
            if err is None:
                check_standin(ns, label, want)

        ns, out, err = run(source, tmp, {SETTING_SOURCE: "DATA_SOURCE = 'csvv'\n"})
        results.append((err is not None and "AssertionError" in err, "a misspelled DATA_SOURCE stops the script"))
        (tmp / "data" / "per_type.csv").unlink()
        ns, out, err = run(source, tmp, {})
        results.append((err is not None and "per_type.csv" in err and "DATA_SOURCE" in err,
                        "a missing per-type table stops the script with a message naming it"))

    ok = all(r for r, _ in results)
    head = [f"Fig 4E logic check, {datetime.datetime.now().isoformat(timespec='seconds')}",
            f"script {args.script}  sha256 {hashlib.sha256(source.encode('utf-8')).hexdigest()}",
            f"tool   {Path(__file__).name}  sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}", ""]
    report = head + [("PASS  " if r else "FAIL  ") + m for r, m in results] + \
        ["", f"{'PASS' if ok else 'FAIL'}: {sum(r for r, _ in results)} of {len(results)} checks"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    # No inputs needed: it makes its own tables in a temporary folder, runs
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/figures/fig4/syntype_x_pathway_layers.py
    # on them, and writes its report to
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/outputs/tools/check_fig4e_logic/report.txt
    # To test another copy of the script, uncomment:
    # sys.argv[1:] = [
    #     "--script", "/Users/ohajyahia/PycharmProjects/mixed-polarity-paper/figures/fig4/syntype_x_pathway_layers.py",
    # ]
    main()
