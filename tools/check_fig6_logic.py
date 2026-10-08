"""Check what figures/fig6/aadd_specific_circuits.py computes, on a small made-up connectome.

The Figure 6 script needs tables this repository does not deposit (the Buhmann neuron and
synapse tables), so it cannot be run from a clean clone. This tool runs the script's own text
on 23 made-up cells (with real-sized IDs) and 156 made-up synapses between 27 pairs of cells,
written to a temporary folder together with a stand-in `config` module, and compares what the
script computes with values worked out by hand
(written below, with their arithmetic, separately from the script).

The script is run as it is, except for:
  - three lines added only to record values (after `fractionMatrixSorted` and its companions are
    sorted, after each between-type panel's edges are summed, and after its edge widths are
    set), found by their exact text; a reworded line is reported as a failure;
  - its settings (DATA_SOURCE, SI_SOURCE), replaced by exact text for the variants below;
  - matplotlib drawing to no screen (the Agg backend), and the SVGs going to the temporary folder.

Checks, on the default settings (DATA_SOURCE = 'buhmann', the tables named
NEURON_TABLE_NONP_FTR and SYNAPSE_TABLE_NONP_FTR):
  - which connections count: cells with SI >= 0.1 (an SI of exactly 0.1 is kept), a known and
    non-empty type, and both cells in the population (an empty type, a missing SI, an SI of 0.05,
    a visual projection cell whose type does not start with LC, an LC type outside
    visual_projection, a cell of an ALPN type in another class, and connections from a population
    cell to a cell outside it are each left out);
  - per type, the fraction of its output synapses that go to its own type, per connection type,
    over a denominator of all its outputs inside the population (link-compartment synapses
    included, outputs to cells outside the population not);
  - per type, the reciprocal fraction 2 x reciprocal pairs / directed same-type connections
    (a cell's connection to itself never makes a pair), and the presynaptic cells per type;
  - the order of the bars: longest first, ties in alphabetical order (the result only: on so few
    types numpy's default sort is stable too, so this cannot tell it from MATLAB's stable sort);
  - the summary lines the script prints for visual projection, ALPN and central complex: the
    mean within-type fraction, the composition, the reciprocal mean +- SEM, and the box of
    synapses per same-type connection (MATLAB's quartile rule, whiskers within 1.5 IQR); a
    population with a single same-type connection gives a SEM and spread of 0 and no crash;
  - a population with no same-type connection is skipped, with no figures, and its summary row
    is computed (cells, mean SI, synapses per different-type connection), not filled with zeros;
  - the population summary table (PopulationSummary.csv);
  - the between-type graphs: the synapse threshold applies to each pre cell, post cell, neuropil
    and connection type row (a row at exactly the threshold is drawn; two rows below it in
    different neuropils are not, although their sum is above it) before the rows are summed per
    pair of types; edge widths are scaled from 1 to 5 within a panel, 3 when all are equal;
  - the summary plot itself: each population's within-type bar and its four stacked shares
    (heights and where each segment starts) are drawn at the x position labelled with its name;
then:
  - DATA_SOURCE = 'princeton' reads NEURON_TABLE_FTR and SYNAPSE_TABLE_FTR instead (the made-up
    Princeton tables differ in one cell's SI and one extra synapse, and the results move as
    worked out below); adding SI_SOURCE = 'SI_updated' takes the SI from SI_UPDATED_FTR (which
    differs again, in another cell); with DATA_SOURCE = 'buhmann', SI_SOURCE is ignored;
  - a misspelled DATA_SOURCE stops the script, and a missing Buhmann table stops it with a message
    that names the Buhmann tables;
  - the script's box_stats and layered_positions functions on their own: quartiles and whiskers
    on two small samples, and a layered layout that ends on a graph with cycles, places every node
    in a layer and puts the first layer on top.

It does not test the rest of the drawing (colours, the per-population figures' layout, graph
node positions), the lower two rows of the summary plot (they are checked through the printed
summary lines, which use the same values), the interactive cells at the
end of the script, or anything about the real tables. A PASS means the script computes what its
code says; it says nothing about whether the figure is the right analysis.

Usage:
    python tools/check_fig6_logic.py
    python tools/check_fig6_logic.py --script path/to/another_copy.py

Needs no data. Takes a few seconds. Writes outputs/check/check_fig6_logic/report.txt.
Exits with status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import io
import logging
import math
import re
import sys
import tempfile
import types
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import OUTPUT_DIR  # noqa: E402

SCRIPT = REPO_ROOT / "figures" / "fig6" / "aadd_specific_circuits.py"
OUT_DIR = OUTPUT_DIR / "check" / "check_fig6_logic"
TOL = 1e-9

# Lines added to the script to record values: (exact anchor line, line inserted after it).
CAPTURES = [
    ("    FractionConnectionsReciprocalSorted = np.array(FractionConnectionsReciprocal_, dtype=float)[sortIdx]\n",
     "    _check['types'][Population] = dict(types=list(preTypesSorted), frac=fractionMatrixSorted.copy(), "
     "recip=FractionConnectionsReciprocalSorted.copy(), n_pre=[int(n) for n in Pre_Cells_PerTypeSorted])\n"),
    ("        agg = Tc.groupby(['pre_Type', 'post_Type'])['count'].sum().reset_index()\n",
     "        _check['edges'][(Population, comp)] = sorted((str(a), str(b), int(n)) for a, b, n in "
     "agg.itertuples(index=False))\n"),
    ("        widths = 1 + 4 * (w - w.min()) / (w.max() - w.min()) if w.max() > w.min() else np.full(len(w), 3.0)\n",
     "        _check['widths'][(Population, comp)] = [float(x) for x in widths]\n"),
]
SETTING_BUHMANN = "DATA_SOURCE = 'buhmann'\n"
SETTING_SI = "SI_SOURCE = 'neuron_table'\n"

# ---------------------------------------------------------------- the made-up connectome
# id, super_class, class, primary_type, SI (the Buhmann neuron table; the Princeton one differs
# only for cell 10, whose SI there is 0.05, and SI_updated.ftr only for cell 5, whose SI there is 0.05)
CELLS = [
    (1, "central", "ALPN", "T1", 0.5), (2, "central", "ALPN", "T1", 0.5), (3, "central", "ALPN", "T1", 0.5),
    (4, "central", "ALPN", "T2", 0.5), (5, "central", "ALPN", "T2", 0.5),
    (6, "central", "ALPN", "", 0.5),          # empty type: left out
    (7, "central", "ALPN", "T1", np.nan),     # no SI: left out
    (8, "central", "ALPN", "T3", 0.05),       # SI below 0.1: left out
    (9, "central", "ALPN", "T4", 0.5), (10, "central", "ALPN", "T4", 0.5),
    (20, "visual_projection", "LC", "LCa", 0.5), (21, "visual_projection", "LC", "LCa", 0.5),
    (22, "visual_projection", "LC", "LCb", 0.1),  # SI exactly at the threshold: kept
    (23, "visual_projection", "LC", "LCb", 0.5),
    (24, "visual_projection", "LT", "LTx", 0.5),   # visual projection, but not an LC type: not in the population
    (25, "central", "LNO", "LCc", 0.5),             # an LC type outside visual_projection: not in the population
    (11, "central", "LHN", "T1", 0.5),              # an ALPN type in another class: not in the ALPN population
    (26, "central", "MBON", "MB1", 0.5),            # in the olfactory population only
    (30, "central", "CX", "C1", 0.5), (31, "central", "CX", "C1", 0.5), (32, "central", "CX", "C2", 0.5),
    (40, "central", "Kenyon_Cell", "KCab", 0.5), (41, "central", "Kenyon_Cell", "KCg", 0.5),
]
# pre, post, neuropil, connection type, synapses (the Buhmann synapse table)
CONNECTIONS = [
    (1, 2, "X", "AA", 3), (1, 2, "Y", "DD", 1), (2, 1, "X", "AA", 2), (1, 1, "X", "AA", 1),
    (1, 3, "X", "AD", 2), (1, 4, "X", "AA", 6), (2, 5, "X", "AL", 1), (4, 5, "X", "DD", 5),
    (6, 1, "X", "AA", 10), (7, 1, "X", "AA", 10), (8, 1, "X", "AA", 10), (1, 8, "X", "AA", 10),
    (9, 10, "X", "DD", 1),
    (20, 21, "X", "AA", 2), (21, 20, "X", "AA", 2), (22, 23, "X", "DD", 6), (20, 22, "X", "DD", 7),
    (24, 20, "X", "AA", 9), (25, 20, "X", "AA", 2),
    (30, 31, "X", "DD", 3), (30, 32, "X", "AD", 25), (31, 32, "X", "AD", 19),
    (40, 41, "X", "AA", 5),                          # exactly at the olfactory threshold (5): drawn
    (41, 40, "X", "DA", 3), (41, 40, "Y", "DA", 3),  # 6 per cell pair, but 3 per row: not drawn
    (26, 40, "X", "AA", 2),
    (2, 30, "X", "AA", 4),                           # from a population cell to a cell outside it
    (11, 1, "X", "AA", 1), (11, 30, "X", "AA", 1),   # from a T1 cell outside the ALPN class
]
ID_BASE = 720575940600000000   # real-sized cell IDs: CELLS and CONNECTIONS give ID_BASE + id
PRINCETON_EXTRA = [(3, 1, "X", "AA", 1)]   # the Princeton synapse table has this one more synapse

# ---------------------------------------------------------------- expected values, by hand
# ALPN (cells 1-5, 9, 10 count; rows with cells 6, 7, 8 are dropped by the filters; 2 -> 30 and
# 11 -> 30 leave the population, and cell 11, type T1 but class LHN, is not in it):
#   T1 -> T1: AA 3 + 2 + 1 (the autapse 1 -> 1) = 6, DD 1, AD 2;  T1 -> T2: AA 6, AL 1
#   T1 outputs inside the population = 6 + 1 + 2 + 6 + 1 = 16, so AA 6/16, DD 1/16, AD 2/16
#   T2 -> T2: DD 5 of 5;  T4 -> T4: DD 1 of 1
#   T1 same-type connections (1,2), (2,1), (1,1), (1,3): one pair, 2 x 1 / 4 = 0.5; T2, T4: 0
#   bar lengths T1 0.5625, T2 1, T4 1: order T2, T4 (tie, alphabetical), T1
#   presynaptic cells: T1 {1, 2}, T2 {4}, T4 {9}
ALPN_TYPES = dict(types=["T2", "T4", "T1"],
                  frac=[[0, 1, 0, 0], [0, 1, 0, 0], [6 / 16, 1 / 16, 2 / 16, 0]],
                  recip=[0, 0, 0.5], n_pre=[1, 1, 2])
# Visual projection (cells 20-23; 24 is not an LC type, 25 is not visual_projection):
#   LCa -> LCa: AA 2 + 2 = 4; LCa -> LCb: DD 7; so LCa: AA 4/11.  LCb -> LCb: DD 6 of 6
#   LCa: (20,21), (21,20), one pair, 2/2 = 1; LCb: 0.  Order LCb (1), LCa (0.364)
VP_TYPES = dict(types=["LCb", "LCa"], frac=[[0, 1, 0, 0], [4 / 11, 0, 0, 0]], recip=[0, 1], n_pre=[1, 2])
# Central complex: C1 -> C1: DD 3; C1 -> C2: AD 25 + 19; so C1: DD 3/47; one connection, not reciprocal;
#   presynaptic C1 cells {30, 31}
CX_TYPES = dict(types=["C1"], frac=[[0, 3 / 47, 0, 0]], recip=[0], n_pre=[2])
# The summary lines (the script prints three decimals and %g for the box):
#   ALPN: mean over types AA 0.375/3 = 0.125, DD (1 + 1 + 0.0625)/3 = 0.6875, AD 0.125/3 = 0.04167;
#     sum 0.85417; shares 0.146, 0.805, 0.049, 0; reciprocal [0, 0, 0.5]: mean 0.1667,
#     SD 0.2887, SEM 0.1667; synapses per same-type connection (1,1) 1, (1,2) 4, (1,3) 2, (2,1) 2,
#     (4,5) 5, (9,10) 1 -> 1 1 2 2 4 5: midpoint quartiles at ranks 2, 3.5, 5 -> 1, 2, 4;
#     IQR 3, whiskers within [-3.5, 8.5] -> 1 and 5
#   Visual projection: AA 0.3636/2 = 0.1818, DD 1/2 = 0.5, sum 0.6818; shares 0.267, 0.733;
#     reciprocal [0, 1]: mean 0.5, SEM 0.5; per connection 2, 2, 6: quartiles at ranks 1.25, 2,
#     2.75 -> 2, 2, 5; whiskers 2 and 6
#   Central complex: DD 3/47 = 0.0638 only; one type, one connection of 3 synapses: SEM 0, box 3
SUMMARY_LINES = {
    "Visual projection": "within-type fraction 0.682; composition AA 0.267, DD 0.733, AD 0.000, DA 0.000; "
                         "reciprocal mean 0.500 +- 0.500 SEM over 2 types; synapses per same-type connection "
                         "median 2, box 2-5, whiskers 2-6 (3 connections)",
    "ALPN": "within-type fraction 0.854; composition AA 0.146, DD 0.805, AD 0.049, DA 0.000; "
            "reciprocal mean 0.167 +- 0.167 SEM over 3 types; synapses per same-type connection "
            "median 2, box 1-4, whiskers 1-5 (6 connections)",
    "Central complex": "within-type fraction 0.064; composition AA 0.000, DD 1.000, AD 0.000, DA 0.000; "
                       "reciprocal mean 0.000 +- 0.000 SEM over 1 types; synapses per same-type connection "
                       "median 3, box 3-3, whiskers 3-3 (1 connections)",
}
# Population summary: cells that appear in a kept connection, their mean SI, the mean synapses per
# same-type and per different-type (pre, post) connection.
#   Visual projection: cells 20-23, SI (0.5 + 0.5 + 0.1 + 0.5)/4 = 0.4; same (2 + 2 + 6)/3; different (20,22) 7
#   ALPN: 7 cells; same 15/6 = 2.5; different (1,4) 6 and (2,5) 1 -> 3.5
#   olfactory (ALPN, Kenyon cells, MBON): the ALPN cells and 40, 41, 26; different (1,4) 6, (2,5) 1,
#     (40,41) 5, (41,40) 3 + 3, (26,40) 2 -> 20/5 = 4
#   Kenyon cells: KCab <-> KCg only, no same-type connection: skipped; cells 40, 41; different 5 and 6 -> 5.5
#   Central complex: same 3; different (30,32) 25 and (31,32) 19 -> 22
POPULATION_SUMMARY = [
    ("Visual projection", 0.4, 4, 10 / 3, 7.0),
    ("ALPN", 0.5, 7, 2.5, 3.5),
    ("olfactory", 0.5, 10, 2.5, 4.0),
    ("Kenyon cells", 0.5, 2, math.nan, 5.5),
    ("Central complex", 0.5, 3, 3.0, 22.0),
]
# Between-type edges after the per-row threshold (5, 5, 5, 1, 20), summed per pair of types:
#   ALPN: (1,4) AA 6 kept, (2,5) AL 1 not a panel; olfactory: the same, (40,41) AA 5 kept (>= 5),
#   (41,40) DA 3 + 3 in two neuropils dropped (each row < 5), (26,40) AA 2 dropped;
#   Visual projection: (20,22) DD 7; Central complex: (30,32) AD 25 kept, (31,32) AD 19 < 20 dropped,
#   so the edge is 25, not 44; Kenyon cells: skipped, no graph
EDGES = {
    ("Visual projection", "DD"): [("LCa", "LCb", 7)],
    ("ALPN", "AA"): [("T1", "T2", 6)],
    ("olfactory", "AA"): [("KCab", "KCg", 5), ("T1", "T2", 6)],
    ("Central complex", "AD"): [("C1", "C2", 25)],
}
# Edge widths 1 + 4 (w - min) / (max - min), or 3 when all are equal: olfactory AA 5 and 6 -> 1 and 5
WIDTHS = {("Visual projection", "DD"): [3.0], ("ALPN", "AA"): [3.0], ("olfactory", "AA"): [1.0, 5.0],
          ("Central complex", "AD"): [3.0]}
# The summary plot (top two rows): x positions 2, 1, 3.2 for visual projection, ALPN, central complex,
# labelled in x order; per population the within-type bar (the mean fractions' sum s), then the AA,
# DD, AD and DA shares (mean fraction / s), stacked from 0
_s_alpn, _s_vp = 0.125 + 0.6875 + 0.125 / 3, 2 / 11 + 0.5
SUMMARY_BARS = {"ALPN": (_s_alpn, 0.125 / _s_alpn, 0.6875 / _s_alpn, (0.125 / 3) / _s_alpn, 0.0),
                "Visual projection": (_s_vp, (2 / 11) / _s_vp, 0.5 / _s_vp, 0.0, 0.0),
                "Central complex": (3 / 47, 0.0, 1.0, 0.0, 0.0)}
GRAPH_POPULATIONS = ["Visual projection", "ALPN", "olfactory", "Central complex"]
# DATA_SOURCE = 'princeton': cell 10 has SI 0.05, so T4 drops out; the extra synapse 3 -> 1 makes
#   T1 -> T1 AA 7 and T1's outputs 17, and (1,3), (3,1) a second pair: 2 x 2 / 5 = 0.8
PRINCETON_ALPN = dict(types=["T2", "T1"], frac=[[0, 1, 0, 0], [7 / 17, 1 / 17, 2 / 17, 0]],
                      recip=[0, 0.8], n_pre=[1, 3])
# ... and SI_SOURCE = 'SI_updated' (cell 10 back at 0.5, cell 5 at 0.05): T4 is back, T2 loses its only
#   same-type connection (4,5), and (2,5) AL goes too, so T1's outputs are 7 + 1 + 2 + 6 = 16
PRINCETON_SI_UPDATED_ALPN = dict(types=["T4", "T1"], frac=[[0, 1, 0, 0], [7 / 16, 1 / 16, 2 / 16, 0]],
                                 recip=[0, 0.8], n_pre=[1, 3])
# With DATA_SOURCE = 'buhmann', SI_SOURCE = 'SI_updated' is ignored: ALPN as in the default run


# ---------------------------------------------------------------- running the script
def write_tables(folder: Path) -> types.ModuleType:
    """Write the made-up tables and return a stand-in config module pointing at them."""
    data = folder / "data"
    data.mkdir()
    cells = pd.DataFrame(CELLS, columns=["root_id", "super_class", "class", "primary_type", "SI"])
    cells["root_id"] = (ID_BASE + cells["root_id"]).astype(np.int64)
    cells[["root_id", "super_class", "class"]].to_csv(data / "classification.csv", index=False)
    buhmann = cells[["root_id", "primary_type", "SI"]].reset_index(drop=True)
    princeton = buhmann.copy()
    princeton.loc[princeton["root_id"] == ID_BASE + 10, "SI"] = 0.05
    si_updated = buhmann[["root_id", "SI"]].copy()
    si_updated.loc[si_updated["root_id"] == ID_BASE + 5, "SI"] = 0.05
    buhmann.to_feather(data / "neurons_buhmann.ftr")
    princeton.to_feather(data / "neurons_princeton.ftr")
    si_updated.to_feather(data / "si_updated.ftr")

    def synapses(conns):
        rows = [(ID_BASE + pre, ID_BASE + post, npil, comp) for pre, post, npil, comp, n in conns for _ in range(n)]
        s = pd.DataFrame(rows, columns=["pre", "post", "npil", "comp"])
        s[["pre", "post"]] = s[["pre", "post"]].astype(np.int64)
        return s
    synapses(CONNECTIONS).to_feather(data / "synapses_buhmann.ftr")
    synapses(CONNECTIONS + PRINCETON_EXTRA).to_feather(data / "synapses_princeton.ftr")

    config = types.ModuleType("config")
    config.CLASSIFICATION_CSV = data / "classification.csv"
    config.NEURON_TABLE_NONP_FTR = data / "neurons_buhmann.ftr"
    config.SYNAPSE_TABLE_NONP_FTR = data / "synapses_buhmann.ftr"
    config.NEURON_TABLE_FTR = data / "neurons_princeton.ftr"
    config.SYNAPSE_TABLE_FTR = data / "synapses_princeton.ftr"
    config.SI_UPDATED_FTR = data / "si_updated.ftr"
    config.OUTPUT_DIR = folder / "outputs"
    return config


def instrument(source: str) -> str:
    for anchor, added in CAPTURES:
        if source.count(anchor) != 1:
            raise LookupError(f"line to record after occurs {source.count(anchor)} times, not once: {anchor.strip()!r}")
        source = source.replace(anchor, anchor + added)
    return source


def run(source: str, config: types.ModuleType, settings: list[tuple[str, str]], path: Path = SCRIPT) -> tuple[dict, str]:
    """Run the script's text with the stand-in config; return the recorded values and the printed text."""
    for old, new in settings:
        if source.count(old) != 1:
            raise LookupError(f"setting occurs {source.count(old)} times, not once: {old.strip()!r}")
        source = source.replace(old, new)
    check = {"types": {}, "edges": {}, "widths": {}}
    namespace = {"__file__": str(path), "__name__": "__main__", "_check": check}
    saved = sys.modules.get("config")
    sys.modules["config"] = config
    printed = io.StringIO()
    try:
        with contextlib.redirect_stdout(printed):
            exec(compile(source, str(path), "exec"), namespace)
    finally:
        plt.close("all")
        if saved is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = saved
    check["namespace"] = namespace
    return check, printed.getvalue()


def close(a, b) -> bool:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.all((np.abs(a - b) <= TOL) | (np.isnan(a) & np.isnan(b))))


def same_types(got: dict | None, want: dict) -> bool:
    return (got is not None and got["types"] == want["types"] and close(got["frac"], want["frac"])
            and close(got["recip"], want["recip"]) and got["n_pre"] == want["n_pre"])


def show(got: dict | None) -> str:
    if got is None:
        return "nothing recorded"
    return (f"types {got['types']}, fractions {np.round(got['frac'], 4).tolist()}, "
            f"reciprocal {np.round(got['recip'], 4).tolist()}, presynaptic cells {got['n_pre']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--script", type=Path, default=SCRIPT,
                        help="the copy of the Figure 6 script to test (default: the repository's)")
    args = parser.parse_args()
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

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

    if not args.script.is_file():
        results.append((False, f"the script to test exists: {args.script}"))
        finish()
    raw = args.script.read_bytes()
    say(f"Checked {datetime.datetime.now():%Y-%m-%d %H:%M}")
    say(f"Figure 6 script: {args.script} (sha256 {hashlib.sha256(raw).hexdigest()[:16]})")
    try:
        source = instrument(raw.decode("utf-8-sig"))
    except LookupError as error:
        results.append((False, f"the lines to record after were found: {error}"))
        finish()

    with tempfile.TemporaryDirectory() as tmp:
        config = write_tables(Path(tmp))

        # -- default settings
        try:
            check, printed = run(source, config, [], args.script)
        except Exception as error:  # noqa: BLE001  any failure of the script is a finding
            results.append((False, f"the script runs on the made-up tables with its default settings: "
                                   f"{type(error).__name__}: {error}"))
            finish()
        results.append((True, "the script runs on the made-up tables with its default settings"))
        try:
            for population, want in [("ALPN", ALPN_TYPES), ("Visual projection", VP_TYPES),
                                     ("Central complex", CX_TYPES), ("olfactory", ALPN_TYPES)]:
                got = check["types"].get(population)
                results.append((same_types(got, want),
                                f"{population}: bar order, within-type fractions, reciprocal fractions and "
                                f"presynaptic cells per type ({show(got)})"))
            for population, want in SUMMARY_LINES.items():
                found = re.findall(rf"  summary {re.escape(population)}: (.*)", printed)
                results.append((found == [want], f"{population}: the printed summary line "
                                                 f"({found[0] if found else 'not printed'})"))
            skipped = "Kenyon cells" not in check["types"] and re.search(
                r"Analyzing population: Kenyon cells\nWithin type\n  no connection between polarized cells of the "
                r"same type in this population \(2 of 2 cells have SI >= 0.1\); skipped", printed) is not None
            results.append((skipped, "Kenyon cells (no same-type connection) are skipped, with the message"))
            summary = check["namespace"]["T_PopulationSummary"]
            got_rows = [tuple(r) for r in summary.itertuples(index=False)]
            rows_ok = (list(summary.columns) == ["Population", "meanSI", "Cells", "SynapsesPerConnection_SameType",
                                                 "SynapsesPerConnection_DiffType"]
                       and [r[0] for r in got_rows] == [r[0] for r in POPULATION_SUMMARY]
                       and all(g[2] == w[2] and close(g[1:2] + g[3:], w[1:2] + w[3:])
                               for g, w in zip(got_rows, POPULATION_SUMMARY)))
            results.append((rows_ok, "the population summary table, including the computed row of the skipped "
                                     "Kenyon cells (" + "; ".join(f"{r[0]} {r[1]:.3g} {r[2]} {r[3]:.4g} {r[4]:.4g}"
                                                                  for r in got_rows) + ")"))
            want_edges = {(p, c): EDGES.get((p, c), []) for p in GRAPH_POPULATIONS for c in ["AD", "DA", "AA", "DD"]}
            results.append((check["edges"] == want_edges,
                            "between-type edges: threshold per row, then summed per pair of types; no graph for the "
                            "skipped population (" + "; ".join(f"{p} {c} {e}" for (p, c), e in check["edges"].items()
                                                               if e) + ")"))
            results.append((check["widths"].keys() == WIDTHS.keys()
                             and all(close(check["widths"][k], w) for k, w in WIDTHS.items()),
                            f"edge widths: scaled 1 to 5 within a panel, 3 when all edges are equal ({check['widths']})"))
            ax_top, ax_comp = check["namespace"]["ax_summary"][0, 0], check["namespace"]["ax_summary"][1, 0]
            label_at = {round(x, 6): t.get_text() for x, t in zip(ax_top.get_xticks(), ax_top.get_xticklabels())}
            bars: dict[str, list] = {}
            for ax in (ax_top, ax_comp):
                for patch in ax.patches:
                    label = label_at.get(round(patch.get_x() + patch.get_width() / 2, 6), "?")
                    bars.setdefault(label, []).append((patch.get_height(), patch.get_y()))

            def bars_right(got: list, want: tuple) -> bool:
                # one top bar from 0, then four segments, each starting where the previous one ends
                starts = [0.0, 0.0, want[1], want[1] + want[2], want[1] + want[2] + want[3]]
                return len(got) == 5 and close([h for h, _ in got], want) and close([y for _, y in got], starts)
            bars_ok = (bars.keys() == SUMMARY_BARS.keys()
                       and all(bars_right(bars[k], w) for k, w in SUMMARY_BARS.items()))
            results.append((bars_ok, "the summary plot: each population's within-type bar and stacked shares sit at its "
                                     "own x label (" + "; ".join(f"{k} " + " ".join(f"{h:.4f}" for h, _ in v)
                                                                 for k, v in bars.items()) + ")"))
            svgs = sorted(p.name for p in (config.OUTPUT_DIR / "fig6" / "aadd_specific_circuits").glob("*.svg"))
            want_svgs = sorted([f"{p}_{k}.svg" for p in GRAPH_POPULATIONS
                                for k in ("conn_SameType", "SynapsesPerSameConnection", "conn_BetweenTypes")]
                               + ["SummaryPlot.svg"])
            results.append((svgs == want_svgs, f"the SVGs written: {len(svgs)}, none for the skipped population"))

            # -- the other data sources
            for label, settings, want in [
                    ("DATA_SOURCE = 'princeton' reads NEURON_TABLE_FTR and SYNAPSE_TABLE_FTR",
                     [(SETTING_BUHMANN, "DATA_SOURCE = 'princeton'\n")], PRINCETON_ALPN),
                    ("and SI_SOURCE = 'SI_updated' takes the SI from SI_UPDATED_FTR",
                     [(SETTING_BUHMANN, "DATA_SOURCE = 'princeton'\n"), (SETTING_SI, "SI_SOURCE = 'SI_updated'\n")],
                     PRINCETON_SI_UPDATED_ALPN),
                    ("SI_SOURCE = 'SI_updated' is ignored with DATA_SOURCE = 'buhmann'",
                     [(SETTING_SI, "SI_SOURCE = 'SI_updated'\n")], ALPN_TYPES)]:
                try:
                    got = run(source, config, settings, args.script)[0]["types"].get("ALPN")
                except Exception as error:  # noqa: BLE001
                    got = None
                    label += f" ({type(error).__name__}: {error})"
                results.append((same_types(got, want), f"{label}: ALPN {show(got)}"))

            # -- settings and missing tables stop the script
            try:
                run(source, config, [(SETTING_BUHMANN, "DATA_SOURCE = 'Buhmann'\n")], args.script)
                stopped = "it ran"
            except AssertionError as error:
                stopped = "" if str(error) == "Buhmann" else f"AssertionError: {error}"
            except Exception as error:  # noqa: BLE001
                stopped = f"{type(error).__name__}: {error}"
            results.append((stopped == "", "a misspelled DATA_SOURCE ('Buhmann') stops the script"
                            + (f" ({stopped})" if stopped else "")))
            missing = types.ModuleType("config")
            missing.__dict__.update({k: v for k, v in config.__dict__.items() if k.isupper()})
            missing.NEURON_TABLE_NONP_FTR = Path(tmp) / "data" / "not_there.ftr"
            try:
                run(source, missing, [], args.script)
                message = "it ran"
            except FileNotFoundError as error:
                message = str(error)
            except Exception as error:  # noqa: BLE001
                message = f"{type(error).__name__}: {error}"
            results.append(("not_there.ftr is missing" in message and "Buhmann tables" in message,
                            f"a missing Buhmann table stops the script with a message naming them (...{message[-110:]})"))

            # -- the two helper functions on their own
            box_stats, layered_positions = check["namespace"]["box_stats"], check["namespace"]["layered_positions"]
            # 1 2 2 4 5: midpoint quartiles at ranks 1.75, 3, 4.25 -> 1.75, 2, 4.25; IQR 2.5, whiskers 1 and 5
            b = box_stats([1, 2, 2, 4, 5])
            ok_a = close([b["q1"], b["med"], b["q3"], b["whislo"], b["whishi"]], [1.75, 2, 4.25, 1, 5]) and b["fliers"] == []
            # 1 1 1 1 2 2 100: quartiles at ranks 2, 4, 6 -> 1, 1, 2; IQR 1, so the upper whisker stops at 2
            b = box_stats([1, 1, 1, 1, 2, 2, 100])
            ok_b = close([b["q1"], b["med"], b["q3"], b["whislo"], b["whishi"]], [1, 1, 2, 1, 2]) and b["fliers"] == []
            results.append((ok_a and ok_b, "box_stats: midpoint quartiles, whiskers at the furthest point within 1.5 IQR, "
                                           "outliers not drawn"))
            G = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("e", "f"), ("f", "g")])
            pos = layered_positions(G)
            ok = (set(pos) == set(G) and all(isinstance(G.nodes[n].get("layer"), int) for n in G)
                  and pos["e"][1] > pos["f"][1] > pos["g"][1])
            results.append((ok, "layered_positions: ends on a graph with a cycle, places every node in a layer, "
                                "first layer on top"))
        except Exception as error:  # noqa: BLE001  a crash in a check is a finding, not a missing report
            results.append((False, f"the checks after the first run could all be completed: "
                                   f"{type(error).__name__}: {error}"))
    finish()


if __name__ == "__main__":
    # No inputs needed: it makes its own tables in a temporary folder, runs
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/figures/fig6/aadd_specific_circuits.py
    # on them, and writes its report to
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/outputs/check/check_fig6_logic/report.txt
    # To test another copy of the script, uncomment:
    # sys.argv[1:] = [
    #     "--script", "/Users/ohajyahia/PycharmProjects/mixed-polarity-paper/figures/fig6/aadd_specific_circuits.py",
    # ]
    main()
