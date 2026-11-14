# -*- coding: utf-8 -*-
"""
Supp Fig. 2-S1 A: each cell's SI on the synapses with proofread partners (the SI of the rest of the paper) against
its SI on all its synapses, including those whose partners are small segments not connected to any proofread neuron
("twigs").

Both SIs are MaxSI SIs with self-synapses left out, so the script runs only with SPLIT_METHOD = "maxsi":
  x  the cell's MaxSI SI (SI_UPDATED_FTR, written by step 09);
  y  the SI of the same chain run on all the cell's synapses in NEUROPIL_SYNAPSE_CSV: the skeleton (upload_swc,
     navis.heal_skeleton), the synapses attached within 10,000 nm (attach_synapses_princeton), the SI of every
     candidate cut (calculate_node_SI), and the cut with the highest SI (pandas idxmax: on a tie, the first node), as
     in step 09. The cut is chosen again on the twig-inclusive synapses.
Cells: a random sample of N_CELLS cells (random_state SEED) of the cells with a MaxSI SI, every super class.

NEUROPIL_SYNAPSE_CSV has no header row. Its 14 columns: a synapse ID of its own, the presynaptic point (1-3), a third
point (4-6), the postsynaptic point (7-9), size (10), the presynaptic and postsynaptic root IDs (11, 12; full 18-digit
IDs) and the neuropil (13). Its rows between two proofread neurons are the Princeton synapses, one to one; the
script checks this for every sampled cell: the MaxSI chain on the cell's rows whose partner is a proofread neuron
(NEURONS_CSV) must give back its MaxSI SI, its cut node (MAXSI_LABEL_BATCH_DIR) and step 09's attached presynaptic and
postsynaptic counts. It stops if any cell fails this, or if any cell's split fails.

Writes to OUTPUT_DIR / fig2 / SI_x_twigs: SI_twigs.svg, si_twigs_cells.csv (one row per cell) and
si_twigs_summary.txt. Reads the 22.8 GB CSV in blocks, keeping only the rows of the sampled cells (about 7 GB of
memory); about 20 minutes for 10,000 cells.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (METHODS_DIR, OUTPUT_DIR, NEUROPIL_SYNAPSE_CSV, NEURONS_CSV, SI_UPDATED_FTR,
                    MAXSI_LABEL_BATCH_DIR, NEURON_TABLE_FTR, SPLIT_METHOD)
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
from node_cut_si import calculate_node_SI

import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr, spearmanr

mpl.rcParams['svg.fonttype'] = 'none'   # SVG text stays editable
mpl.rcParams['font.family'] = 'Arial'
from methods.figure_text import use_arial_8
use_arial_8()   # every text in Arial 8 pt when a figure is saved

if SPLIT_METHOD != "maxsi":
    raise SystemExit('Supp 2-S1 A compares MaxSI SIs: run it with SPLIT_METHOD = "maxsi"')

N_CELLS = 10_000
SEED = 0
POLAR = 0.1   # the paper's SI cut-off for a polarized cell
INTRINSIC = ['central', 'optic', 'visual_projection', 'visual_centrifugal']   # the paper's analysis set

#%% The cells: the MaxSI SI and the cut node of every cell step 09 split
si = pd.read_feather(SI_UPDATED_FTR)[["root_id", "SI"]].rename(columns={"root_id": "neuron", "SI": "SI_maxsi"})
cut = pd.concat([pd.read_feather(f, columns=["neuron", "best_node_id", "MaxSI_SI", "n_attached_pre",
                                             "n_attached_post"])
                 for f in sorted(MAXSI_LABEL_BATCH_DIR.glob("summary_batch_*.ftr"))], ignore_index=True)
assert si["neuron"].dtype == np.int64 and si["neuron"].is_unique and cut["neuron"].is_unique
ref = si.merge(cut, on="neuron", how="inner", validate="1:1")
assert np.array_equal(ref["SI_maxsi"].to_numpy(), ref["MaxSI_SI"].to_numpy())
sample = ref.sample(n=N_CELLS, random_state=SEED).sort_values("neuron").reset_index(drop=True)
ids = np.sort(sample["neuron"].to_numpy())
proofread = np.sort(pd.read_csv(NEURONS_CSV, usecols=["root_id"], dtype={"root_id": np.int64})["root_id"].to_numpy())
print(f"{len(ref):,} cells with a MaxSI SI; {len(sample):,} sampled")

#%% Their synapses in the twig-inclusive table (every row with a sampled cell at either end, self-synapses out)
t0 = time.time()
COLS = [f"c{i}" for i in range(14)]
NAMES = {"c1": "pre_x", "c2": "pre_y", "c3": "pre_z", "c7": "post_x", "c8": "post_y", "c9": "post_z",
         "c10": "size", "c11": "pre", "c12": "post"}
reader = pacsv.open_csv(NEUROPIL_SYNAPSE_CSV,
                        read_options=pacsv.ReadOptions(column_names=COLS, block_size=1 << 27),
                        convert_options=pacsv.ConvertOptions(include_columns=list(NAMES),
                                                             column_types={c: pa.int64() for c in NAMES}))
keep, n_rows, start = [], 0, 0
for batch in reader:
    for c in ("c11", "c12"):
        assert batch.column(c).null_count == 0, "an empty root ID"
    d = batch.to_pandas().rename(columns=NAMES)
    d.insert(0, "synapse_id", np.arange(start, start + len(d), dtype=np.int64))   # the row's number in the file
    start += len(d)
    m = ((np.isin(d["pre"].to_numpy(), ids) | np.isin(d["post"].to_numpy(), ids))
         & (d["pre"].to_numpy() != d["post"].to_numpy()))
    if m.any():
        keep.append(d[m])
allsynapses = pd.concat(keep, ignore_index=True)
assert allsynapses["pre"].min() >= 720575940 * 10**9 and allsynapses["post"].min() >= 720575940 * 10**9, \
    "root IDs must be full 18-digit IDs"
print(f"{start:,} rows read; {len(allsynapses):,} with a sampled cell ({time.time() - t0:.0f} s)")

#%% The MaxSI split of each cell, on all its synapses and on those with proofread partners
def maxsi_split(neuron_id, synapses):
    """Step 09's MaxSI chain: (SI of the best cut, its node, the attached connectors)."""
    swc = upload_swc(int(neuron_id))
    if swc is None:
        raise OSError(f"no readable skeleton for {neuron_id}")
    neuron = attach_synapses_princeton(navis.heal_skeleton(swc), synapses, progress=False)
    node_SI = calculate_node_SI(neuron, fisher=False).dropna(subset=["SI"])
    best = node_SI.loc[node_SI["SI"].idxmax()]
    cell = neuron[0] if isinstance(neuron, navis.NeuronList) else neuron
    return float(best["SI"]), np.int64(best["node_id"]), cell.connectors


by_pre = allsynapses.groupby("pre").indices
by_post = allsynapses.groupby("post").indices
none = np.array([], dtype=np.int64)
rows = []
for k, r in enumerate(sample.itertuples(index=False)):
    if k % 500 == 0:
        print(f"{k:,} / {len(sample):,} cells ({time.time() - t0:.0f} s)")
    nid = np.int64(r.neuron)
    mine = allsynapses.iloc[np.union1d(by_pre.get(nid, none), by_post.get(nid, none))]
    partner = np.where(mine["pre"].to_numpy() == nid, mine["post"].to_numpy(), mine["pre"].to_numpy())
    twig = ~np.isin(partner, proofread)
    si_twigs, cut_twigs, conn = maxsi_split(nid, mine)
    si_pr, cut_pr, conn_pr = maxsi_split(nid, mine[~twig]) if twig.any() else (si_twigs, cut_twigs, conn)
    types = conn_pr["type"].astype(str)
    assert np.isclose(si_pr, r.SI_maxsi, rtol=0, atol=1e-12) and cut_pr == r.best_node_id \
        and (types == "pre").sum() == r.n_attached_pre and (types == "post").sum() == r.n_attached_post, \
        f"{nid}: its rows with proofread partners do not give back its MaxSI split"
    rows.append({"neuron": nid, "SI_maxsi": r.SI_maxsi, "SI_twigs": si_twigs, "cut_moved": cut_twigs != r.best_node_id,
                 "synapses": len(mine), "twig_synapses": int(twig.sum()), "attached": len(conn)})
cells = pd.DataFrame(rows)
print(f"done ({time.time() - t0:.0f} s)")

#%% Numbers for the legend, for all cells and by super class (the panel shows all cells, as published)
sc = pd.read_feather(NEURON_TABLE_FTR, columns=["neuron", "super_class"])
assert sc["neuron"].dtype == np.int64 and sc["neuron"].is_unique
cells = cells.merge(sc, on="neuron", how="left", validate="1:1")
d = cells["SI_twigs"] - cells["SI_maxsi"]
share = cells["twig_synapses"] / cells["synapses"]
up = int(((cells["SI_maxsi"] < POLAR) & (cells["SI_twigs"] >= POLAR)).sum())
down = int(((cells["SI_maxsi"] >= POLAR) & (cells["SI_twigs"] < POLAR)).sum())
r_p = pearsonr(cells["SI_maxsi"], cells["SI_twigs"]).statistic
summary = [
    f"cells {len(cells):,} (random_state {SEED}, of {len(ref):,} with a MaxSI SI); every cell's rows with proofread "
    "partners give back its MaxSI SI, cut node and attached counts",
    f"twig share of a cell's synapses: median {share.median():.1%}, IQR {share.quantile(.25):.1%} to "
    f"{share.quantile(.75):.1%}; attached within 10,000 nm {cells['attached'].sum() / cells['synapses'].sum():.2%}",
    f"Pearson r {r_p:.3f}, Spearman {spearmanr(cells['SI_maxsi'], cells['SI_twigs']).statistic:.3f}",
    f"SI with twigs minus SI: median {d.median():+.4f}, IQR {d.quantile(.25):+.4f} to {d.quantile(.75):+.4f}; "
    f"|change| > 0.05 {(d.abs() > 0.05).mean():.1%}, > 0.1 {(d.abs() > 0.1).mean():.1%}",
    f"cells crossing SI {POLAR}: {down:,} of {int((cells['SI_maxsi'] >= POLAR).sum()):,} polarized cells fall below, "
    f"{up:,} of {int((cells['SI_maxsi'] < POLAR).sum()):,} mixed cells rise to it or above",
    f"cut moved: {int(cells['cut_moved'].sum()):,} cells",
]
groups = [("all cells", cells.index == cells.index), ("intrinsic super classes", cells["super_class"].isin(INTRINSIC)),
          ("sensory", cells["super_class"] == "sensory"),
          ("other super classes", ~cells["super_class"].isin(INTRINSIC + ["sensory"]))]
for name, m in groups:
    x = cells[m]
    dx = x["SI_twigs"] - x["SI_maxsi"]
    a, b = int((x["SI_maxsi"] >= POLAR).sum()), int((x["SI_twigs"] >= POLAR).sum())
    summary.append(
        f"{name}: {len(x):,} cells; Pearson r {pearsonr(x['SI_maxsi'], x['SI_twigs']).statistic:.3f}; median |change| "
        f"{dx.abs().median():.4f}, |change| <= 0.05 {(dx.abs() <= 0.05).mean():.1%}; SI >= {POLAR}: {a:,} -> {b:,} "
        f"({a / len(x):.2%} -> {b / len(x):.2%}); fall below {int(((x['SI_maxsi'] >= POLAR) & (x['SI_twigs'] < POLAR)).sum()):,}, "
        f"rise {int(((x['SI_maxsi'] < POLAR) & (x['SI_twigs'] >= POLAR)).sum()):,}")
near = (cells["SI_maxsi"] - POLAR).abs() <= 0.02
crossed = (cells["SI_maxsi"] >= POLAR) != (cells["SI_twigs"] >= POLAR)
summary.append(f"of the {int(crossed.sum()):,} cells crossing SI {POLAR}, {int((crossed & near).sum()):,} start within 0.02 of it")
print("\n".join(summary))

#%% Figure: cells per hexagon (log scale), as 10,000 points would print as one block where the low SIs crowd
from matplotlib.ticker import MultipleLocator
fig, ax = plt.subplots(figsize=(2.2, 2.2))
lim = (0, max(cells["SI_maxsi"].max(), cells["SI_twigs"].max()) * 1.02)
hb = ax.hexbin(cells["SI_maxsi"], cells["SI_twigs"], gridsize=70, extent=(0, lim[1], 0, lim[1]), bins="log",
               mincnt=1, cmap="viridis", linewidths=0, zorder=2)
ax.plot(lim, lim, color="0.5", lw=0.5, zorder=1)
ax.axvline(POLAR, color="0.4", lw=0.5, ls=(0, (3, 2)), zorder=3)
ax.axhline(POLAR, color="0.4", lw=0.5, ls=(0, (3, 2)), zorder=3)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect("equal")
for a in (ax.xaxis, ax.yaxis):
    a.set_major_locator(MultipleLocator(0.2))
ax.tick_params(width=0.5, length=2.5)
for s_ in ("left", "bottom"):
    ax.spines[s_].set_linewidth(0.5)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.set_xlabel("SI, proofread partners only")
ax.set_ylabel("SI, incl. twig partners")
cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, shrink=0.6)
cb.set_label("Cells per bin", labelpad=2)
cb.ax.tick_params(length=2, width=0.5)
cb.outline.set_linewidth(0.5)
ax.text(0.04, 0.96, f"r = {r_p:.2f}\nn = {len(cells):,}\n{up + down:,} cross SI {POLAR}",
        transform=ax.transAxes, ha="left", va="top")

_out_dir = OUTPUT_DIR / "fig2" / "SI_x_twigs"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "SI_twigs.svg", bbox_inches="tight")
cells.to_csv(_out_dir / "si_twigs_cells.csv", index=False)
(_out_dir / "si_twigs_summary.txt").write_text("\n".join(summary) + "\n")
