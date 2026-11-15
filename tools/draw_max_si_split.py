"""Draw cells split at their maximum-SI cut (MaxSI): dendrite in orange, axon in purple.

For each cell the tool reads its skeleton from SWC_DIR, heals it (navis.heal_skeleton) and attaches the cell's
synapses from SYNAPSE_TABLE_RAW_FTR (methods_all.attach_synapses_princeton), as step 09 does. It then cuts the cell
with methods/split_max_si.py and draws the two parts with navis.plot2d. The title gives the cut's SI and each side's
canonical share: the dendrite's share of postsynapses and the axon's share of presynapses, with the synapses on the
cut node on neither side. The shares count the synapses attached to the skeleton, so they can differ by a few
synapses from the neuron table's, which counts rows of the synapse table.

Before drawing a cell the tool checks the cut, and stops on any difference:
  - against step 09's rule, run here on the same cell: node_cut_si.calculate_node_SI and the highest SI, the first
    node on a tie. The cut node and the side counts must be the same, and the axon must be the side whose
    presynaptic fraction is above the whole cell's. A cell step 09 would refuse (a side with no synapse, or two sides
    with the same presynaptic fraction, numpy isclose) is not drawn;
  - against step 09's saved run, if its MaxSI batches are in MAXSI_LABEL_BATCH_DIR (or the folder given with
    --batches): the cut node, SI and side counts of step 09's summary. Without them it says it did not compare.
The paper's SI leaves out synapses from a cell to itself, so the tool also stops if the raw table has any for a cell
it draws (RAW_TABLE_KEEP_SELF_SYNAPSES must be False in config.py), if a cell has no skeleton under SWC_DIR or no
synapse in the raw table, or if a cell is in step 09's summaries more than once.

    python tools/draw_max_si_split.py 720575940628463468 720575940633095137
    python tools/draw_max_si_split.py 720575940628463468 --view -90 -90 -30
    python tools/draw_max_si_split.py 720575940628463468 --batches data/intermediate_rebuild_maxsi/maxsi_labels/batches

--view sets the camera for every cell: elevation, azimuth and roll in degrees, as matplotlib's 3D axes take them
(default -90 -90 0).
Writes outputs/tools/draw_max_si_split/<cell>_max_si_split.svg (text kept as text) and summary.csv there.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import config  # noqa: E402
from methods.figure_text import use_arial_8  # noqa: E402
sys.path.insert(0, str(config.METHODS_DIR))
from methods_all import upload_swc, attach_synapses_princeton  # noqa: E402
from node_cut_si import calculate_node_SI  # noqa: E402
from split_max_si import AXON_COLOR, DENDRITE_COLOR, split_max_si  # noqa: E402
import navis  # noqa: E402

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
use_arial_8()

OUT_DIR = config.OUTPUT_DIR / "tools" / "draw_max_si_split"
RAW = config.SYNAPSE_TABLE_RAW_FTR
COLUMNS = ["synapse_id", "pre_x", "pre_y", "pre_z", "post_x", "post_y", "post_z", "size", "pre", "post"]
SIDES = ["side_1_pre", "side_1_post", "side_2_pre", "side_2_post"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("cells", nargs="+", type=int, help="cell IDs (root IDs)")
ap.add_argument("--view", nargs=3, type=float, default=[-90.0, -90.0, 0.0], metavar=("ELEV", "AZIM", "ROLL"),
                help="camera angles in degrees (default -90 -90 0)")
ap.add_argument("--batches", type=Path, default=config.MAXSI_LABEL_BATCH_DIR,
                help="step 09's MaxSI batch folder, to compare the cuts with (default MAXSI_LABEL_BATCH_DIR)")
args = ap.parse_args()
ids = [np.int64(c) for c in args.cells]
OUT_DIR.mkdir(parents=True, exist_ok=True)

#%% The cells' skeletons and synapses, without self-synapses
local = {int(p.stem) for p in Path(config.SWC_DIR).glob("*/*.swc") if p.stem.isdigit()}
missing = [int(c) for c in ids if int(c) not in local]
if missing:
    raise SystemExit(f"no skeleton under SWC_DIR for {missing}")
syn = (ds.dataset(str(RAW), format="feather")
       .to_table(columns=COLUMNS, filter=ds.field("pre").isin(ids) | ds.field("post").isin(ids)).to_pandas())
if syn["pre"].dtype != np.int64 or syn["post"].dtype != np.int64:
    raise SystemExit("pre and post must be int64 cell IDs")
n_self = int((syn["pre"] == syn["post"]).sum())
if n_self:
    raise SystemExit(f"{n_self:,} self-synapses for these cells in {RAW.name}; the paper's "
                     "SI leaves them out (set RAW_TABLE_KEEP_SELF_SYNAPSES = False in config.py)")

#%% Step 09's cuts, if its batches are there
summaries = sorted(Path(args.batches).glob("summary_batch_*.ftr")) if Path(args.batches).is_dir() else []
if summaries:
    step09 = pd.concat([ds.dataset(str(p), format="feather").to_table(filter=ds.field("neuron").isin(ids)).to_pandas()
                        for p in summaries])
    if step09["neuron"].duplicated().any():
        raise SystemExit(f"cells in more than one of step 09's summaries in {args.batches}: "
                         f"{sorted(step09.loc[step09['neuron'].duplicated(), 'neuron'].tolist())}")
    step09 = step09.set_index("neuron")
    print(f"comparing with step 09's MaxSI summaries in {args.batches}")
else:
    step09 = None
    print(f"no step 09 MaxSI summaries in {args.batches}: the cuts are checked against step 09's rule only")


def stop(nid, what):
    raise SystemExit(f"{nid}: {what}; not drawn")


#%% Cut, check and draw each cell
rows = []
for nid in ids:
    mine_syn = syn.loc[(syn["pre"] == nid) | (syn["post"] == nid)]
    if len(mine_syn) == 0:
        stop(nid, f"no synapse in {RAW.name}")
    swc = upload_swc(nid)
    if swc is None:
        stop(nid, "navis could not read its skeleton")
    cell = attach_synapses_princeton(navis.heal_skeleton(swc), mine_syn)
    cell = cell[0] if isinstance(cell, navis.NeuronList) else cell
    parts, m = split_max_si(cell, return_metrics=True)

    # step 09's rule on the same cell
    node_SI = calculate_node_SI(cell, fisher=False).dropna(subset=["SI"])
    if len(node_SI) == 0:
        stop(nid, "no cut with an SI")
    best = node_SI.loc[node_SI["SI"].idxmax()]
    s1, s2 = (int(best["side_1_pre"]), int(best["side_1_post"])), (int(best["side_2_pre"]), int(best["side_2_post"]))
    if min(sum(s1), sum(s2)) == 0:
        stop(nid, "the cut leaves a side with no synapse (step 09 refuses this cell)")
    f1, f2, whole = s1[0] / sum(s1), s2[0] / sum(s2), (s1[0] + s2[0]) / (sum(s1) + sum(s2))
    if np.isclose(f1, f2):
        stop(nid, "both sides have the same presynaptic fraction (step 09 refuses this cell)")
    axon, dend = (s1, s2) if f1 > whole else (s2, s1)
    rule = (int(best["node_id"]), axon[0], sum(axon), dend[1], sum(dend))
    mine = (int(m["max_si_node_id"]), m["axon_canonical_synapses"], m["axon_total_synapses"],
            m["dendrite_canonical_synapses"], m["dendrite_total_synapses"])
    if mine != rule:
        stop(nid, f"cut node or side counts differ from step 09's rule (here {mine}, rule {rule})")
    row = {"neuron": nid, "cut_node": mine[0], "SI": m["max_si"], "axon_pre": mine[1], "axon_synapses": mine[2],
           "dendrite_post": mine[3], "dendrite_synapses": mine[4], "compared_with_saved_step09": False}

    # step 09's saved run
    if step09 is not None:
        if nid not in step09.index:
            stop(nid, f"not in step 09's summaries in {args.batches}")
        s = step09.loc[nid]
        ax1 = int(s["axon_side"]) == 1
        saved = (int(s["best_node_id"]), int(s["side_1_pre"] if ax1 else s["side_2_pre"]),
                 int(s["side_1_pre"] + s["side_1_post"] if ax1 else s["side_2_pre"] + s["side_2_post"]),
                 int(s["side_2_post"] if ax1 else s["side_1_post"]),
                 int(s["side_2_pre"] + s["side_2_post"] if ax1 else s["side_1_pre"] + s["side_1_post"]))
        if mine != saved or not np.isclose(m["max_si"], s["MaxSI_SI"], rtol=0, atol=1e-12):
            stop(nid, f"cut node, side counts or SI differ from step 09's saved run (here {mine}, SI {m['max_si']}; "
                      f"step 09 {saved}, SI {s['MaxSI_SI']})")
        row["compared_with_saved_step09"] = True
    rows.append(row)

    fig, ax = navis.plot2d(parts, connectors=False, method="3d", colors=[DENDRITE_COLOR, AXON_COLOR],
                           linewidth=0.5)
    ax.elev, ax.azim, ax.roll = args.view
    ax.set_facecolor("white")
    fig.patch.set_alpha(1)
    ax.patch.set_alpha(1)
    ax.set_title(f"{nid} | MaxSI SI {m['max_si']:.4f}\n"
                 f"Dendrite: {m['dendrite_canonical_percent']:.1f}% | Axon: {m['axon_canonical_percent']:.1f}%")
    fig.savefig(OUT_DIR / f"{nid}_max_si_split.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"{nid}: cut node {row['cut_node']}, SI {m['max_si']:.6f}, dendrite {row['dendrite_post']}/"
          f"{row['dendrite_synapses']}, axon {row['axon_pre']}/{row['axon_synapses']}; as step 09's rule"
          + (" and its saved run" if row["compared_with_saved_step09"] else ""))

pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False)
print(f"wrote {len(rows)} drawings and summary.csv to {OUT_DIR}")
