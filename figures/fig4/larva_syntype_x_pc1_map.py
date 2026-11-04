# The larval version of Fig 4D, beside the adult one built the same way: a random forest predicts a synapse's type
# (AA, AD, DA, DD) from the PC1 of its two cells, tested on cells it never saw, and drawn as a map over the two cells'
# PC1. Runs with SPLIT_METHOD = "maxsi" (the larva is split with MaxSI only), after processing/larva/larva_maxsi_labels.py
# and processing/larva/larva_pc1.py.
#
# Synapses: both cells at SI >= 0.1, type AA, AD, DA or DD, both cells with a PC1.
#   Larva: LARVA_MAXSI_SYNAPSES_FTR (both cells split, no self-synapses), its type built from the two compartment
#          columns, each cell's MaxSI_SI from LARVA_MAXSI_SUMMARY_FTR, PC1 from LARVA_MAXSI_PC1_FTR (six features).
#          The larva has no super-class, so every larval cell is in (the adult keeps four super-classes).
#   Adult: SYNAPSE_TABLE_FTR, cells of the four intrinsic super-classes, and a six-feature PC1 built here the way
#          larva_pc1.py builds the larval one (the same six columns of NEURON_TABLE_FTR, StandardScaler, PCA with two
#          components, the sign set so the loading on length is positive), so both animals use the same features.
# PC1 axis: each cell's PC1 as a rank among its own animal's cells with a PC1 (0 to 1, drawn as 0 to 100): the 2,842
# larval cells of LARVA_MAXSI_PC1_FTR, which include cells left out below, and the adult intrinsic cells with all six
# features. The two maps then share axes. A forest splits on order, so this gives nearly the same predictions as raw PC1.
# Model: 100 trees with at most 100 leaves each (random_state 42), as the adult 4A and 4D. Training balanced like the
# published scripts (each type drawn to the training DA count with replacement), but drawn in the fixed order AA, AD,
# DA, DD with random_state 0 to N_DRAWS - 1, and the class probabilities averaged over the N_DRAWS draws.
# Test (as figures/fig4/fig4_heldout_models.py): each cell goes to the test set with probability 0.2; training
# synapses join two training cells, test synapses two test cells. Scored by balanced accuracy (the mean of the four
# recalls, chance 0.25) and each type's recall, on the real mix of types.
#   Larva: LARVA_SEEDS (31 splits: 0 to 29, and 42), each with a two-way 95% bootstrap interval over cells; and a
#          chance level: for each split, N_SHUFFLES runs with PC1 shuffled among the larval cells (every cell keeps its
#          synapses and their types), on the same cell split, with the same balance and fit. Each split's real score is
#          compared with its own shuffled runs (the paired difference), and the real mean with every shuffled run.
#   Adult: ADULT_SEEDS (42, 1, 2, 3, 4), drawn here over the cells with synapses, so not the same splits as 4A's,
#          and on the six-feature PC1 (4A uses eight): the adult numbers here are not 4A's.
# Maps: each animal's forest fitted on all its synapses, asked for the most likely type at each point of a 200 x 200
# grid. Grid squares of 5 x 5 percentile points holding synapses of fewer than MIN_PAIRS distinct cell pairs are left
# white (too little data). The maps are fitted on all synapses, so they are not held out: on unseen larval cells only
# AA is predicted above the shuffled level (printed below). The colours are the most likely type after balancing, not
# the mix of synapses: most larval synapses are AD.
# Outputs:
#   LARVA_4D_RESULTS_FTR    one row per animal, test (real or shuffled), split and shuffle: balanced accuracy,
#                           accuracy, the four recalls, test synapses (all and per type) and test cells, and for the
#                           larva's real splits the bootstrap interval
#   LARVA_4D_CONFUSION_FTR  each animal's held-out predictions against the true types, counted over its splits
#   SVGs in OUTPUT_DIR/fig4/larva_syntype_x_pc1_map/: the two maps, both side by side, and the recalls on unseen cells

#%% Imports
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from methods.figure_text import use_arial_8
use_arial_8()   # every text in Arial 8 pt when a figure is saved
from config import (SPLIT_METHOD, NEURON_TABLE_FTR, SYNAPSE_TABLE_FTR, LARVA_MAXSI_SYNAPSES_FTR,
                    LARVA_MAXSI_SUMMARY_FTR, LARVA_MAXSI_PC1_FTR, LARVA_4D_RESULTS_FTR, LARVA_4D_CONFUSION_FTR,
                    OUTPUT_DIR)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

if SPLIT_METHOD != "maxsi":
    raise SystemExit('Set SPLIT_METHOD = "maxsi" in config.py: the larva is split with MaxSI only.')

TYPES = ["AA", "AD", "DA", "DD"]
KEY_ORDER = ["AD", "AA", "DD", "DA"]   # the order of the type names above the published 4D
COLORS = {"AA": "#8b2be2", "AD": "#9F4800", "DA": "#B3B3B3", "DD": "#F4B95A"}
SCLASS = ["central", "optic", "visual_projection", "visual_centrifugal"]
SIX = sorted(["in_partners", "in_synapses_count", "leafs", "length", "out_partners", "out_synapses_count"])
LARVA_SEEDS = list(range(30)) + [42]
ADULT_SEEDS = [42, 1, 2, 3, 4]
TEST_SHARE = 0.2
N_DRAWS = 3
N_SHUFFLES = 3
N_BOOT = 1000
GRID = 200
MASK_BIN = 20          # 20 x 20 squares of 5 percentile points
MIN_PAIRS = 10
_out_dir = OUTPUT_DIR / "fig4" / "larva_syntype_x_pc1_map"
_out_dir.mkdir(parents=True, exist_ok=True)


#%% Larval synapses and PC1
lsyn = pd.read_feather(LARVA_MAXSI_SYNAPSES_FTR, columns=["pre", "post", "pre_compartment", "post_compartment"])
lsyn["comp"] = lsyn["pre_compartment"] + lsyn["post_compartment"]
lsi = pd.read_feather(LARVA_MAXSI_SUMMARY_FTR, columns=["neuron", "MaxSI_SI"]).set_index("neuron")["MaxSI_SI"]
lpc = pd.read_feather(LARVA_MAXSI_PC1_FTR, columns=["neuron", "PC1", "PC1_percentile"]).set_index("neuron")
assert lsyn["pre"].dtype == np.int64 and lsi.index.dtype == np.int64 and lpc.index.dtype == np.int64
n_all = len(lsyn)
lsyn = lsyn[(lsyn["pre"].map(lsi) >= 0.1) & (lsyn["post"].map(lsi) >= 0.1) & lsyn["comp"].isin(TYPES)]
n_si = len(lsyn)
lsyn = lsyn[lsyn["pre"].isin(lpc.index) & lsyn["post"].isin(lpc.index)][["pre", "post", "comp"]].reset_index(drop=True)
larva = {"syn": lsyn, "pct": lpc["PC1_percentile"]}
print(f"larva: {n_all:,} synapses with both cells split; {n_si:,} with both cells at SI >= 0.1 and type AA/AD/DA/DD; "
      f"{len(lsyn):,} with both cells having a PC1, between {len(set(lsyn.pre) | set(lsyn.post)):,} cells; "
      f"PC1 percentiles over {len(lpc):,} larval cells")


#%% Adult synapses and six-feature PC1
cells = pd.read_feather(NEURON_TABLE_FTR, columns=["neuron", "super_class", "length_nm"] +
                        [c for c in SIX if c != "length"]).rename(columns={"length_nm": "length"})
cells = cells[cells["super_class"].isin(SCLASS)].dropna(subset=SIX)
assert cells["neuron"].dtype == np.int64 and cells["neuron"].is_unique
scaled = StandardScaler().fit_transform(cells[SIX])
pca = PCA(n_components=2).fit(scaled)
apc1 = pca.transform(scaled)[:, 0]
if pca.components_[0][SIX.index("length")] < 0:
    apc1 = -apc1
apct = pd.Series(apc1, index=cells["neuron"]).rank(pct=True)
keep = cells["neuron"].to_numpy(np.int64)
reader = ipc.open_file(pa.memory_map(str(SYNAPSE_TABLE_FTR)))
parts = []
for i in range(reader.num_record_batches):
    b = reader.get_batch(i).select(["pre", "post", "comp", "SI_pre", "SI_post"]).to_pandas()
    b = b[(b["SI_pre"] >= 0.1) & (b["SI_post"] >= 0.1) & b["comp"].isin(TYPES)]
    b = b[np.isin(b["pre"].to_numpy(np.int64), keep) & np.isin(b["post"].to_numpy(np.int64), keep)]
    parts.append(b[["pre", "post", "comp"]].reset_index(drop=True))
asyn = pd.concat(parts, ignore_index=True)
del parts
adult = {"syn": asyn, "pct": apct}
print(f"adult: six-feature PC1 on {len(cells):,} intrinsic cells, explained variance {pca.explained_variance_ratio_[0]:.4f}; "
      f"{len(asyn):,} synapses with both cells at SI >= 0.1, type AA/AD/DA/DD and a PC1")
for name, a in (("larva", larva), ("adult", adult)):
    share = a["syn"]["comp"].value_counts(normalize=True).reindex(TYPES)
    print(f"{name} type shares: " + ", ".join(f"{t} {share[t]:.3f}" for t in TYPES))


#%% Forest, balancing and scoring
def inputs(syn, pct):
    return np.column_stack([syn["pre"].map(pct).to_numpy(np.float32), syn["post"].map(pct).to_numpy(np.float32)])


def balanced(train, draw):
    n_da = int((train["comp"] == "DA").sum())
    return pd.concat([train[train["comp"] == t].sample(n=n_da, random_state=draw, replace=True) for t in TYPES],
                     ignore_index=True)


def forest_proba(train, pct, X_test, n_draws):
    """Class probabilities (columns in TYPES order) averaged over n_draws balanced draws of the training synapses."""
    proba = np.zeros((len(X_test), len(TYPES)))
    for draw in range(n_draws):
        b = balanced(train, draw)
        clf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=100, random_state=42, n_jobs=10)
        clf.fit(inputs(b, pct), b["comp"].to_numpy())
        assert list(clf.classes_) == TYPES
        proba += clf.predict_proba(X_test)
    return proba / n_draws


def recalls(y, pred):
    return {t: float(np.mean(pred[y == t] == t)) for t in TYPES}


def boot_ba(y, pre, post, pred, seed=0):
    codes, _ = pd.factorize(np.concatenate([pre, post]))
    cp, cq = codes[:len(pre)], codes[len(pre):]
    k = codes.max() + 1
    masks = [y == t for t in TYPES]
    oks = [(pred == t)[m] for t, m in zip(TYPES, masks)]
    rng = np.random.default_rng(seed)
    out, skipped = [], 0
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, k, k), minlength=k).astype(float)
        rw = w[cp] * w[cq]
        tot = [rw[m].sum() for m in masks]
        if min(tot) == 0:
            skipped += 1
            continue
        out.append(np.mean([(rw[m][o]).sum() / T for m, o, T in zip(masks, oks, tot)]))
    return np.percentile(out, [2.5, 97.5]), skipped


def run_split(animal, a, seed, test_kind="real", shuffle=0, pct=None):
    pct = a["pct"] if pct is None else pct
    syn = a["syn"]
    cell_ids = np.unique(np.concatenate([syn["pre"].to_numpy(), syn["post"].to_numpy()]))
    in_test = pd.Series(np.random.default_rng(seed).random(len(cell_ids)) < TEST_SHARE, index=cell_ids)
    pre_t, post_t = syn["pre"].map(in_test).to_numpy(), syn["post"].map(in_test).to_numpy()
    train, test = syn[~pre_t & ~post_t], syn[pre_t & post_t]
    pred = np.array(TYPES)[forest_proba(train, pct, inputs(test, pct), N_DRAWS).argmax(1)]
    y = test["comp"].to_numpy()
    rc = recalls(y, pred)
    row = {"animal": animal, "test": test_kind, "seed": seed, "shuffle": shuffle,
           "balanced_accuracy": float(np.mean(list(rc.values()))), "accuracy": float(np.mean(pred == y)),
           **{f"recall_{t}": v for t, v in rc.items()}, "n_test_synapses": len(test),
           **{f"n_test_{t}": int((y == t).sum()) for t in TYPES}, "n_test_cells": int(in_test.sum()),
           "n_test_cells_DA": int(len(set(test.loc[y == "DA", "pre"]) | set(test.loc[y == "DA", "post"]))),
           "ba_low": np.nan, "ba_high": np.nan, "boot_skipped": 0}
    if animal == "larva" and test_kind == "real":
        (row["ba_low"], row["ba_high"]), row["boot_skipped"] = boot_ba(y, test["pre"].to_numpy(), test["post"].to_numpy(), pred)
    cm = pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="predicted")).reindex(
        index=TYPES, columns=TYPES, fill_value=0)
    return row, cm


#%% Held-out tests
rows, confusion = [], {}
for seed in LARVA_SEEDS:
    row, cm = run_split("larva", larva, seed)
    rows.append(row)
    confusion["larva"] = confusion.get("larva", 0) + cm
    print(f"larva split {seed}: balanced accuracy {row['balanced_accuracy']:.4f} (95% interval over cells "
          f"{row['ba_low']:.4f} to {row['ba_high']:.4f}); recall " +
          ", ".join(f"{t} {row['recall_' + t]:.3f}" for t in TYPES) + f"; {row['n_test_synapses']:,} test synapses")
    for k in range(N_SHUFFLES):
        lp = larva["pct"]
        shuffled = pd.Series(np.random.default_rng(1000 + 10 * seed + k).permutation(lp.to_numpy()), index=lp.index)
        rows.append(run_split("larva", larva, seed, "shuffled", k + 1, shuffled)[0])
for seed in ADULT_SEEDS:
    row, cm = run_split("adult", adult, seed)
    rows.append(row)
    confusion["adult"] = confusion.get("adult", 0) + cm
    print(f"adult split {seed}: balanced accuracy {row['balanced_accuracy']:.4f}; recall " +
          ", ".join(f"{t} {row['recall_' + t]:.3f}" for t in TYPES) + f"; {row['n_test_synapses']:,} test synapses")
results = pd.DataFrame(rows)
LARVA_4D_RESULTS_FTR.parent.mkdir(parents=True, exist_ok=True)
results.to_feather(LARVA_4D_RESULTS_FTR)
pd.concat({k: v.stack().rename("synapses") for k, v in confusion.items()}, names=["animal"]).reset_index().to_feather(
    LARVA_4D_CONFUSION_FTR)

metrics = ["balanced_accuracy"] + [f"recall_{t}" for t in TYPES]
real_l = results[(results.animal == "larva") & (results.test == "real")]
null_l = results[(results.animal == "larva") & (results.test == "shuffled")]
real_a = results[(results.animal == "adult") & (results.test == "real")]
null_mean = null_l.groupby("seed")[metrics].mean()
for m in metrics:
    nb = np.percentile(null_l[m], [2.5, 97.5])
    k = int(np.sum(null_l[m].to_numpy() >= real_l[m].mean()))
    d = real_l.set_index("seed")[m] - null_mean[m]
    above = int(np.sum(real_l[m].to_numpy() > nb[1]))
    print(f"{m}: larva {real_l[m].mean():.4f} (sd over {len(real_l)} splits {real_l[m].std():.4f}, range "
          f"{real_l[m].min():.4f} to {real_l[m].max():.4f}); larva with PC1 shuffled {null_l[m].mean():.4f} (95% of "
          f"{len(null_l)} runs {nb[0]:.4f} to {nb[1]:.4f}; {k} of {len(null_l)} runs at or above the real mean, "
          f"p = {(k + 1) / (len(null_l) + 1):.3f}); real minus the same split's shuffled mean {d.mean():+.4f} (sd "
          f"{d.std():.4f}), above zero on {int((d > 0).sum())} of {len(d)} splits; {above} of {len(real_l)} splits above "
          f"the shuffled 97.5% point; adult {real_a[m].mean():.4f} (range over {len(real_a)} splits "
          f"{real_a[m].min():.4f} to {real_a[m].max():.4f}; six-feature PC1, not 4A's splits)")
print(f"larval bootstrap intervals that include 0.25: {int((real_l.ba_low <= 0.25).sum())} of {len(real_l)}; resamples "
      f"skipped for an empty type: up to {int(real_l.boot_skipped.max())} of {N_BOOT}; test cells with a DA synapse per "
      f"split: {int(real_l.n_test_cells_DA.min())} to {int(real_l.n_test_cells_DA.max())}")
print("plain accuracy (training is balanced, so it is far below calling every synapse its most common type): larva "
      f"{real_l.accuracy.mean():.4f}, larva shuffled {null_l.accuracy.mean():.4f}, adult {real_a.accuracy.mean():.4f}")
parity = real_l[real_l.seed.isin(ADULT_SEEDS)]
print(f"larva on the adult's 5 seeds only: balanced accuracy {parity.balanced_accuracy.mean():.4f} "
      f"({parity.balanced_accuracy.min():.4f} to {parity.balanced_accuracy.max():.4f})")
print("larval majority-type accuracy (every synapse called AD):",
      f"{(larva['syn']['comp'] == 'AD').mean():.4f}")
for k, v in confusion.items():
    print(f"{k} held-out confusion, % of each true type (rows true, columns predicted):")
    print((100 * v.div(v.sum(axis=1), axis=0)).round(1).to_string())


#%% Maps
centres = (np.arange(GRID) + 0.5) / GRID
xx, yy = np.meshgrid(centres, centres)
grid_X = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
cmap = ListedColormap([COLORS[t] for t in TYPES])
norm = BoundaryNorm(np.arange(-0.5, len(TYPES)), cmap.N)
maps = {}
for name, a in (("adult", adult), ("larva", larva)):
    proba = forest_proba(a["syn"], a["pct"], grid_X, N_DRAWS)
    Z = proba.argmax(1).reshape(GRID, GRID).astype(float)
    X = inputs(a["syn"], a["pct"])
    sq = np.minimum((X * MASK_BIN).astype(int), MASK_BIN - 1)   # square of each synapse, [pre, post]
    pairs = pd.DataFrame({"x": sq[:, 0], "y": sq[:, 1], "pre": a["syn"]["pre"].to_numpy(),
                          "post": a["syn"]["post"].to_numpy()}).drop_duplicates()
    n_pairs = pairs.groupby(["y", "x"]).size().unstack(fill_value=0).reindex(
        index=range(MASK_BIN), columns=range(MASK_BIN), fill_value=0).to_numpy()
    low = n_pairs < MIN_PAIRS                             # rows = post (y), columns = pre (x)
    mask = np.kron(low, np.ones((GRID // MASK_BIN, GRID // MASK_BIN), dtype=bool))
    Zm = np.ma.masked_where(mask, Z)
    gi = np.minimum((X * GRID).astype(int), GRID - 1)
    syn_type = Z[gi[:, 1], gi[:, 0]]
    print(f"{name} map: grid share " + ", ".join(f"{t} {np.mean(Z == i):.3f}" for i, t in enumerate(TYPES)) +
          f"; squares left white (synapses of fewer than {MIN_PAIRS} cell pairs) {low.mean():.3f}; share of synapses "
          f"whose point the map colours (in-sample) " + ", ".join(f"{t} {np.mean(syn_type == i):.3f}" for i, t in enumerate(TYPES)))
    maps[name] = Zm


def draw_map(ax, Zm, title):
    ax.pcolormesh(xx * 100, yy * 100, Zm, cmap=cmap, norm=norm, shading="auto", rasterized=True)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("PC1 pre neuron (percentile)")
    ax.set_title(title, pad=14)
    for k, t in enumerate(KEY_ORDER):
        ax.text(0.2 + 0.2 * k, 1.01, t, color=COLORS[t], transform=ax.transAxes, ha="center", va="bottom")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


for name, title in (("adult", "Adult"), ("larva", "Larva")):
    fig = plt.figure(figsize=(2.4, 2.6))
    ax = fig.add_axes([0.22, 0.16, 0.72, 0.66])
    draw_map(ax, maps[name], title)
    ax.set_ylabel("PC1 post neuron (percentile)")
    fig.savefig(_out_dir / f"decision_map_pc1_percentile_{name}.svg", dpi=600)
    plt.show()

fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.6), sharex=True, sharey=True)
draw_map(axes[0], maps["adult"], "Adult")
draw_map(axes[1], maps["larva"], "Larva")
axes[0].set_ylabel("PC1 post neuron (percentile)")
fig.tight_layout()
fig.savefig(_out_dir / "decision_map_pc1_percentile_adult_larva.svg", dpi=600)
plt.show()


#%% Recalls on unseen cells: adult and larva, with the larva's PC1-shuffled range
labels = ["balanced\naccuracy"] + TYPES
fig, ax = plt.subplots(figsize=(2.6, 2.2))
for i, m in enumerate(metrics):
    nb = np.percentile(null_l[m], [2.5, 97.5])
    ax.fill_between([i - 0.3, i + 0.3], nb[0], nb[1], color="#dddddd", lw=0, zorder=0)
    for dx, df, face in ((-0.12, real_a, "black"), (0.12, real_l, "white")):
        ax.errorbar(i + dx, df[m].mean(), yerr=[[df[m].mean() - df[m].min()], [df[m].max() - df[m].mean()]],
                    fmt="o", ms=4, mfc=face, mec="black", ecolor="black", elinewidth=0.8, capsize=0)
ax.axhline(0.25, ls="--", lw=0.6, color="grey")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
ax.set_ylabel("Recall on unseen cells")
ax.set_xlim(-0.6, len(labels) - 0.4)
ax.plot([], [], "o", ms=4, mfc="black", mec="black", label="adult")
ax.plot([], [], "o", ms=4, mfc="white", mec="black", label="larva")
ax.fill_between([], [], [], color="#dddddd", label="larva, PC1 shuffled")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.3,
          columnspacing=0.8, handlelength=1.2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(_out_dir / "recall_unseen_cells_adult_larva.svg", dpi=300)
plt.show()
