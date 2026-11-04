# Larval PC1: each larval cell's place on the size-and-connectivity axis, built from the six features the larva
# has, the way step 07 builds the adult PC1. For the larval version of Figure 4D. Runs after larva_maxsi_labels.py.
#
# Features per cell (the larva has no mesh, so the adult's area_nm and size_nm are missing):
#   length              navis cable_length of the cell's skeleton (LARVA_SWC_NAVIS_DIR), read and not healed, as the
#                       adult length_nm is the cable length of the lod1 skeletons;
#   leafs               navis n_leafs on the same skeleton, as the adult swc_data.ftr;
#   in_synapses_count, out_synapses_count   synapses where the cell is post or pre, and
#   in_partners, out_partners               distinct partner cells on each side, all counted on the larval MaxSI
#                       synapse table (both cells split, no self-synapses), as step 05 counts the adult ones.
# Cells: every larval cell with a MaxSI split (the larva has no super-class, so there is no filter like step 07's four
# adult super-classes). A cell with no counted synapse on one side (none of its partners on that side was split) has a
# missing feature and is left out, as step 07 leaves out cells with a missing feature. The skeletons are not healed,
# so the step stops if one has more than one piece (none does in the larval files).
#
# PC1 as step 07: the six columns in sorted order, StandardScaler, PCA with two components, fitted on the larval
# cells only. PCA's sign is arbitrary, so it is set here: the loading on length is positive (a longer cell gets a
# larger PC1). PC1_percentile, which the adult table does not have, is the cell's rank among the larval cells (above
# 0, up to 1), the axis of the larval 4D map.
#
# Output LARVA_MAXSI_PC1_FTR: one row per cell with all six features: neuron, the six features, PC1, PC1_percentile.

#%% Imports
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import LARVA_SWC_NAVIS_DIR, LARVA_MAXSI_SUMMARY_FTR, LARVA_MAXSI_SYNAPSES_FTR, LARVA_MAXSI_PC1_FTR

import navis
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SIX = sorted(["in_partners", "in_synapses_count", "leafs", "length", "out_partners", "out_synapses_count"])

#%% Inputs
t0 = time.time()
for _p in (LARVA_MAXSI_SUMMARY_FTR, LARVA_MAXSI_SYNAPSES_FTR, LARVA_SWC_NAVIS_DIR):
    if not _p.exists():
        raise SystemExit(f"Missing {_p}; run processing/larva/larva_maxsi_labels.py first.")
summary = pd.read_feather(LARVA_MAXSI_SUMMARY_FTR)
syn = pd.read_feather(LARVA_MAXSI_SYNAPSES_FTR, columns=["pre", "post"])
assert summary.neuron.dtype == np.int64 and syn.pre.dtype == np.int64 and syn.post.dtype == np.int64
assert not (syn.pre == syn.post).any()
cells = summary.neuron.to_numpy(np.int64)

#%% Features
pairs = syn.drop_duplicates()
feat = pd.concat([syn.groupby("pre").size().rename("out_synapses_count"),
                  syn.groupby("post").size().rename("in_synapses_count"),
                  pairs.groupby("pre").size().rename("out_partners"),
                  pairs.groupby("post").size().rename("in_partners")], axis=1).reindex(cells)

rows = []
for nid in cells:
    n = navis.read_swc(LARVA_SWC_NAVIS_DIR / f"{nid}.swc")
    if n.n_trees != 1:
        raise ValueError(f"cell {nid}: skeleton has {n.n_trees} pieces")
    rows.append((nid, float(n.cable_length), int(n.n_leafs)))
skel = pd.DataFrame(rows, columns=["neuron", "length", "leafs"]).set_index("neuron")
larva = feat.join(skel)
n_all = len(larva)
larva = larva.dropna(subset=SIX)
larva[["in_partners", "in_synapses_count", "out_partners", "out_synapses_count"]] = \
    larva[["in_partners", "in_synapses_count", "out_partners", "out_synapses_count"]].astype(np.int64)
larva.index.name = "neuron"

#%% PC1
pca = PCA(n_components=2, svd_solver="full")
scores = pca.fit_transform(StandardScaler().fit_transform(larva[SIX]))
load = pca.components_[0]
sign = 1.0 if load[SIX.index("length")] > 0 else -1.0
larva["PC1"] = scores[:, 0] * sign
larva["PC1_percentile"] = larva.PC1.rank(pct=True)

out = larva[SIX + ["PC1", "PC1_percentile"]].reset_index()
out["neuron"] = out["neuron"].astype(np.int64)
LARVA_MAXSI_PC1_FTR.parent.mkdir(parents=True, exist_ok=True)
out.to_feather(LARVA_MAXSI_PC1_FTR)

print(f"navis {navis.__version__}; {time.time() - t0:.0f} s")
print(f"larval cells with a MaxSI split {n_all}; with all six features {len(out)}; left out {n_all - len(out)}")
print(f"PC1 explained variance {pca.explained_variance_ratio_[0]:.4f}; loadings " +
      ", ".join(f"{f} {v * sign:+.3f}" for f, v in zip(SIX, load)))
