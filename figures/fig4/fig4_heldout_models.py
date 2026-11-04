# The values of Fig 4A and 4B: how well each model names a synapse's type (AA, AD, DA, DD) for neurons it has never
# seen. Run before models_comparison.py, which draws 4A and 4B from the two tables this script writes.
#
# Synapses: between cells of the four intrinsic super-classes, both at SI >= 0.1, of type AA, AD, DA or DD (as
# syntype_x_pc1_simple_model.py and syntype_x_features.py take them).
# Test: five cell splits (seeds 42, 1, 2, 3, 4). In each, every cell goes to the test set with probability 0.2; the
# training synapses join two training cells and the test synapses two test cells, so every test synapse joins two
# neurons the model never saw (synapses between a training and a test cell are left out). Training is balanced as the
# published scripts balance it (each type drawn to the DA count, with replacement); the test synapses keep the real
# mix of types, so each model is scored by balanced accuracy, the mean of the four types' recalls (chance 0.25).
# Models, each trained on the same balanced synapses of a split:
#   pc1              PC1 of the two cells, random forest of 100 trees with at most 100 leaves each (Fig 4A, 4D)
#   features         the eight features of the two cells (as Fig 3J), random forest of 100 trees, default settings
#   pc1_superclass   PC1 and super-class of the two cells, the capped forest
#   cell_types       for each pair of cell types, the synapse type most common among the training synapses; a test
#                    synapse whose type pair has no training synapse, or a cell with no cell type, counts as wrong
# Outputs:
#   FIG4_HELDOUT_MODELS_FTR   one row per model and split: balanced accuracy, plain accuracy, the four recalls, the
#                             number of test synapses, test cells and their cell types, and the share of test
#                             synapses with a cell type pair seen in training (the cell-type model's reach) and with a
#                             PC1 for both cells (a cell step 07 left out has none; the forest still takes the synapse)
#   FIG4_HELDOUT_CONFUSION_FTR the pc1 model's predictions against the true types, counted over the five splits (4A)

#%% Imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (NEURON_TABLE_FTR, SYNAPSE_TABLE_FTR, PCA_TABLE_FTR, FIG4_HELDOUT_MODELS_FTR,
                    FIG4_HELDOUT_CONFUSION_FTR)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from sklearn.ensemble import RandomForestClassifier

TYPES = ["AA", "AD", "DA", "DD"]
SCLASS = ["central", "optic", "visual_projection", "visual_centrifugal"]
FEATURES = ["length_nm", "area_nm", "size_nm", "out_synapses_count", "in_synapses_count", "out_partners",
            "in_partners", "leafs"]
SEEDS = (42, 1, 2, 3, 4)
TEST_SHARE = 0.2

#%% Cells and synapses
nodesG = pd.read_feather(NEURON_TABLE_FTR, columns=["neuron", "super_class", "primary_type"] + FEATURES)
nodesG = nodesG[nodesG["super_class"].isin(SCLASS)]
nodesG = nodesG.merge(pd.read_feather(PCA_TABLE_FTR)[["neuron", "PC1"]], on="neuron", how="left")
assert nodesG["neuron"].dtype == np.int64 and nodesG["neuron"].is_unique
keep = nodesG["neuron"].to_numpy(np.int64)
cell = nodesG.set_index("neuron")
cell["sc"] = cell["super_class"].map({s: i for i, s in enumerate(SCLASS)}).astype(np.int64)

reader = ipc.open_file(pa.memory_map(str(SYNAPSE_TABLE_FTR)))
parts = []
for i in range(reader.num_record_batches):
    b = reader.get_batch(i).select(["pre", "post", "comp", "SI_pre", "SI_post"]).to_pandas()
    b = b[(b["SI_pre"] >= 0.1) & (b["SI_post"] >= 0.1) & b["comp"].isin(TYPES)]
    b = b[np.isin(b["pre"].to_numpy(np.int64), keep) & np.isin(b["post"].to_numpy(np.int64), keep)]
    parts.append(b[["pre", "post", "comp"]].reset_index(drop=True))
syn = pd.concat(parts, ignore_index=True)
del parts
order = list(pd.unique(syn["comp"]))   # the order the types are drawn in when balancing
syn["type_x"] = syn["pre"].map(cell["primary_type"])
syn["type_y"] = syn["post"].map(cell["primary_type"])
print(f"cells {len(keep):,}; synapses {len(syn):,}")


#%% Models
def inputs(rows, cols):
    """The model inputs for these synapses: each column for the pre cell (_x) and the post cell (_y)."""
    out = []
    for c in cols:
        out += [rows["pre"].map(cell[c]), rows["post"].map(cell[c])]
    return np.column_stack([np.asarray(v, dtype=np.float32) for v in out])


def forest_predict(X_train, y_train, X_test, **settings):
    """100 trees (random_state 42), fitted ten at a time; each tree votes on the test synapses and is then dropped,
    so the full forest never sits in memory at once. The votes add up as the whole forest's would."""
    clf = RandomForestClassifier(n_estimators=10, random_state=42, warm_start=True, n_jobs=10, **settings)
    proba = None
    for k in range(10, 101, 10):
        clf.set_params(n_estimators=k)
        clf.fit(X_train, y_train)
        for j in range(k - 10, k):
            p = clf.estimators_[j].predict_proba(X_test)
            proba = p if proba is None else proba + p
            clf.estimators_[j] = None
    return clf.classes_[proba.argmax(1)]


def type_lookup(train, test):
    counts = (train.dropna(subset=["type_x", "type_y"]).groupby(["type_x", "type_y", "comp"]).size()
              .unstack(fill_value=0).reindex(columns=TYPES, fill_value=0))
    key = pd.MultiIndex.from_arrays([test["type_x"], test["type_y"]])
    return counts.idxmax(axis=1).reindex(key).to_numpy()


#%% Run the splits
rows, confusion = [], []
for seed in SEEDS:
    in_test = pd.Series(np.random.default_rng(seed).random(len(keep)) < TEST_SHARE, index=keep)
    pre_t, post_t = syn["pre"].map(in_test).to_numpy(), syn["post"].map(in_test).to_numpy()
    train, test = syn[~pre_t & ~post_t], syn[pre_t & post_t]
    n_da = int((train["comp"] == "DA").sum())
    balanced = pd.concat([train[train["comp"] == c].sample(n=n_da, random_state=42, replace=True) for c in order],
                         ignore_index=True)
    y_train, y_test = balanced["comp"].to_numpy(), test["comp"].to_numpy()
    preds = {
        "pc1": forest_predict(inputs(balanced, ["PC1"]), y_train, inputs(test, ["PC1"]), max_leaf_nodes=100),
        "features": forest_predict(inputs(balanced, FEATURES), y_train, inputs(test, FEATURES)),
        "pc1_superclass": forest_predict(inputs(balanced, ["PC1", "sc"]), y_train, inputs(test, ["PC1", "sc"]),
                                         max_leaf_nodes=100),
        "cell_types": type_lookup(train, test),
    }
    test_cells = keep[in_test.to_numpy()]
    split_counts = {"n_test_synapses": len(test), "n_test_cells": len(test_cells),
                    "n_test_cell_types": int(cell.loc[test_cells, "primary_type"].nunique()),
                    "share_with_type_pair": float(pd.notna(preds["cell_types"]).mean()),
                    "share_with_pc1": float((test["pre"].map(cell["PC1"]).notna()
                                             & test["post"].map(cell["PC1"]).notna()).mean())}
    for model, pred in preds.items():
        recall = {t: float(np.mean(pred[y_test == t] == t)) for t in TYPES}
        rows.append({"model": model, "seed": seed, "balanced_accuracy": float(np.mean(list(recall.values()))),
                     "accuracy": float(np.mean(pred == y_test)), **split_counts,
                     **{f"recall_{t}": v for t, v in recall.items()}})
        print(f"seed {seed} {model}: balanced accuracy {rows[-1]['balanced_accuracy']:.4f}")
    confusion.append(pd.crosstab(pd.Series(y_test, name="true"), pd.Series(preds["pc1"], name="predicted"))
                     .reindex(index=TYPES, columns=TYPES, fill_value=0))
    del balanced, train, test

results = pd.DataFrame(rows)
counts = sum(confusion)
FIG4_HELDOUT_MODELS_FTR.parent.mkdir(parents=True, exist_ok=True)
results.to_feather(FIG4_HELDOUT_MODELS_FTR)
counts.stack().rename("synapses").reset_index().to_feather(FIG4_HELDOUT_CONFUSION_FTR)
print(results.groupby("model")["balanced_accuracy"].agg(["mean", "min", "max"]).round(4).to_string())
