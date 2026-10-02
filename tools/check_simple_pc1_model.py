"""Check figures/fig4/syntype_x_pc1_simple_model.py against an independent rebuild of its model.

The figure script trains a random forest that predicts a synapse's type (AA, AD, DA, DD)
from PC1 of its two cells. This tool rebuilds the same model in its own code, from the
derived tables, following the script's procedure:

  1. synapses with SI_pre >= 0.1 and SI_post >= 0.1 and a type in AA/AD/DA/DD;
  2. both cells in the four intrinsic super-classes (central, optic, visual_projection,
     visual_centrifugal); each cell gets its PC1 from PCA_TABLE_FTR, PC1_x for the
     presynaptic cell and PC1_y for the postsynaptic one (cells with no PC1 keep NaN,
     as in the script);
  3. every type resampled with replacement to the number of DA synapses
     (random_state=42), in the order the types first appear;
  4. a stratified 80/20 train/test split (random_state=42);
  5. RandomForestClassifier(n_estimators=100, max_leaf_nodes=100, random_state=42)
     on PC1_x and PC1_y.

It prints the counts after each step, the test accuracy, the confusion matrix (percent of
each true type, which is what the script plots) and the feature importances.

--run DIR compares the rebuild with a run of the figure script made by
tools/run_figure.py. It stops before the rebuild if DIR is missing a file or the run did
not finish (run_status.txt), and it prints a warning if the figure script or one of the
three tables it reads changed after the run (by file date). It then compares: the accuracy
printed in DIR/stdout.log (exactly); the 16 percentages written in the confusion-matrix
SVG, placed by their position under the axis labels, which must read AA, AD, DA, DD (to
0.005, since the SVG shows two decimals); and the two feature importances, measured from
the bar lengths in the feature-importance SVG (to --tol-importance; the red bar is PC1_x
and the cyan bar PC1_y, as the script colours them; the two must sum to 1). It prints PASS
or FAIL, exits with status 1 on FAIL, and counts an unreadable SVG as FAIL.

A PASS means the figure script computes what its code says, on the same tables. It does
not test the tables, and it does not test whether the model is statistically sound: the
script resamples before splitting and splits by synapse, so copies of rows and synapses of
the same cell pair can sit on both sides of the split.

--variants repeats the rebuild with one change each: balancing with random_state=7 (chance
alone), PC1 sign flipped (the sign of a principal component is arbitrary, though the forest
can break ties differently) and synapses of cells without PC1 left out. It prints how much
the accuracy and each confusion-matrix cell move. Three runs give only a rough idea of the
noise; the split and forest seeds are not varied.

--earlier DIR (with --run) compares the run with an earlier run's SVGs, cell by cell. It is
information only: it never changes the exit status, and an earlier SVG that cannot be read
is reported and skipped. The earlier accuracy is taken as the mean of its confusion-matrix
diagonal, which equals the accuracy when every type has the same number of test synapses
(true for this script up to one synapse, since balanced data are split with
stratification), within the rounding of the two-decimal percentages. With --variants it
also says whether the differences are larger than the variants' movement.

Usage:
    python tools/run_figure.py figures/fig4/syntype_x_pc1_simple_model.py --out outputs/check/simple_model
    python tools/check_simple_pc1_model.py --run outputs/check/simple_model
    python tools/check_simple_pc1_model.py --run outputs/check/simple_model --variants \\
        --earlier path/to/earlier_svgs --report outputs/check/simple_model/check_report.txt

Reads only the five needed columns of the synapse table, batch by batch. Memory and time
were measured on macOS on 2 Oct 2026 and are given in the README. The result is exact only
with the pinned library versions (scikit-learn 1.5.2, pandas, numpy, matplotlib): other
versions can resample or split differently, or write the SVGs differently.
"""
import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from config import NEURON_TABLE_FTR, PCA_TABLE_FTR, SYNAPSE_TABLE_FTR  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.ipc as ipc  # noqa: E402
import sklearn  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

FIGURE_SCRIPT = REPO_ROOT / "figures" / "fig4" / "syntype_x_pc1_simple_model.py"
INTRINSIC = ["central", "optic", "visual_projection", "visual_centrifugal"]
TYPES = ["AA", "AD", "DA", "DD"]
SYNAPSE_COLUMNS = ["pre", "post", "comp", "SI_pre", "SI_post"]
CM_SVG = "confusion_matrix_filtered_pc_usingpc1_not_simpler_non_balanced_princeton.svg"
FI_SVG = "feature_importance_filtered_pc_usingpc1_not_simpler_non_balanced_princeton.svg"
RUN_FILES = ["run_status.txt", "stdout.log", CM_SVG, FI_SVG]
BAR_COLOURS = {"#ff0000": "PC1_x", "#00ffff": "PC1_y"}
SVG = "{http://www.w3.org/2000/svg}"
VARIANTS = {"seed7": "balancing with random_state=7 instead of 42",
            "flip": "PC1 sign flipped",
            "dropnan": "synapses of cells without PC1 left out"}

lines = []


class Unreadable(Exception):
    """A run file or SVG that is not in the expected form."""


def say(text=""):
    print(text, flush=True)
    lines.append(text)


def load_synapses():
    """Steps 1 and 2: the filtered synapses, with each cell's super-class and PC1."""
    schema = ipc.open_file(SYNAPSE_TABLE_FTR).schema
    for c in ("pre", "post"):
        if schema.field(c).type != pa.int64():
            sys.exit(f"{SYNAPSE_TABLE_FTR.name}: {c} is {schema.field(c).type}, expected int64")
    fields = [schema.get_field_index(c) for c in SYNAPSE_COLUMNS]
    reader = ipc.open_file(SYNAPSE_TABLE_FTR, options=ipc.IpcReadOptions(included_fields=fields))
    total = n_si = 0
    parts = []
    for i in range(reader.num_record_batches):
        b = reader.get_batch(i)
        total += b.num_rows
        b = b.filter(pc.and_(pc.greater_equal(b["SI_pre"], 0.1), pc.greater_equal(b["SI_post"], 0.1)))
        n_si += b.num_rows
        b = b.filter(pc.is_in(b["comp"], value_set=pa.array(TYPES)))
        parts.append(pd.DataFrame({"pre": b["pre"].to_numpy(), "post": b["post"].to_numpy(),
                                   "comp": b["comp"].to_numpy(zero_copy_only=False)}))
    syn = pd.concat(parts, ignore_index=True)
    say(f"synapses {total:,}; SI >= 0.1 on both ends {n_si:,}; of type AA/AD/DA/DD {len(syn):,}")

    cells = pd.read_feather(NEURON_TABLE_FTR, columns=["neuron", "super_class"])
    pca = pd.read_feather(PCA_TABLE_FTR)[["neuron", "PC1"]]
    for name, table in ((NEURON_TABLE_FTR.name, cells), (PCA_TABLE_FTR.name, pca)):
        if table["neuron"].dtype != np.int64:
            sys.exit(f"{name}: neuron is {table['neuron'].dtype}, expected int64")
    cells = cells[cells["super_class"].isin(INTRINSIC)]
    cells = cells.merge(pca, on="neuron", how="left", validate="one_to_one").set_index("neuron")
    say(f"intrinsic cells {len(cells):,}, of which without PC1 {int(cells['PC1'].isna().sum())}")

    syn["sc_x"] = syn["pre"].map(cells["super_class"])
    syn["sc_y"] = syn["post"].map(cells["super_class"])
    syn["PC1_x"] = syn["pre"].map(cells["PC1"])
    syn["PC1_y"] = syn["post"].map(cells["PC1"])
    syn = syn.dropna(subset=["sc_x", "sc_y"]).reset_index(drop=True)
    say(f"both cells intrinsic {len(syn):,}; of them touching a cell without PC1 "
        f"{int((syn['PC1_x'].isna() | syn['PC1_y'].isna()).sum()):,}")
    say(f"per type {syn['comp'].value_counts().to_dict()}")
    return syn


def fit(syn, variant=None):
    """Steps 3 to 5, with at most one change (see VARIANTS). The sign flip is applied
    after resampling, which picks the same rows, since resampling ignores the values."""
    data = syn.dropna(subset=["PC1_x", "PC1_y"]).reset_index(drop=True) if variant == "dropnan" else syn
    seed = 7 if variant == "seed7" else 42
    n_da = int((data["comp"] == "DA").sum())
    bal = pd.concat([data[data["comp"] == t].sample(n=n_da, random_state=seed, replace=True)
                     for t in data["comp"].unique()], ignore_index=True)
    X, y = bal[["PC1_x", "PC1_y"]], bal["comp"]
    if variant == "flip":
        X = -X
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=100, random_state=42).fit(X_tr, y_tr)
    pred = rf.predict(X_te)
    counts = confusion_matrix(y_te, pred, labels=TYPES)
    result = {"n_per_type": n_da, "n_train": len(X_tr), "n_test": len(X_te),
              "accuracy": accuracy_score(y_te, pred),
              "percent": counts / counts.sum(axis=1, keepdims=True) * 100,
              "importance": dict(zip(X.columns, rf.feature_importances_))}
    if variant is None:
        result["train_accuracy"] = accuracy_score(y_tr, rf.predict(X_tr))
    return result


def check_run_folder(run):
    """Stop before the rebuild if DIR is not a finished run; return warnings about inputs
    that changed after it."""
    missing = [f for f in RUN_FILES if not (run / f).exists()]
    if missing:
        sys.exit(f"{run}: missing {', '.join(missing)}; is it a tools/run_figure.py run of the script?")
    status = (run / "run_status.txt").read_text().strip()
    if status != "finished":
        sys.exit(f"{run}: run_status.txt says '{status}', not 'finished'")
    oldest_output = min((run / f).stat().st_mtime for f in RUN_FILES)
    changed = [p for p in (FIGURE_SCRIPT, SYNAPSE_TABLE_FTR, NEURON_TABLE_FTR, PCA_TABLE_FTR)
               if p.stat().st_mtime > oldest_output]
    return [f"{p.name} changed after this run was made, so the run may not show the current "
            f"{'code' if p == FIGURE_SCRIPT else 'data'}" for p in changed]


def run_accuracy(run):
    """The test accuracy the figure script printed, from DIR/stdout.log."""
    m = re.search(r"Accuracy Score:\s*([0-9.eE-]+)", (run / "stdout.log").read_text())
    if not m:
        raise Unreadable(f"{run / 'stdout.log'}: no 'Accuracy Score' line")
    return float(m.group(1))


def svg_root(path):
    try:
        return ET.parse(path).getroot()
    except (ET.ParseError, OSError) as e:
        raise Unreadable(f"{path}: {e}")


def svg_texts(root):
    """(x, y, text) of every text element."""
    out = []
    for el in root.iter(SVG + "text"):
        try:
            out.append((float(el.get("x")), float(el.get("y")), (el.text or "").strip()))
        except (TypeError, ValueError):
            continue
    return out


def svg_percent(path):
    """The 16 percentages of a confusion-matrix SVG as a 4 x 4 array, rows = true type and
    columns = predicted type, each placed by its position, after checking that the axis
    labels read AA, AD, DA, DD left to right and top to bottom."""
    texts = svg_texts(svg_root(path))
    if not any(t == "True Label" for _, _, t in texts) or not any(t == "Predicted Label" for _, _, t in texts):
        raise Unreadable(f"{path}: no 'True Label' / 'Predicted Label' axis titles")
    labels = [(x, y, t) for x, y, t in texts if t in TYPES]
    by_y = {}
    for x, y, t in labels:
        by_y.setdefault(round(y, 3), []).append((x, t))
    x_axis = [g for g in by_y.values() if len(g) == 4]                    # four labels in one row
    if len(x_axis) != 1 or len(labels) != 8:
        raise Unreadable(f"{path}: could not find the four labels under each axis")
    predicted = [t for _, t in sorted(x_axis[0])]
    true = [t for _, y, t in sorted((y, x, t) for x, y, t in labels if (x, t) not in x_axis[0])]
    if predicted != TYPES or true != TYPES:
        raise Unreadable(f"{path}: axis labels read {predicted} (predicted) and {true} (true), expected {TYPES}")
    cells = [(x, y, float(t)) for x, y, t in texts if re.fullmatch(r"\d+\.\d\d", t)]
    rows, cols = sorted({round(y, 3) for _, y, _ in cells}), sorted({round(x, 3) for x, _, _ in cells})
    if len(cells) != 16 or len(rows) != 4 or len(cols) != 4:
        raise Unreadable(f"{path}: expected 16 two-decimal numbers on a 4 x 4 grid, found {len(cells)}")
    m = np.full((4, 4), np.nan)
    for x, y, v in cells:
        m[rows.index(round(y, 3)), cols.index(round(x, 3))] = v
    return m


def svg_importances(path):
    """Feature importances from the bar lengths, measured against the x-axis tick labels
    0.0 and 0.5; a bar is a red or cyan shape that starts at 0.0 (to 0.001 of a point)."""
    root = svg_root(path)
    ticks = {}
    for x, _, t in svg_texts(root):
        if t in ("0.0", "0.5"):
            if t in ticks:
                raise Unreadable(f"{path}: more than one '{t}' label")
            ticks[t] = x
    if set(ticks) != {"0.0", "0.5"}:
        raise Unreadable(f"{path}: could not find the x-axis labels 0.0 and 0.5")
    origin, unit = ticks["0.0"], (ticks["0.5"] - ticks["0.0"]) / 0.5
    found = {}
    for g in root.iter(SVG + "g"):
        path_el = g.find(SVG + "path")
        if not (g.get("id") or "").startswith("patch_") or path_el is None:
            continue
        fill = re.search(r"fill:\s*(#[0-9a-f]{6})", path_el.get("style") or "")
        xs = [float(v) for v in re.findall(r"-?\d+\.?\d*", path_el.get("d") or "")][0::2]
        if not fill or fill.group(1) not in BAR_COLOURS or not xs or abs(min(xs) - origin) > 1e-3:
            continue                                # not a bar, e.g. a legend swatch
        name = BAR_COLOURS[fill.group(1)]
        if name in found:
            raise Unreadable(f"{path}: more than one {name} bar")
        found[name] = (max(xs) - origin) / unit
    if set(found) != set(BAR_COLOURS.values()):
        raise Unreadable(f"{path}: expected one red and one cyan bar starting at 0.0, found {found}")
    return found


def show_model(r, title):
    say(title)
    say(f"  balanced to {r['n_per_type']:,} per type; train {r['n_train']:,}, test {r['n_test']:,}")
    say(f"  test accuracy {r['accuracy']!r} (train {r['train_accuracy']:.4f}; chance 0.25)")
    say("  confusion matrix, percent of each true type (rows true, columns predicted "
        + "/".join(TYPES) + ")")
    for t, row in zip(TYPES, r["percent"]):
        say(f"    {t}  " + "  ".join(f"{v:6.2f}" for v in row))
    say(f"  feature importances PC1_x {r['importance']['PC1_x']:.4f}, "
        f"PC1_y {r['importance']['PC1_y']:.4f}")


def compare_with_run(r, run, tol_importance, warnings):
    """True if the rebuild matches the figure script's run in DIR. A missing (NaN) value
    counts as a disagreement, and a file that cannot be read as a FAIL."""
    say(f"\nAgainst the figure script's run in {run}:")
    for w in warnings:
        say(f"  WARNING: {w}")
    try:
        run_acc = run_accuracy(run)
        percent = svg_percent(run / CM_SVG)
        bars = svg_importances(run / FI_SVG)
    except Unreadable as e:
        say(f"  cannot read the run: {e}")
        say("  FAIL")
        return False
    same = run_acc == r["accuracy"]
    say(f"  accuracy        run {run_acc!r}  rebuild {r['accuracy']!r}  {'same' if same else 'DIFFERENT'}")
    diff = np.abs(percent - r["percent"])
    agree = diff <= 0.005 + 1e-9
    say(f"  confusion matrix  {int(agree.sum())} of 16 percentages agree to 0.005 "
        f"(largest difference {np.nanmax(diff) if np.isfinite(diff).any() else float('nan'):.4f})")
    total = sum(bars.values())
    sums_to_one = abs(total - 1) <= 2 * tol_importance
    say(f"  importances from the bars sum to {total:.4f}")
    ok = same and bool(agree.all()) and sums_to_one
    for f in ("PC1_x", "PC1_y"):
        good = abs(bars[f] - r["importance"][f]) <= tol_importance
        ok = ok and bool(good)
        say(f"  importance {f}  bar {bars[f]:.4f}  rebuild {r['importance'][f]:.4f}  "
            f"{'agree' if good else 'DIFFERENT'} (difference {abs(bars[f] - r['importance'][f]):.5f})")
    say(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def compare_with_earlier(run, earlier, noise):
    """Information only: the run against an earlier run's SVGs."""
    say(f"\nThis run against the earlier run in {earlier} (information only):")
    try:
        run_acc = run_accuracy(run)
        now, before = svg_percent(run / CM_SVG), svg_percent(earlier / CM_SVG)
        b_now, b_before = svg_importances(run / FI_SVG), svg_importances(earlier / FI_SVG)
    except Unreadable as e:
        say(f"  skipped: {e}")
        return
    before_acc = np.trace(before) / 4 / 100
    say(f"  accuracy        earlier {before_acc * 100:.2f}% (mean of its diagonal)  "
        f"this run {run_acc * 100:.2f}%  difference {abs(run_acc - before_acc) * 100:.2f} points")
    say("  confusion matrix, percent of each true type, earlier -> this run")
    for i, t in enumerate(TYPES):
        say(f"    {t}  " + "  ".join(f"{before[i, j]:6.2f} -> {now[i, j]:6.2f}" for j in range(4)))
    change = np.abs(now - before)
    i, j = np.unravel_index(change.argmax(), change.shape)
    say(f"  largest cell change {change.max():.2f} points ({TYPES[i]} predicted as {TYPES[j]})")
    say(f"  importance PC1_x  earlier {b_before['PC1_x']:.3f}  this run {b_now['PC1_x']:.3f}; "
        f"PC1_y  earlier {b_before['PC1_y']:.3f}  this run {b_now['PC1_y']:.3f}")
    if noise is not None:
        acc_spread, cell_move = noise
        acc_diff = abs(run_acc - before_acc) * 100
        say(f"  accuracy difference {acc_diff:.2f} points against the variants' spread of {acc_spread:.2f}: "
            f"{'larger' if acc_diff > acc_spread else 'not larger'}")
        say(f"  largest cell change {change.max():.2f} points against the variants' largest cell movement "
            f"of {cell_move:.2f}: {'larger' if change.max() > cell_move else 'not larger'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, help="tools/run_figure.py output folder of the figure script")
    ap.add_argument("--variants", action="store_true", help="also run the three variants")
    ap.add_argument("--earlier", type=Path, help="folder with an earlier run's two SVGs (needs --run)")
    ap.add_argument("--report", type=Path, help="also write everything printed to this file")
    ap.add_argument("--tol-importance", type=float, default=0.001,
                    help="largest difference between bar and rebuild importance (default 0.001)")
    args = ap.parse_args()
    if args.earlier and not args.run:
        ap.error("--earlier needs --run")
    for d in [args.run, args.earlier]:
        if d is not None and not d.is_dir():
            ap.error(f"{d} is not a folder")
    warnings = check_run_folder(args.run) if args.run else []

    ok = True
    try:
        say(f"scikit-learn {sklearn.__version__}, pandas {pd.__version__}, numpy {np.__version__}")
        syn = load_synapses()
        base = fit(syn)
        show_model(base, "\nRebuild of the figure script's model:")
        if args.run:
            ok = compare_with_run(base, args.run, args.tol_importance, warnings)

        noise = None
        if args.variants:
            accs, moves = [base["accuracy"]], []
            say("\nVariants (one change each):")
            say(f"  {'variant':<46}{'accuracy':>9}{'largest cell movement':>24}")
            say(f"  {'none':<46}{base['accuracy'] * 100:>8.2f}%{0:>19.2f} pts")
            for v, label in VARIANTS.items():
                r = fit(syn, v)
                accs.append(r["accuracy"])
                moves.append(np.abs(r["percent"] - base["percent"]).max())
                say(f"  {label:<46}{r['accuracy'] * 100:>8.2f}%{moves[-1]:>19.2f} pts")
            noise = ((max(accs) - min(accs)) * 100, max(moves))
            say(f"  spread of accuracy {noise[0]:.2f} points; largest cell movement {noise[1]:.2f} points")

        if args.earlier:
            compare_with_earlier(args.run, args.earlier, noise)
    finally:
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text("\n".join(lines) + "\n")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
