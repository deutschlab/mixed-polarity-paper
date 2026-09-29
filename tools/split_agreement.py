"""Compare the synapse types (AA, AD, DA, DD) given by two or more axon/dendrite splits.

The synapse table must hold one column of synapse types per split, for example comp
(the published split) and maxSI_compartment. For every pair of columns the tool
prints, for four groups of synapses (all; SI >= 0.1 on both ends; intrinsic neurons
only; intrinsic neurons with SI >= 0.1):

- how many synapses have one of the four types in both columns, in only one of them,
  and in neither (a linker type, such as AL or LD, or an empty value, is not a type);
- agreement: the share of synapses typed in both columns that get the same type;
- chance agreement and Cohen's kappa, from the same synapses;
- per type: of the synapses of that type in the first column that are also typed in the
  second, the share with the same type in the second (this depends on the order of the
  columns), and the specific
  agreement 2 x (same type in both) / (that type in the first + in the second), which
  does not;
- the 4 x 4 table of counts.

It then prints:

- agreement by SI band, where a synapse's band is the lower SI of its two neurons, so
  that agreement in mixed and in segregated neurons can be told apart;
- a per-neuron view: each synapse has two ends, and each end carries its own neuron's
  label (A or D; the first letter of the type for the presynaptic end, the second for
  the postsynaptic end). For each neuron, the share of its ends that get the same
  label from both columns (linker ends and empty values left out), summarised over
  neurons: pooled, median, mean, and the share of neurons with every end the same,
  below 90% and at most 50%, and the median number of ends compared per neuron. This
  counts every neuron once, whatever its size, so neurons with only a few ends weigh as
  much as large ones. Here a neuron is intrinsic if it is itself, whatever its partner,
  and the typed end of a synapse with a linker type (the A end of AL) is counted.

Agreement is counted only on synapses typed in both columns: a synapse with a linker
type in one column is counted under "only in" the other column, not as a
disagreement.

"Intrinsic" and the SI filter work as in tools/synapse_type_shares.py: super_class
central, optic, visual_projection or visual_centrifugal, with axon_correct,
dend_correct and primary_type present in the neuron table; by default the SI filter
uses the SI_pre and SI_post columns of the synapse table, and with --si corrected it
uses SI_updated.ftr instead. Either way it is one SI per neuron, the same for every
column compared: it does not use each split's own score. Both SIs come from the published
split, so the SI groups and bands pick out neurons that split separates well, and
agreement with it looks higher there.

For a table other than the delivered synapse table, the tool also reports how many
synapses and neurons of the delivered table it lacks, so that its counts are not
mistaken for the paper's.

Synapses are not independent: each belongs to two neurons. --bootstrap N adds 95%
intervals for agreement and kappa by resampling clusters with Poisson(1) weights: a
synapse weighs the product of the weights of its two neurons' clusters, or that one
weight when both neurons are in the same cluster. By default the clusters are cell
types (primary_type; a neuron without one is its own cluster), which also keeps
left/right homologs together; --cluster neuron resamples single neurons and gives
narrower intervals. Use at least 1000 replicates for intervals that will be quoted.

Usage:
    python tools/split_agreement.py --table path/to/table.ftr comp maxSI_compartment
    python tools/split_agreement.py --table path/to/table.ftr comp maxSI_compartment \\
        fisher_compartment maxPhi_compartment --bootstrap 1000

Needs about 12 GB of memory for a table of 76 million synapses (about a minute for four
columns); each bootstrap replicate adds about a second per pair of columns.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import NEURON_TABLE_FTR, SYNAPSE_TABLE_FTR  # noqa: E402
from synapse_type_shares import INTRINSIC, SI_CUTOFF, TYPES, corrected_si, intrinsic_ids  # noqa: E402

N_TYPES = len(TYPES)
NOT_TYPED = N_TYPES  # code for a linker type or an empty value
NO_LABEL = -1  # an end whose label is L, empty or unknown
GROUPS = ["all synapses", f"all synapses, SI >= {SI_CUTOFF}",
          "intrinsic neurons", f"intrinsic neurons, SI >= {SI_CUTOFF}"]
SI_BANDS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, np.inf]


def read_labels(table_path: Path, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each synapse: the type code (0-3 in the order of TYPES, NOT_TYPED otherwise),
    and the presynaptic and postsynaptic end labels (0 = A, 1 = D, NO_LABEL otherwise)."""
    values = feather.read_table(table_path, columns=[column])[column]
    text = values.type.value_type if pa.types.is_dictionary(values.type) else values.type
    if not (pa.types.is_string(text) or pa.types.is_large_string(text)):
        sys.exit(f"{column} holds {values.type}, not text such as 'AD': is it a column of synapse types?")
    codes = np.empty(len(values), np.int8)
    pre_end = np.empty(len(values), np.int8)
    post_end = np.empty(len(values), np.int8)
    start = 0
    for chunk in values.chunks:
        # work on the few distinct labels, then spread them over the rows
        encoded = chunk if pa.types.is_dictionary(chunk.type) else pc.dictionary_encode(chunk)
        words = encoded.dictionary.to_pylist()
        type_of = np.array([TYPES.index(w) if w in TYPES else NOT_TYPED for w in words] + [NOT_TYPED], np.int8)
        pre_of = np.array([("AD".index(w[0]) if w and w[0] in "AD" else NO_LABEL) for w in words] + [NO_LABEL], np.int8)
        post_of = np.array([("AD".index(w[1]) if w and len(w) > 1 and w[1] in "AD" else NO_LABEL)
                            for w in words] + [NO_LABEL], np.int8)
        rows = pc.fill_null(encoded.indices, len(words)).to_numpy()  # an empty value points past the labels
        stop = start + len(chunk)
        codes[start:stop], pre_end[start:stop], post_end[start:stop] = type_of[rows], pre_of[rows], post_of[rows]
        start = stop
    if (codes == NOT_TYPED).all():
        sys.exit(f"{column} has no value {', '.join(TYPES)} at all: is it a column of synapse types?")
    return codes, pre_end, post_end


def cluster_of_neurons(neurons: np.ndarray, by_type: bool) -> np.ndarray:
    """Cluster index of each neuron: its cell type, or the neuron itself."""
    if not by_type:
        return np.arange(len(neurons))
    types = feather.read_table(NEURON_TABLE_FTR, columns=["neuron", "primary_type"]).to_pandas()
    types = types.set_index(types["neuron"].astype(np.int64))["primary_type"]
    labels = types.reindex(neurons)
    # a neuron without a type, or missing from the neuron table, is its own cluster
    labels = labels.where(labels.notna(), pd.Series(neurons, index=neurons).astype(str).radd("neuron "))
    return np.unique(labels.to_numpy(dtype=str), return_inverse=True)[1]


def tabulate(key: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """Counts per stratum (see group_tables) x first type x second type."""
    counts = np.bincount(key, weights=weights, minlength=4 * (N_TYPES + 1) ** 2)
    return counts.reshape(4, N_TYPES + 1, N_TYPES + 1)


def group_tables(strata: np.ndarray) -> list[np.ndarray]:
    """The four reported groups, summed from the strata (intrinsic x SI >= cutoff)."""
    # strata: 0 = neither, 1 = SI only, 2 = intrinsic only, 3 = both
    return [strata.sum(axis=0), strata[1] + strata[3], strata[2] + strata[3], strata[3]]


def agreement_and_kappa(table: np.ndarray) -> tuple[float, float, float]:
    """Agreement, chance agreement and kappa on the synapses typed in both columns."""
    both = table[:N_TYPES, :N_TYPES]
    total = both.sum()
    if total == 0:
        return np.nan, np.nan, np.nan
    observed = np.trace(both) / total
    chance = float((both.sum(axis=1) / total) @ (both.sum(axis=0) / total))
    kappa = (observed - chance) / (1 - chance) if chance < 1 else np.nan
    return observed, chance, kappa


def percent(numerator: float, denominator: float, digits: int = 1) -> str:
    return f"{100 * numerator / denominator:.{digits}f}%" if denominator else "-"


def print_pair(name_a: str, name_b: str, table: np.ndarray, interval: dict | None) -> None:
    both = table[:N_TYPES, :N_TYPES]
    only_a = table[:N_TYPES, NOT_TYPED].sum()
    only_b = table[NOT_TYPED, :N_TYPES].sum()
    neither = table[NOT_TYPED, NOT_TYPED]
    observed, chance, kappa = agreement_and_kappa(table)
    print(f"  {name_a} vs {name_b}")
    print(f"    typed in both: {both.sum():,.0f}   only in {name_a}: {only_a:,.0f}   "
          f"only in {name_b}: {only_b:,.0f}   in neither: {neither:,.0f}")
    if both.sum() == 0:
        print("    no synapse typed in both columns")
        return
    line = f"    agreement {100 * observed:.2f}%   chance {100 * chance:.2f}%   kappa {kappa:.3f}"
    if interval:
        line += (f"   (95% interval: agreement {100 * interval['agreement'][0]:.2f}-"
                 f"{100 * interval['agreement'][1]:.2f}%, kappa {interval['kappa'][0]:.3f}-"
                 f"{interval['kappa'][1]:.3f})")
    print(line)
    rows, cols = both.sum(axis=1), both.sum(axis=0)
    print(f"    same type in {name_b}, by type in {name_a}: "
          + "   ".join(f"{t} {percent(both[i, i], rows[i])}" for i, t in enumerate(TYPES)))
    print("    specific agreement per type: "
          + "   ".join(f"{t} {percent(2 * both[i, i], rows[i] + cols[i])}" for i, t in enumerate(TYPES)))
    print(f"    counts (rows {name_a}, columns {name_b}):")
    print("        " + "".join(f"{t:>14}" for t in TYPES))
    for i, t in enumerate(TYPES):
        print(f"      {t}" + "".join(f"{both[i, j]:>14,.0f}" for j in range(N_TYPES)))


def coverage_against_delivered(table_path: Path) -> None:
    """Print how many synapses and neurons of the delivered synapse table this table lacks."""
    names = pa.ipc.open_file(table_path).schema.names
    if "synapse_id" not in names:
        print("Coverage: the table has no synapse_id column, so it cannot be matched to the "
              "delivered synapse table")
        return
    def column(path: Path, name: str) -> np.ndarray:
        return feather.read_table(path, columns=[name])[name].to_numpy()

    # sorted arrays and searchsorted keep this to a few GB, where hash sets need about 10
    here_ids = np.sort(column(table_path, "synapse_id"))
    if (np.diff(here_ids) == 0).any():
        print("Coverage: synapse_id is not unique in this table, so it cannot be matched to the "
              "delivered synapse table")
        return
    delivered_ids = column(SYNAPSE_TABLE_FTR, "synapse_id")
    position = np.minimum(np.searchsorted(here_ids, delivered_ids), len(here_ids) - 1)
    found = int((here_ids[position] == delivered_ids).sum())
    n_here, n_delivered = len(here_ids), len(delivered_ids)
    del here_ids, delivered_ids, position

    def neurons_of(path: Path) -> np.ndarray:
        return np.union1d(np.unique(column(path, "pre")), np.unique(column(path, "post")))

    neurons_delivered = neurons_of(SYNAPSE_TABLE_FTR)
    absent = np.setdiff1d(neurons_delivered, neurons_of(table_path)).size
    print(f"Coverage: {n_delivered - found:,} of the {n_delivered:,} synapses of the delivered synapse "
          f"table are not in this table ({n_here - found:,} of this table's are not in it), and "
          f"{absent:,} of its {neurons_delivered.size:,} neurons have no synapse here")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("columns", nargs="+",
                        help="two or more columns of synapse types to compare, for example "
                             "comp maxSI_compartment")
    parser.add_argument("--table", type=Path, default=SYNAPSE_TABLE_FTR,
                        help="synapse table with pre, post, the type columns, and SI_pre and "
                             "SI_post for the default SI filter (default: the delivered synapse table)")
    parser.add_argument("--si", choices=["table", "corrected"], default="table",
                        help="SI for the >= 0.1 filter, the SI bands and the per-neuron groups: the "
                             "table's own SI_pre/SI_post (default) or SI_updated.ftr (corrected)")
    parser.add_argument("--bootstrap", type=int, default=0, metavar="N",
                        help="number of bootstrap replicates for 95%% intervals (default 0: none)")
    parser.add_argument("--cluster", choices=["type", "neuron"], default="type",
                        help="unit resampled by the bootstrap: cell type (default) or neuron")
    parser.add_argument("--seed", type=int, default=0, help="seed for the bootstrap (default 0)")
    args = parser.parse_args()

    if len(args.columns) < 2 or len(set(args.columns)) != len(args.columns):
        sys.exit("give at least two different columns of synapse types")
    names = pa.ipc.open_file(args.table).schema.names
    needed = ["pre", "post"] + args.columns + (["SI_pre", "SI_post"] if args.si == "table" else [])
    missing = [c for c in needed if c not in names]
    if missing:
        sys.exit(f"{args.table.name} has no column {', '.join(missing)}")

    ids = feather.read_table(args.table, columns=["pre", "post"])
    for column in ("pre", "post"):
        if ids[column].type != pa.int64():
            sys.exit(f"{column} is {ids[column].type}, not int64: neuron IDs must be "
                     "64-bit integers, or they will not match the neuron table")
        if ids[column].null_count:
            sys.exit(f"{column} has {ids[column].null_count:,} empty values")

    print(f"Synapse table: {args.table}")
    other_table = args.table.resolve() != SYNAPSE_TABLE_FTR.resolve()
    if other_table:
        coverage_against_delivered(args.table)
        print("Note: the intrinsic neurons come from the delivered neuron table"
              + (", and the corrected SI from SI_updated.ftr" if args.si == "corrected" else ""))

    if args.si == "table":
        si = feather.read_table(args.table, columns=["SI_pre", "SI_post"])
        si_pre = si["SI_pre"].to_numpy(zero_copy_only=False)
        si_post = si["SI_post"].to_numpy(zero_copy_only=False)
        del si
        si_source = "SI_pre and SI_post in the table" + ("" if other_table else " (older SI)")
    else:
        si_pre, si_post = corrected_si(ids["pre"]), corrected_si(ids["post"])
        si_source = "SI_updated.ftr (corrected SI)"
    lower_si = np.fmin(si_pre, si_post)
    lower_si[np.isnan(si_pre) | np.isnan(si_post)] = np.nan
    polarized = lower_si >= SI_CUTOFF  # NaN compares False

    pre = ids["pre"].to_numpy()
    post = ids["post"].to_numpy()
    del ids
    # np.unique(..., return_inverse=True) on both columns at once needs about 8 GB more
    neurons = np.union1d(np.unique(pre), np.unique(post))
    index_pre = np.searchsorted(neurons, pre).astype(np.int32)
    index_post = np.searchsorted(neurons, post).astype(np.int32)
    neuron_si = np.full(len(neurons), np.nan)
    neuron_si[index_pre] = si_pre
    neuron_si[index_post] = si_post
    del si_pre, si_post
    intrinsic_neuron = np.isin(neurons, intrinsic_ids().to_numpy())
    intrinsic = intrinsic_neuron[index_pre] & intrinsic_neuron[index_post]
    stratum = (polarized.astype(np.int16) + 2 * intrinsic.astype(np.int16))
    del polarized, intrinsic
    band = np.digitize(lower_si, SI_BANDS) - 1
    band[np.isnan(lower_si) | (lower_si < 0)] = -1
    del lower_si

    labels = {c: read_labels(args.table, c) for c in args.columns}
    codes = {c: labels[c][0] for c in args.columns}
    pairs = list(combinations(args.columns, 2))
    size = (N_TYPES + 1) ** 2
    # key = stratum x first type x second type; at most 4 * 25 values, so int8 is enough
    keys = {pair: (stratum * size + codes[pair[0]].astype(np.int16) * (N_TYPES + 1)
                   + codes[pair[1]]).astype(np.int8)
            for pair in pairs}
    del stratum

    intervals: dict = {}
    failed_replicates = 0
    if args.bootstrap > 0:
        cluster = cluster_of_neurons(neurons, args.cluster == "type")
        cluster_pre, cluster_post = cluster[index_pre], cluster[index_post]
        same_cluster = cluster_pre == cluster_post
        n_clusters = int(cluster.max()) + 1
        rng = np.random.default_rng(args.seed)
        draws = {(pair, g): {"agreement": [], "kappa": []} for pair in pairs for g in range(4)}
        for _ in range(args.bootstrap):
            weight_of_cluster = rng.poisson(1.0, n_clusters).astype(np.float64)
            weight_pre = weight_of_cluster[cluster_pre]
            # both ends in one cluster: that cluster's weight once, not squared
            weight = np.where(same_cluster, weight_pre, weight_pre * weight_of_cluster[cluster_post])
            del weight_pre
            for pair in pairs:
                for g, table in enumerate(group_tables(tabulate(keys[pair], weight))):
                    observed, _, kappa = agreement_and_kappa(table)
                    failed_replicates += int(np.isnan(observed))
                    draws[(pair, g)]["agreement"].append(observed)
                    draws[(pair, g)]["kappa"].append(kappa)
            del weight
        for key, values in draws.items():
            intervals[key] = {name: tuple(np.nanpercentile(v, [2.5, 97.5])) if not np.isnan(v).all()
                              else (np.nan, np.nan) for name, v in values.items()}

    print(f"SI filter, bands and per-neuron groups: {si_source}")
    print("  one SI for every column compared, from the published split: the SI groups and bands "
          "favour agreement with that split")
    print(f"Intrinsic: {', '.join(INTRINSIC)}, with axon_correct, dend_correct and primary_type "
          "in the neuron table (both ends of the synapse; in the per-neuron view, the neuron itself)")
    if args.bootstrap > 0:
        print(f"Bootstrap: {args.bootstrap} replicates, clusters = "
              f"{'cell types' if args.cluster == 'type' else 'neurons'}, seed {args.seed}"
              + (f"; {failed_replicates} group results with no synapse left out of the intervals"
                 if failed_replicates else ""))
    for column in args.columns:
        typed = int((codes[column] != NOT_TYPED).sum())
        print(f"{column}: {typed:,} of {len(pre):,} synapses have one of {'/'.join(TYPES)}")

    tables = {pair: group_tables(tabulate(keys[pair], None)) for pair in pairs}
    for g, group in enumerate(GROUPS):
        print(f"\n{group}")
        for pair in pairs:
            print_pair(pair[0], pair[1], tables[pair][g], intervals.get((pair, g)))

    print("\nAgreement by SI band (the lower SI of the two neurons), all synapses; agreement "
          "counts the synapses typed in both columns")
    band_labels = [f"[{lo:g}, {hi:g})" if np.isfinite(hi) else f">= {lo:g}"
                   for lo, hi in zip(SI_BANDS[:-1], SI_BANDS[1:])]
    pair_names = [f"{a} vs {b}" for a, b in pairs]
    width = max(len("synapses in band"), *(len(n) for n in pair_names))
    print(f"  {'':<{width}}" + "".join(f"{label:>14}" for label in band_labels))
    print(f"  {'synapses in band':<{width}}"
          + "".join(f"{int((band == b).sum()):>14,}" for b in range(len(band_labels))))
    for name, pair in zip(pair_names, pairs):
        cells = []
        for b in range(len(band_labels)):
            in_band = band == b
            first, second = codes[pair[0]][in_band], codes[pair[1]][in_band]
            both = (first != NOT_TYPED) & (second != NOT_TYPED)
            cells.append(percent((both & (first == second)).sum(), both.sum()))
        print(f"  {name:<{width}}" + "".join(f"{c:>14}" for c in cells))
    no_band = int((band == -1).sum())
    if no_band:
        print(f"  In no band, because one of the two neurons has no SI (or one below 0): {no_band:,} synapses")

    print("\nPer neuron: the share of each neuron's synapse ends that get the same label (A or D) "
          "in both columns; linker ends and empty values left out, the typed end of a linker-typed synapse "
          "kept; every neuron counts once")
    groups = [("all neurons", np.ones(len(neurons), bool)),
              ("intrinsic neurons", intrinsic_neuron),
              (f"intrinsic, neuron SI >= {SI_CUTOFF}", intrinsic_neuron & (neuron_si >= SI_CUTOFF)),
              (f"intrinsic, neuron SI < {SI_CUTOFF}", intrinsic_neuron & (neuron_si < SI_CUTOFF))]
    for name, pair in zip(pair_names, pairs):
        (_, pre_a, post_a), (_, pre_b, post_b) = labels[pair[0]], labels[pair[1]]
        compared = np.zeros(len(neurons))
        same = np.zeros(len(neurons))
        for index, end_a, end_b in ((index_pre, pre_a, pre_b), (index_post, post_a, post_b)):
            valid = (end_a != NO_LABEL) & (end_b != NO_LABEL)
            compared += np.bincount(index[valid], minlength=len(neurons))
            same += np.bincount(index[valid & (end_a == end_b)], minlength=len(neurons))
        print(f"  {name}")
        for group_name, members in groups:
            keep = members & (compared > 0)
            if not keep.any():
                print(f"    {group_name:<30} no neuron")
                continue
            share = same[keep] / compared[keep]
            print(f"    {group_name:<30} {int(keep.sum()):>8,} neurons   ends {percent(same[keep].sum(), compared[keep].sum())}"
                  f"   median {100 * np.median(share):.1f}%   mean {100 * share.mean():.1f}%   "
                  f"ends per neuron (median) {np.median(compared[keep]):.0f}   "
                  f"all same {percent((share == 1).sum(), keep.sum())}   "
                  f"below 90% {percent((share < 0.9).sum(), keep.sum())}   "
                  f"50% or less {percent((share <= 0.5).sum(), keep.sum())}")


if __name__ == "__main__":
    main()
