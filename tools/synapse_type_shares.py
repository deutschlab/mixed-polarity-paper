"""Print the share of each synapse type (AD, AA, DD, DA) in the synapse table.

Four groups of synapses are counted:
- all synapses in the table labelled AA, AD, DA or DD;
- the same, only where the pre and post neurons both have SI >= 0.1;
- intrinsic neurons only (both ends), every labelled synapse;
- intrinsic neurons only, with the same SI >= 0.1 filter.

The delivered synapse table already holds only synapses whose two neurons were
split successfully (step 04 drops the others), so "all synapses" means all of
those, including non-intrinsic neurons such as sensory and descending ones.
Synapses with a linker end (AL, LD and so on) are left out, as in the figures;
their number is printed.

"Intrinsic" means what the figure scripts keep: super_class central, optic,
visual_projection or visual_centrifugal, with axon_correct, dend_correct and
primary_type present in the neuron table.

By default the SI filter uses the SI_pre and SI_post columns stored in the synapse
table, which hold the older SI. With --si corrected it uses SI_updated.ftr instead;
a neuron missing from that file does not pass the filter.

With the defaults, the delivered table gives 47.7 / 20.9 / 22.1 / 9.3 %
(AD / AA / DD / DA) for all synapses, and 18,146,677 synapses for intrinsic neurons
with SI >= 0.1 (the n of the paper's Fig 3F).

--table points the tool at another synapse table with the same columns (pre, post,
comp, and SI_pre and SI_post for the default SI filter). --column counts another
column of synapse types instead of comp, for example a column of types made with a
different axon/dendrite split. The intrinsic neurons and the corrected SI still come
from the delivered neuron table and SI_updated.ftr. The default SI filter uses the
table's own SI_pre and SI_post, whatever they hold: in a table made with a different
split they may still be the published split's SI, so check that before reading the
SI >= 0.1 lines as that split's.

Usage:
    python tools/synapse_type_shares.py
    python tools/synapse_type_shares.py --si corrected
    python tools/synapse_type_shares.py --table path/to/other_synapse_table.ftr
    python tools/synapse_type_shares.py --table path/to/other_synapse_table.ftr --column maxSI_compartment

Needs about 4 GB of memory (about 5 GB with --si corrected) for the full table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import NEURON_TABLE_FTR, SI_UPDATED_FTR, SYNAPSE_TABLE_FTR  # noqa: E402

TYPES = ["AD", "AA", "DD", "DA"]
INTRINSIC = ["central", "optic", "visual_projection", "visual_centrifugal"]
SI_CUTOFF = 0.1


def intrinsic_ids() -> pa.Array:
    """IDs of the intrinsic neurons, as the figure scripts select them."""
    nodes = feather.read_table(
        NEURON_TABLE_FTR,
        columns=["neuron", "super_class", "axon_correct", "dend_correct", "primary_type"],
    ).to_pandas()
    nodes = nodes[nodes["super_class"].isin(INTRINSIC)]
    nodes = nodes.dropna(subset=["axon_correct", "dend_correct", "primary_type"])
    return pa.array(nodes["neuron"].astype(np.int64).to_numpy())


def corrected_si(ids: pa.ChunkedArray) -> np.ndarray:
    """Corrected SI of each neuron in ids (NaN if it has none)."""
    updated = feather.read_table(SI_UPDATED_FTR, columns=["root_id", "SI"]).to_pandas()
    if updated["root_id"].duplicated().any():
        sys.exit(f"{SI_UPDATED_FTR.name} has duplicated root_id values")
    known = updated["root_id"].astype(np.int64).to_numpy()
    si = updated["SI"].to_numpy()
    order = np.argsort(known)
    known, si = known[order], si[order]
    ids = ids.to_numpy()
    position = np.clip(np.searchsorted(known, ids), 0, len(known) - 1)
    return np.where(known[position] == ids, si[position], np.nan)


def print_shares(comp: pa.ChunkedArray, mask, label: str) -> None:
    """Print how many synapses the mask keeps and the share of each type among them."""
    chosen = pc.filter(comp, mask)
    total = len(chosen)
    if total == 0:
        print(f"  {label:40s} n = 0")
        return
    counts = {row["values"]: row["counts"] for row in pc.value_counts(chosen).to_pylist()}
    shares = " / ".join(f"{100 * counts.get(t, 0) / total:.1f}" for t in TYPES)
    print(f"  {label:40s} n = {total:>11,}   {shares}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--table", type=Path, default=SYNAPSE_TABLE_FTR,
                        help="synapse table with pre, post, the type column, and SI_pre and "
                             "SI_post for the default SI filter (default: the delivered synapse table)")
    parser.add_argument("--column", default="comp",
                        help="column of synapse types to count (default: comp)")
    parser.add_argument("--si", choices=["table", "corrected"], default="table",
                        help="SI for the >= 0.1 filter: the table's own SI_pre/SI_post "
                             "(older SI, default) or SI_updated.ftr (corrected)")
    args = parser.parse_args()

    if args.column in ("pre", "post", "SI_pre", "SI_post"):
        sys.exit(f"{args.column} is not a column of synapse types")
    columns = ["pre", "post", args.column] + (["SI_pre", "SI_post"] if args.si == "table" else [])
    names = pa.ipc.open_file(args.table).schema.names
    missing = [c for c in columns if c not in names]
    if missing:
        sys.exit(f"{args.table.name} has no column {', '.join(missing)}")
    table = feather.read_table(args.table, columns=columns)
    for column in ("pre", "post"):
        if table[column].type != pa.int64():
            sys.exit(f"{column} is {table[column].type}, not int64: neuron IDs must be "
                     "64-bit integers, or they will not match the neuron table")
        if table[column].null_count:
            sys.exit(f"{column} has {table[column].null_count:,} empty values")
    comp = table[args.column]
    if pa.types.is_dictionary(comp.type):
        comp = comp.cast(pa.string())
    if not (pa.types.is_string(comp.type) or pa.types.is_large_string(comp.type)):
        sys.exit(f"{args.column} holds {comp.type}, not text such as 'AD': is it a column of synapse types?")

    labelled = pc.is_in(comp, value_set=pa.array(TYPES))
    if not pc.any(labelled).as_py():
        sys.exit(f"{args.column} has no value {', '.join(TYPES)} at all: is it a column of synapse types?")
    if args.si == "table":
        polarized = pc.and_(pc.greater_equal(table["SI_pre"], SI_CUTOFF),
                            pc.greater_equal(table["SI_post"], SI_CUTOFF))
        polarized = pc.fill_null(polarized, False)
        si_source = "SI_pre and SI_post in the table" + (
            " (older SI)" if args.table.resolve() == SYNAPSE_TABLE_FTR.resolve() else "")
    else:
        polarized = pa.array((corrected_si(table["pre"]) >= SI_CUTOFF)
                             & (corrected_si(table["post"]) >= SI_CUTOFF))
        si_source = "SI_updated.ftr (corrected SI)"
    ids = intrinsic_ids()
    intrinsic = pc.and_(pc.is_in(table["pre"], value_set=ids),
                        pc.is_in(table["post"], value_set=ids))

    print(f"Synapse table: {args.table}")
    if args.table.resolve() != SYNAPSE_TABLE_FTR.resolve():
        print("Note: the intrinsic neurons come from the delivered neuron table"
              + (", and the corrected SI from SI_updated.ftr" if args.si == "corrected" else ""))
    print(f"Synapse types: {args.column}")
    print(f"SI filter: {si_source}")
    unlabelled = len(comp) - pc.sum(labelled).as_py()
    print(f"Left out: {unlabelled:,} synapses with a linker end or no type")
    print(f"Shares in %: {' / '.join(TYPES)}")
    print_shares(comp, labelled, "all synapses")
    print_shares(comp, pc.and_(labelled, polarized), f"all synapses, SI >= {SI_CUTOFF}")
    print_shares(comp, pc.and_(labelled, intrinsic), "intrinsic neurons")
    print_shares(comp, pc.and_(pc.and_(labelled, intrinsic), polarized),
                 f"intrinsic neurons, SI >= {SI_CUTOFF}")


if __name__ == "__main__":
    # Prints the share of each synapse type (AA, AD, DA, DD) in
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/data/derived/synapses_783_article_princeton.ftr
    # in four groups: all synapses, SI >= 0.1 on both ends, intrinsic neurons, and intrinsic neurons with
    # SI >= 0.1 (intrinsic cells from /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/data/derived/neuron_data_full_article_princeton.ftr).
    # --si corrected takes the SI from
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/data/derived/SI_updated.ftr
    # instead of the table's own SI_pre / SI_post. It writes no file. To run it, uncomment:
    # sys.argv[1:] = ["--si", "corrected"]
    main()
