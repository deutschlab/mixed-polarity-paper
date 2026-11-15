"""
Split a healed navis neuron with attached synapses at its maximum-SI cut (MaxSI), for drawing.

split_max_si is Amit Gross's function. It scores the same candidate cuts as methods/node_cut_si.py: every node with
a parent and exactly one child. Side 1 is the child and everything below it, side 2 the rest of the cell without
the cut node, and the synapses on the cut node count on neither side. It takes the cut with the highest SI (on a
tie, the first node in the node table's order, as step 09 does) and calls the side with the higher presynaptic
fraction the axon. It returns the dendrite and the axon as a navis NeuronList, in that order, and with
return_metrics=True also the SI and each side's canonical share: dendrite postsynapses / dendrite synapses, axon
presynapses / axon synapses.

It computes the entropy with math.log2, where node_cut_si uses math.log(p, 2). The two can differ in the last
digit, so where two cuts with different side counts have almost the same SI, it can choose another cut than step
09 (processing/09_maxsi_synapse_labels.py); otherwise it chooses the same one. Unlike step 09 it does not refuse a
cell whose two sides have the same or nearly the same presynaptic fraction (numpy isclose), or a cut that leaves a
side with no synapse; there the axon and dendrite it returns mean nothing. tools/draw_max_si_split.py checks every
cell it draws against node_cut_si and step 09's rules.
"""
from collections import Counter
from math import isfinite, log2

import navis


DENDRITE_COLOR = "#fbb040"
AXON_COLOR = "#92278f"


def _entropy(pre, post):
    total = pre + post
    if total == 0:
        return 0.0

    pre_probability = pre / total
    post_probability = post / total

    entropy = 0.0
    if pre_probability > 0:
        entropy -= pre_probability * log2(pre_probability)
    if post_probability > 0:
        entropy -= post_probability * log2(post_probability)
    return entropy


def _si_for_sides(side_1_pre, side_1_post, side_2_pre, side_2_post):
    total_pre = side_1_pre + side_2_pre
    total_post = side_1_post + side_2_post
    normalizing_entropy = _entropy(total_pre, total_post)

    if normalizing_entropy == 0:
        return float("nan")

    side_1_total = side_1_pre + side_1_post
    side_2_total = side_2_pre + side_2_post
    total = side_1_total + side_2_total

    weighted_entropy = (
        side_1_total * _entropy(side_1_pre, side_1_post)
        + side_2_total * _entropy(side_2_pre, side_2_post)
    ) / total

    return 1 - weighted_entropy / normalizing_entropy


def split_max_si(neuron, return_metrics=False):
    """Split a synapse-annotated Navis neuron at its maximum-SI node.

    Returns a Navis NeuronList ordered as dendrite, axon, ready for
    ``navis.plot2d``. Set return_metrics=True to
    also receive the maximum SI and the canonical percentages.
    """
    required_synapse_tables = ("presynapses", "postsynapses")
    for table_name in required_synapse_tables:
        table = getattr(neuron, table_name, None)
        if table is None or "node_id" not in table.columns:
            raise ValueError(f"neuron.{table_name} must contain a node_id column")

    nodes = neuron.nodes[["node_id", "parent_id"]].drop_duplicates("node_id")
    node_ids = nodes["node_id"].to_numpy()
    parent_ids = nodes["parent_id"].to_numpy()
    id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}

    parent_index = [-1] * len(node_ids)
    children = [[] for _ in node_ids]
    for child_index, parent_id in enumerate(parent_ids):
        if parent_id in id_to_index:
            parent_node_index = id_to_index[parent_id]
            parent_index[child_index] = parent_node_index
            children[parent_node_index].append(child_index)

    roots = [index for index, parent in enumerate(parent_index) if parent == -1]
    if len(roots) != 1:
        raise ValueError(f"Expected one healed tree, found {len(roots)} roots")

    traversal = []
    stack = roots.copy()
    while stack:
        current = stack.pop()
        traversal.append(current)
        stack.extend(children[current])

    if len(traversal) != len(node_ids):
        raise ValueError("Could not traverse every skeleton node")

    direct_pre_counts = Counter(neuron.presynapses["node_id"].dropna())
    direct_post_counts = Counter(neuron.postsynapses["node_id"].dropna())
    direct_pre = [direct_pre_counts[node_id] for node_id in node_ids]
    direct_post = [direct_post_counts[node_id] for node_id in node_ids]
    subtree_pre = direct_pre.copy()
    subtree_post = direct_post.copy()

    for current in reversed(traversal):
        parent = parent_index[current]
        if parent != -1:
            subtree_pre[parent] += subtree_pre[current]
            subtree_post[parent] += subtree_post[current]

    total_pre = sum(direct_pre)
    total_post = sum(direct_post)
    candidates = []

    for node_index, node_id in enumerate(node_ids):
        if parent_index[node_index] == -1 or len(children[node_index]) != 1:
            continue

        child_index = children[node_index][0]
        side_1_pre = subtree_pre[child_index]
        side_1_post = subtree_post[child_index]
        side_2_pre = total_pre - subtree_pre[node_index]
        side_2_post = total_post - subtree_post[node_index]
        si = _si_for_sides(
            side_1_pre,
            side_1_post,
            side_2_pre,
            side_2_post,
        )

        if isfinite(si):
            candidates.append({
                "node_id": node_id,
                "child_index": child_index,
                "SI": si,
                "side_1_pre": side_1_pre,
                "side_1_post": side_1_post,
                "side_2_pre": side_2_pre,
                "side_2_post": side_2_post,
            })

    if not candidates:
        raise ValueError("No valid node-level SI values")

    best = max(candidates, key=lambda candidate: candidate["SI"])

    side_1_indices = set()
    stack = [best["child_index"]]
    while stack:
        current = stack.pop()
        if current in side_1_indices:
            continue
        side_1_indices.add(current)
        stack.extend(children[current])

    side_1_ids = [node_ids[index] for index in side_1_indices]
    side_2_ids = [
        node_id
        for index, node_id in enumerate(node_ids)
        if index not in side_1_indices and node_id != best["node_id"]
    ]

    side_1_neuron = navis.subset_neuron(
        neuron,
        subset=side_1_ids,
        inplace=False,
    )
    side_2_neuron = navis.subset_neuron(
        neuron,
        subset=side_2_ids,
        inplace=False,
    )

    side_1_total = best["side_1_pre"] + best["side_1_post"]
    side_2_total = best["side_2_pre"] + best["side_2_post"]
    side_1_pre_fraction = (
        best["side_1_pre"] / side_1_total if side_1_total else float("nan")
    )
    side_2_pre_fraction = (
        best["side_2_pre"] / side_2_total if side_2_total else float("nan")
    )

    if side_1_pre_fraction >= side_2_pre_fraction:
        axon_neuron, dendrite_neuron = side_1_neuron, side_2_neuron
        axon_counts = best["side_1_pre"], side_1_total
        dendrite_counts = best["side_2_pre"], side_2_total
    else:
        axon_neuron, dendrite_neuron = side_2_neuron, side_1_neuron
        axon_counts = best["side_2_pre"], side_2_total
        dendrite_counts = best["side_1_pre"], side_1_total

    dendrite_canonical_synapses = dendrite_counts[1] - dendrite_counts[0]
    axon_canonical_synapses = axon_counts[0]

    print(f"Max SI: {best['SI']:.6f} at node {best['node_id']}")
    _print_canonical_fraction(
        "Dendrite",
        dendrite_canonical_synapses,
        dendrite_counts[1],
        "postsynapses",
    )
    _print_canonical_fraction(
        "Axon",
        axon_canonical_synapses,
        axon_counts[1],
        "presynapses",
    )

    split_neurons = navis.NeuronList([dendrite_neuron, axon_neuron])
    metrics = {
        "max_si": best["SI"],
        "max_si_node_id": best["node_id"],
        "dendrite_presynapses": dendrite_counts[0],
        "dendrite_postsynapses": dendrite_canonical_synapses,
        "dendrite_total_synapses": dendrite_counts[1],
        "dendrite_canonical_synapses": dendrite_canonical_synapses,
        "dendrite_canonical_percent": (
            100 * dendrite_canonical_synapses / dendrite_counts[1]
            if dendrite_counts[1]
            else float("nan")
        ),
        "axon_presynapses": axon_counts[0],
        "axon_total_synapses": axon_counts[1],
        "axon_canonical_synapses": axon_canonical_synapses,
        "axon_canonical_percent": (
            100 * axon_counts[0] / axon_counts[1]
            if axon_counts[1]
            else float("nan")
        ),
    }

    if return_metrics:
        return split_neurons, metrics
    return split_neurons


def _print_canonical_fraction(compartment, canonical_count, total_count, kind):
    if total_count:
        percentage = 100 * canonical_count / total_count
        print(
            f"{compartment}: {canonical_count}/{total_count} {kind} "
            f"({percentage:.2f}%)"
        )
    else:
        print(f"{compartment}: 0/0 {kind} (n/a)")
