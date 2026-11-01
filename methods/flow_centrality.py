"""Synapse flow centrality without the dense geodesic matrix.

navis 1.7.0 without navis-fastcore computes ``synapse_flow_centrality``
(mode 'sum') from a dense float64 geodesic matrix of synapse nodes by
skeleton nodes. For cells with tens of thousands of nodes that matrix takes
many gigabytes. The matrix is only used to ask, for each node, how many pre-
and postsynapses lie at or below it (distal to it, in the direction of the
leaves). Those counts are subtree sums, which a single pass from the leaves to
the root gives in memory linear in the number of nodes.

Everything after the counts follows navis's own code
(navis/morpho/mmetrics.py, ``synapse_flow_centrality``):

* the flow at a node with ``dpre`` presynapses and ``dpost`` postsynapses at
  or below it is ``(total_post - dpost) * dpre + dpost * (total_pre - dpre)``;
* navis computes this only for branch points and nodes that carry a
  connector of any type, and gives every other node the flow of its child
  (walking ``small_segments`` from the distal end). For every node except
  the root this equals the formula above. The root is a special case: navis
  types it 'root' even when it has several children, so it is never a
  branch point; if it carries no connector it takes the flow of the node
  before it in the first entry of ``small_segments`` that ends at the root,
  and if it carries a connector the formula gives 0 there;
* a branch point then gets the largest flow among its children (taken from
  the values before this step), and the column is cast to int.

``prepare_for_split`` puts the column on a copy of the neuron so that
``navis.split_axon_dendrite(..., metric='synapse_flow_centrality')`` finds it
and uses it instead of computing it. The split itself, including the soma
reroot of its own working copy, is left to navis.

The equality with navis holds for navis 1.7.0 without navis-fastcore, the
version the published tables were made with: with navis-fastcore installed,
navis computes the flow, and orders ``small_segments``, differently. Importing
this module stops on any other setup. A neuron the light calculation can't
handle is left to navis, which then builds the dense matrix again; its ID is
added to ``FALLBACK_IDS`` and a warning is logged.
"""

import logging

import numpy as np
import pandas as pd
import navis

logger = logging.getLogger(__name__)

METRIC = 'synapse_flow_centrality'
FALLBACK_IDS = []   # neurons left to navis's own (dense) calculation

if navis.__version__ != '1.7.0' or getattr(navis.utils, 'fastcore', None) is not None:
    raise ImportError(f'flow_centrality matches navis 1.7.0 without navis-fastcore; found navis '
                      f'{navis.__version__} with fastcore {getattr(navis.utils, "fastcore", None)!r}. '
                      'Install the pinned requirements (requirements.txt).')


def _connector_type_labels(x):
    """Return navis's (pre, post) labels for this neuron's connector types."""
    cn_types = x.connectors.type.unique()
    if any(np.isin(['pre', 'post'], cn_types)):
        return 'pre', 'post'
    if any(np.isin([0, 1], cn_types)):
        return 0, 1
    raise ValueError(f'Unable to parse connector types "{cn_types}" for neuron {x.id}')


def synapse_flow_centrality_light(x):
    """Return navis 1.7.0's synapse flow centrality (mode 'sum') for a TreeNeuron.

    The neuron must already be rooted where navis would compute the metric,
    i.e. at the soma if the split reroots to it. The neuron is not changed.

    Parameters
    ----------
    x : navis.TreeNeuron
        Single-rooted neuron with connectors (column ``node_id`` on every
        connector).

    Returns
    -------
    numpy.ndarray of int64
        One value per row of ``x.nodes``, in the same order.

    Raises
    ------
    ValueError
        In the cases where navis's own function raises before computing
        anything: no connectors, unreadable connector types, a pre- or
        postsynapse on a node that is not in the skeleton. Also when the
        neuron has more than one root, which navis's split refuses earlier.
    """
    if not isinstance(x, navis.TreeNeuron):
        raise ValueError(f'Expected TreeNeuron, got "{type(x)}"')
    if not x.has_connectors:
        raise ValueError('Neuron must have connectors.')

    nodes = x.nodes
    node_ids = nodes.node_id.to_numpy()
    n = len(node_ids)
    index = pd.Index(node_ids)
    parent_ix = index.get_indexer(nodes.parent_id.to_numpy())   # -1 for a root
    roots = np.flatnonzero(parent_ix < 0)
    if len(roots) != 1:
        raise ValueError(f'Neuron {x.id} has {len(roots)} roots.')
    root = roots[0]
    if np.any((parent_ix < 0) & (nodes.parent_id.to_numpy() >= 0)):
        raise ValueError(f'Neuron {x.id} has parents that are not nodes.')

    pre, post = _connector_type_labels(x)
    cn = x.connectors
    pre_ix = index.get_indexer(cn.loc[cn.type == pre, 'node_id'].to_numpy())
    post_ix = index.get_indexer(cn.loc[cn.type == post, 'node_id'].to_numpy())
    if np.any(pre_ix < 0) or np.any(post_ix < 0):
        raise ValueError('Node/vertex IDs not present in the skeleton.')
    n_pre = np.bincount(pre_ix, minlength=n).astype(np.int64)
    n_post = np.bincount(post_ix, minlength=n).astype(np.int64)
    total_pre = np.int64(len(pre_ix))
    total_post = np.int64(len(post_ix))

    # Order nodes so that every node comes before its parent: depth-first from
    # the root, then reversed.
    child_order = np.argsort(parent_ix, kind='stable')
    sorted_parents = parent_ix[child_order]
    first = np.searchsorted(sorted_parents, np.arange(n), side='left')
    last = np.searchsorted(sorted_parents, np.arange(n), side='right')
    order = np.empty(n, dtype=np.int64)
    stack = [root]
    k = 0
    while stack:
        v = stack.pop()
        order[k] = v
        k += 1
        stack.extend(child_order[first[v]:last[v]].tolist())
    if k != n:
        raise ValueError(f'Neuron {x.id}: {n - k} nodes are not connected to the root.')

    # Synapses at or below each node.
    d_pre = n_pre.copy()
    d_post = n_post.copy()
    for v in order[:0:-1]:          # every node but the root, leaves first
        p = parent_ix[v]
        d_pre[p] += d_pre[v]
        d_post[p] += d_post[v]

    flow = (total_post - d_post) * d_pre + d_post * (total_pre - d_pre)

    # Root: never a branch point in navis. Without a connector of any type it
    # takes the flow its first small segment brings to it.
    has_connector = index.isin(cn.node_id.to_numpy())
    if not has_connector[root] and last[root] > first[root]:
        root_id = node_ids[root]
        seg = next(s for s in x.small_segments if s[-1] == root_id)
        flow[root] = flow[index.get_loc(seg[-2])]

    # Branch points take the largest flow among their children.
    is_bp = (nodes['type'] == 'branch').to_numpy()
    out = flow.copy()
    for v in np.flatnonzero(is_bp):
        out[v] = flow[child_order[first[v]:last[v]]].max()
    return out.astype(np.int64)


def prepare_for_split(x):
    """Return a copy of ``x`` that carries a precomputed flow column for the split.

    ``navis.split_axon_dendrite`` computes the metric only if the node table
    does not already have it. This function computes the column on a copy
    rerooted exactly as the split reroots its working copy (to the soma, if
    there is one and it is not already the root), and writes it, matched by
    node ID, onto an unrerooted copy of ``x``. The split's pieces are cut from
    the neuron passed in, so leaving the copy unrerooted keeps them as they
    were; their node tables gain the one extra column.

    Anything that is not a TreeNeuron (for example the string
    'heal or attach issue') is returned unchanged; a NeuronList is handled
    neuron by neuron. When navis would refuse the neuron before computing the
    metric, or when the light calculation raises, the neuron is returned
    unchanged and navis behaves as it does without this module (raising, or computing the
    metric itself).
    """
    if isinstance(x, navis.NeuronList):
        return navis.NeuronList([prepare_for_split(n) for n in x])
    if not isinstance(x, navis.TreeNeuron):
        return x
    if METRIC in x.nodes.columns:
        return x
    if not x.has_connectors or len(x.root) > 1:
        return x
    try:
        rooted = x.copy()
        if np.any(rooted.soma) and not np.all(np.isin(rooted.soma, rooted.root)):
            rooted.reroot(rooted.soma, inplace=True)
        flow = pd.Series(synapse_flow_centrality_light(rooted),
                         index=rooted.nodes.node_id.to_numpy())
        del rooted
    except Exception as e:
        FALLBACK_IDS.append(x.id)
        logger.warning(f'Light flow centrality failed for neuron {x.id} ({e!r}); '
                       'navis will compute it itself.')
        return x
    out = x.copy()
    out.nodes[METRIC] = flow.reindex(out.nodes.node_id.to_numpy()).to_numpy()
    return out
