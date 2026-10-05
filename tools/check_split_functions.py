"""Check step 08's split code against an independent reference, on small trees.

Step 08 (processing/08_alternative_split_methods.py) scores every candidate cut of a
neuron: each node that is not the root and has exactly one child. For a candidate, side 1
is the subtree of its child, side 2 is everything outside its own subtree, and the
synapses on the node itself are left out. Each cut gets an SI and a two-sided Fisher p on
the 2x2 table (side x pre/post). MaxSI keeps the cut with the highest SI, MinFisherP the cut
with the lowest p, and the axon is the side with the higher presynaptic fraction. Step 08
computes |phi| (absolute_phi) only after the batches are combined, for the SFC cut and the
MinFisherP cut; this tool checks absolute_phi on every candidate's table.

This tool takes step 08's code out of the file without running the file (its top level
loads the full synapse table) and runs step 08's own batch function,
run_neuron_batch_pipeline, on navis neurons built from small trees. Only the loading and
the SFC split are replaced: the skeleton and its synapses come from the test tree, and the
published SFC split is a stand-in, so the SFC part is not tested here. Everything else is step 08's code. The
results are compared with a reference written separately (it finds the two sides by
removing the node and walking the tree), for every candidate and for the summary step 08
saves:
  - the candidate set, the four side counts and the two excluded counts (exact),
  - SI and Fisher p (relative tolerance 1e-12),
  - |phi| from step 08's absolute_phi against sqrt(chi^2 / n) from scipy,
  - that the MaxSI and MinFisherP nodes are among the reference's best (ties allowed, and
    counted in an INFO line, since step 08 breaks them by sort order),
  - the side counts saved for both chosen nodes, and the axon side (including its tie and
    empty-side cases), and that the split and the axon neuron carry the counted synapses,
  - that neurons with no valid SI, no candidate or two roots fail,
  - random trees, a 20,000-node chain and re-rooted trees.
Hand-worked values for three trees check the reference itself. Not checked: the columns
IG, fisher_q and the odds ratios, and the SFC split.

Run from the repository root:
    python tools/check_split_functions.py
    python tools/check_split_functions.py --random 1000 --seed 1
    python tools/check_split_functions.py --script path/to/another_version.py
PASS means step 08's per-node scores, its MaxSI and MinFisherP choices, its saved side
counts, axon side and |phi| agree with the reference on every tree. The tool prints one line
per check, writes the same to outputs/check/split_functions/report.txt and exits with
status 1 if anything fails. About 40 seconds with the defaults.
"""
import argparse
import ast
import builtins
import contextlib
import hashlib
import io
import math
import sys
import time
import warnings
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import navis
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from config import OUTPUT_DIR  # noqa: E402

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
STEP08 = REPO / "processing" / "08_alternative_split_methods.py"
FUNCTIONS = ["SI_calc", "calc_s", "safe_SI_calc", "calculate_node_SI", "get_descendants_from_node",
             "split_neuron_by_si_node", "assign_axon_and_dendrite", "find_neuron_id_column",
             "run_neuron_batch_pipeline", "absolute_phi"]
STUBBED = {"upload_swc", "heal_attach_princeton_non_process", "get_split"}  # loading, replaced below
TABLE = ["side_1_pre", "side_1_post", "side_2_pre", "side_2_post"]
COUNTS = TABLE + ["node_pre_excluded", "node_post_excluded"]


# ------------------------------------------------------------------ step 08's own code
def load_step08(path, trees):
    """Step 08's functions, with only the loading replaced by the test trees."""
    source = Path(path).read_text(encoding="utf-8-sig")
    try:
        tree, escaped = ast.parse(source), False
    except SyntaxError:
        # Some original versions keep a Windows path with a bare \U in a dead string, which
        # does not parse. Doubling every backslash leaves code without backslashes unchanged.
        tree, escaped = ast.parse(source.replace("\\", "\\\\")), True
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    missing = [f for f in FUNCTIONS if f not in defs]
    if missing:
        sys.exit(f"{path}: functions not found: {missing}")
    if escaped and any("\\" in ast.unparse(defs[f]) for f in FUNCTIONS):
        sys.exit(f"{path} only parses with its backslashes doubled, which would change these functions.")
    namespace = {"np": np, "pd": pd, "navis": navis, "log": math.log,
                 "fisher_exact": fisher_exact, "multipletests": multipletests,
                 "upload_swc": lambda neuron_id: trees[neuron_id]["neuron"].copy(),
                 "heal_attach_princeton_non_process": lambda swc, synapses: swc,
                 "get_split": lambda neuron, flow_thresh=1: navis.NeuronList([neuron, neuron])}
    free = set()
    for f in FUNCTIONS:  # names each function reads but neither binds itself nor gets from here
        body = list(ast.walk(defs[f]))
        loads = {n.id for n in body if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        bound = {n.id for n in body if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}
        bound |= {a.arg for a in body if isinstance(a, ast.arg)}
        bound |= {n.name for n in body if isinstance(n, ast.ExceptHandler) and n.name}
        bound |= {(al.asname or al.name).split(".")[0] for n in body if isinstance(n, (ast.Import, ast.ImportFrom))
                  for al in n.names}
        free |= loads - bound
    free -= set(FUNCTIONS) | set(namespace) | set(dir(builtins))
    if free:
        sys.exit(f"{path}: the step 08 code uses names this tool does not provide: {sorted(free)}")
    exec(compile(ast.Module(body=[defs[f] for f in FUNCTIONS], type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def run_step08(f, trees):
    """Run step 08's batch function on the trees; returns (summary, node table, failures)."""
    ids = list(trees)
    synapses = pd.DataFrame({"pre": ids, "post": ids})  # only used to find "any synapse of this neuron"
    with contextlib.redirect_stdout(io.StringIO()):  # step 08 prints a line per neuron
        summary, nodes, failures = f["run_neuron_batch_pipeline"](
            nodes_dataframe=pd.DataFrame({"neuron": ids}), synapses_dataframe=synapses,
            neuron_id_col="neuron", flow_thresh=1, plot=False)
    return summary.set_index("neuron_id") if len(summary) else summary, nodes, failures


# ----------------------------------------------------------------- independent reference
def entropy(pre, post):
    n = pre + post
    if n == 0 or pre == 0 or post == 0:
        return 0.0
    p, q = pre / n, post / n
    return -(p * math.log2(p) + q * math.log2(q))


def ref_si(a, b, c, d):
    n = a + b + c + d
    snorm = entropy(a + c, b + d)
    if n == 0 or snorm == 0:
        return math.nan
    return 1 - ((a + b) * entropy(a, b) + (c + d) * entropy(c, d)) / n / snorm


def ref_p(a, b, c, d):
    if a + b == 0 or c + d == 0:
        return math.nan
    return float(fisher_exact([[a, b], [c, d]], alternative="two-sided")[1])


def ref_phi(a, b, c, d):
    """|phi| as sqrt(chi^2 / n), uncorrected chi-square; NaN when a margin is empty."""
    if min(a + b, c + d, a + c, b + d) == 0:
        return math.nan
    return math.sqrt(chi2_contingency([[a, b], [c, d]], correction=False)[0] / (a + b + c + d))


def ref_axon(a, b, c, d):
    """Step 08's stated rule: side 1 if its pre fraction is at least side 2's; an empty side
    has no fraction, and a comparison with it is false, so the axon is then side 2."""
    if a + b == 0 or c + d == 0:
        return 2
    return 1 if a * (c + d) >= c * (a + b) else 2


def walk(adjacent, start, blocked):
    seen, queue = {start}, deque([start])
    while queue:
        for w in adjacent[queue.popleft()]:
            if w != blocked and w not in seen:
                seen.add(w)
                queue.append(w)
    return seen


def reference(tree):
    """One row per candidate cut; sides found by removing the node and walking the tree."""
    parent = dict(zip(tree["nodes"]["node_id"], tree["nodes"]["parent_id"]))
    roots = [n for n, p in parent.items() if p == -1]
    if len(roots) != 1:
        raise ValueError(f"expected one root, found {len(roots)}")
    adjacent, children = {n: [] for n in parent}, {n: [] for n in parent}
    for n, p in parent.items():
        if p != -1:
            adjacent[n].append(p)
            adjacent[p].append(n)
            children[p].append(n)
    pre, post = Counter(tree["pre"]), Counter(tree["post"])  # synapses per node id
    count = lambda counts, nodes: sum(counts[n] for n in nodes)  # noqa: E731
    rows = []
    for v in parent:
        if v == roots[0] or len(children[v]) != 1:
            continue
        side1, side2 = walk(adjacent, children[v][0], v), walk(adjacent, parent[v], v)
        a, b, c, d = count(pre, side1), count(post, side1), count(pre, side2), count(post, side2)
        rows.append(dict(node_id=v, child_id=children[v][0], side_1_pre=a, side_1_post=b, side_2_pre=c,
                         side_2_post=d, node_pre_excluded=pre[v],
                         node_post_excluded=post[v],
                         SI=ref_si(a, b, c, d), fisher_p=ref_p(a, b, c, d), phi=ref_phi(a, b, c, d)))
    return pd.DataFrame(rows, columns=["node_id", "child_id"] + COUNTS + ["SI", "fisher_p", "phi"])


# ------------------------------------------------------------------------- test trees
def chain(n):
    return pd.DataFrame({"node_id": range(n), "parent_id": [-1] + list(range(n - 1))})


def nodes(pairs):
    return pd.DataFrame(pairs, columns=["node_id", "parent_id"])


def named_trees():
    """Small trees, each aimed at one behaviour. pre/post list the node of each synapse."""
    shifted = lambda i: i * 7919 + 1000003  # noqa: E731  non-contiguous ids
    return [
        dict(name="chain", nodes=chain(6), pre=[2, 4, 5, 5], post=[0, 0, 1, 2]),
        dict(name="clean chain", nodes=chain(5), pre=[3, 4, 4], post=[0, 1, 1]),
        dict(name="synapses on the cut node", nodes=chain(5), pre=[0, 0, 0, 4, 4, 4], post=[2] * 10 + [0, 4]),
        dict(name="root with one child", nodes=chain(4), pre=[3, 3], post=[0, 0, 0]),
        dict(name="boundary at a fork", nodes=nodes([(0, -1), (1, 0), (2, 1), (3, 2), (4, 2), (5, 3), (6, 4), (7, 5)]),
             pre=[4, 6], post=[2, 3, 5, 7]),
        dict(name="node with three children", nodes=nodes([(0, -1), (1, 0), (2, 1), (3, 2), (4, 2), (5, 2), (6, 3),
                                                           (7, 4)]), pre=[6, 6, 7], post=[0, 1, 1, 5]),
        dict(name="empty child side", nodes=chain(5), pre=[0], post=[1, 1]),
        dict(name="equal pre fractions at the best cut", nodes=chain(5), pre=[0, 4], post=[0, 0, 4, 4]),
        dict(name="tie, swapped tables", nodes=chain(5), pre=[0, 0, 4, 4], post=[2, 2, 2]),
        dict(name="tie, SI = 1 with different p", nodes=chain(4), pre=[2, 3, 3], post=[0, 0]),
        dict(name="tie, identical tables", nodes=chain(6), pre=[5, 5, 5], post=[0, 0]),
        dict(name="p underflows to 0", nodes=chain(5), pre=[2] * 1000 + [4] * 3000, post=[0] * 3000),
        dict(name="large mixed counts", nodes=chain(6), pre=[0] * 40 + [3] * 2500 + [5] * 2500,
             post=[0] * 3000 + [1] * 500 + [5] * 60),
        dict(name="rows out of order", nodes=nodes([(70, 50), (90, 70), (10, 30), (50, 30), (30, -1), (11, 10),
                                                    (12, 11)]), pre=[90, 90, 70], post=[12, 11, 30]),
        dict(name="non-contiguous ids, shuffled",
             nodes=nodes([(shifted(i), shifted(i - 1) if i else -1) for i in range(6)]).sample(frac=1, random_state=3),
             pre=[shifted(i) for i in (2, 4, 5, 5)], post=[shifted(i) for i in (0, 0, 1, 2)]),
        dict(name="input only (no SI: must fail)", nodes=chain(5), pre=[], post=[0, 1, 2, 3, 4],
             fails="No valid node-level SI values"),
        dict(name="output only (no SI: must fail)", nodes=chain(5), pre=[0, 1, 2, 3, 4, 4], post=[],
             fails="No valid node-level SI values"),
        dict(name="no synapses (must fail)", nodes=chain(4), pre=[], post=[], fails="No valid node-level SI values"),
        # Step 08 has no explicit check for this case: its dropna on an empty table raises KeyError 'SI'.
        dict(name="two nodes, no candidate (must fail; step 08 raises KeyError 'SI')", nodes=chain(2), pre=[1],
             post=[0], fails="'SI'"),
        dict(name="two roots (must fail)", nodes=nodes([(0, -1), (1, 0), (2, -1), (3, 2)]), pre=[1], post=[3],
             fails="found 2 roots"),
    ]


def random_tree(seed, n_nodes, n_synapses):
    """A random tree with long unbranched runs, like a skeleton, and depth-biased polarity."""
    rng = np.random.default_rng(seed)
    parent = [-1] + [i - 1 if rng.random() < 0.7 else int(rng.integers(0, i)) for i in range(1, n_nodes)]
    ids = rng.choice(10 ** 9, size=n_nodes, replace=False) + 10 ** 6
    tree = pd.DataFrame({"node_id": ids, "parent_id": [-1 if p == -1 else ids[p] for p in parent]})
    on = rng.integers(0, n_nodes, size=n_synapses)
    is_pre = rng.random(n_synapses) < np.where(on > rng.random() * n_nodes, 0.8, 0.2)
    return dict(name=f"random {seed}", nodes=tree.sample(frac=1, random_state=seed),
                pre=list(ids[on[is_pre]]), post=list(ids[on[~is_pre]]))


def hand_checks():
    """Values worked out by hand; they check the reference before it checks step 08."""
    by = {t["name"]: t for t in named_trees()}
    expect = []
    r = reference(by["chain"]).set_index("node_id")
    expect += [("chain, cut at 1: counts", tuple(r.loc[1, COUNTS]) == (4, 1, 0, 2, 0, 1)),
               ("chain, cut at 1: SI 0.4766056", abs(r.loc[1, "SI"] - 0.4766056) < 5e-7),
               ("chain, cut at 2: counts", tuple(r.loc[2, COUNTS]) == (3, 0, 0, 3, 1, 1)),
               ("chain, cut at 2: SI 1, p 0.1, |phi| 1", r.loc[2, "SI"] == 1 and abs(r.loc[2, "fisher_p"] - 0.1) < 1e-12
                and abs(r.loc[2, "phi"] - 1) < 1e-12),
               ("chain, cut at 3: SI 0.5487949, p 1/7, |phi| 0.7745967", abs(r.loc[3, "SI"] - 0.5487949) < 5e-7
                and abs(r.loc[3, "fisher_p"] - 1 / 7) < 1e-12 and abs(r.loc[3, "phi"] - 0.7745967) < 5e-7)]
    r = reference(by["tie, swapped tables"]).set_index("node_id")
    expect += [("swapped tables: SI 0.2960653 at both ends", abs(r.loc[1, "SI"] - 0.2960653) < 5e-7
                and r.loc[1, "SI"] == r.loc[3, "SI"]),
               ("swapped tables: no SI at the middle", math.isnan(r.loc[2, "SI"]))]
    r = reference(by["boundary at a fork"]).set_index("node_id")
    expect += [("fork: candidates 1, 3, 4, 5", sorted(r.index) == [1, 3, 4, 5]),
               ("fork: SI 0.4325377 at 3, 1 at 4, 0 at 1", abs(r.loc[3, "SI"] - 0.4325377) < 5e-7
                and r.loc[4, "SI"] == 1 and r.loc[1, "SI"] == 0)]
    return [name for name, ok in expect if not ok]


# ------------------------------------------------------------------------- comparison
def as_neuron(tree, neuron_id):
    """A navis TreeNeuron of the tree, with its synapses attached by node id."""
    n = tree["nodes"].assign(y=0.0, z=0.0, radius=1.0, label=0)
    n["x"] = np.arange(len(n), dtype=float)
    out = navis.TreeNeuron(n[["node_id", "label", "x", "y", "z", "radius", "parent_id"]], id=neuron_id)
    connectors = pd.DataFrame({"node_id": list(tree["pre"]) + list(tree["post"]),
                               "type": ["pre"] * len(tree["pre"]) + ["post"] * len(tree["post"])})
    connectors[["x", "y", "z"]] = 0.0
    connectors["type"] = connectors["type"].astype("category")
    connectors.insert(0, "connector_id", np.arange(len(connectors)))
    out.connectors = connectors
    return out


def same(x, y, rel=1e-12):
    if pd.isna(x) or pd.isna(y):
        return bool(pd.isna(x) and pd.isna(y))
    return math.isclose(float(x), float(y), rel_tol=rel, abs_tol=1e-15)


def check_neuron(f, tree, nid, summary, node_table, failed, ties):
    """Problems for one tree (empty list = pass). ties counts picks decided by a tie."""
    try:
        ref = reference(tree)
    except ValueError:
        ref = None
    has_si = ref is not None and ref["SI"].notna().any()
    if tree.get("fails") or not has_si:
        if nid not in failed or nid in summary.index:
            return ["step 08 kept a neuron that has no valid cut"]
        reason = tree.get("fails") or "No valid node-level SI values"
        return [] if reason in str(failed[nid]) else [f"step 08 failed it for another reason: {failed[nid]}"]
    if nid not in summary.index:
        return [f"step 08 failed the neuron: {failed.get(nid, '?')}"]
    got = node_table[node_table["neuron_id"] == nid].set_index("node_id")
    r = ref.set_index("node_id")
    if set(got.index) != set(r.index):
        return [f"candidates differ: step 08 {sorted(got.index)[:6]}, reference {sorted(r.index)[:6]}"]
    r = r.loc[got.index]
    problems = [f"{c} differs at {int((got[c].astype(np.int64) != r[c].astype(np.int64)).sum())} candidates"
                for c in COUNTS + ["child_id"] if (got[c].astype(np.int64) != r[c].astype(np.int64)).any()]
    for c in ("SI", "fisher_p"):
        bad = [i for i in got.index if not same(got.at[i, c], r.at[i, c])]
        if bad:
            problems.append(f"{c} differs at {len(bad)} candidates (node {int(bad[0])}: {got.at[bad[0], c]} vs "
                            f"{r.at[bad[0], c]})")
    phi_bad = [i for i in r.index if not same(f["absolute_phi"](*(float(r.at[i, c]) for c in TABLE)), r.at[i, "phi"],
                                              rel=1e-9)]
    if phi_bad:
        problems.append(f"absolute_phi differs at {len(phi_bad)} candidates")
    s = summary.loc[nid]
    for label, node_col, side_col, score, best in (("MaxSI", "best_node_id", "best_side_", "SI", max),
                                                   ("MinFisherP", "best_fisher_node_id", "best_fisher_side_", "fisher_p", min)):
        if r[score].notna().sum() == 0:
            continue
        top = r[score].dropna()
        winners = {i for i in top.index if same(top[i], best(top))}
        node = int(s[node_col])
        if node not in winners:
            problems.append(f"{label} picked node {node}, not one of the reference's best {sorted(winners)[:4]}")
            continue
        saved = tuple(int(s[f"{side_col}{k}"]) for k in ("1_pre", "1_post", "2_pre", "2_post"))
        if saved != tuple(int(r.at[node, c]) for c in TABLE):
            problems.append(f"{label}: saved side counts {saved} are not node {node}'s")
        tables = {tuple(int(r.at[w, c]) for c in TABLE) for w in winners}
        if len(winners) > 1:
            ties[label] += 1
            ties[label + " (different cuts)"] += len({min(t, (t[2], t[3], t[0], t[1])) for t in tables}) > 1
    a, b, c, d = (int(s[f"best_side_{k}"]) for k in ("1_pre", "1_post", "2_pre", "2_post"))
    if int(s["axon_side"]) != ref_axon(a, b, c, d):
        problems.append(f"axon is side {int(s['axon_side'])}, the rule gives side {ref_axon(a, b, c, d)}")
    neuron = tree["neuron"]
    split = f["split_neuron_by_si_node"](neuron, got.reset_index(), node_id=int(s["best_node_id"]))
    carried = (len(split["side_1_neuron"].presynapses), len(split["side_1_neuron"].postsynapses),
               len(split["side_2_neuron"].presynapses), len(split["side_2_neuron"].postsynapses))
    if carried != (a, b, c, d):
        problems.append(f"the split carries {carried} synapses, the counts say {(a, b, c, d)}")
    polarity = f["assign_axon_and_dendrite"](split)
    axon = polarity["axon_neuron"]
    if (len(axon.presynapses), len(axon.postsynapses)) != ((a, b) if polarity["axon_side"] == 1 else (c, d)):
        problems.append("the returned axon neuron is not the axon side")
    return problems


def check_trees(f_path, trees):
    """Run step 08 on all trees at once and check each; returns {name: problems}, tie counts."""
    by_id = {}
    for i, tree in enumerate(trees, start=1):
        tree["neuron"] = as_neuron(tree, i)
        by_id[i] = tree
    f = load_step08(f_path, by_id)
    summary, node_table, failures = run_step08(f, by_id)
    failed = dict(zip(failures["neuron_id"], failures["error"])) if len(failures) else {}
    ties = {k: 0 for k in ("MaxSI", "MaxSI (different cuts)", "MinFisherP", "MinFisherP (different cuts)")}
    results = {}
    for nid, tree in by_id.items():
        try:
            results[tree["name"]] = check_neuron(f, tree, nid, summary, node_table, failed, ties)
        except Exception as e:  # a crash in a check is a failure, not a stop
            results[tree["name"]] = [f"check crashed: {e!r}"]
    return results, ties, f


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--script", type=Path, default=STEP08,
                    help="file holding the step 08 functions (default: processing/08_alternative_split_methods.py)")
    ap.add_argument("--random", type=int, default=300, help="number of random trees (default 300)")
    ap.add_argument("--seed", type=int, default=0, help="first random-tree seed (default 0)")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "check" / "split_functions",
                    help="output folder (default outputs/check/split_functions)")
    args = ap.parse_args()
    if args.random < 0:
        ap.error("--random must be 0 or more")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.txt").unlink(missing_ok=True)  # never leave an earlier run's PASS behind

    digest = hashlib.sha256(args.script.read_bytes()).hexdigest()[:16]
    header = [f"Checked {datetime.now():%Y-%m-%d %H:%M}", f"Step 08 code: {args.script} (sha256 {digest})", ""]
    print("\n".join(header))
    lines, failures = [], 0

    def report(ok, text, info=False):
        nonlocal failures
        failures += not ok and not info
        lines.append(f"{'INFO' if info else 'PASS' if ok else 'FAIL'}  {text}")
        print(lines[-1])

    wrong = hand_checks()
    report(not wrong, "reference reproduces the hand-worked values" + (f": wrong {wrong}" if wrong else ""))

    start = time.time()
    named = named_trees()
    results, ties, f = check_trees(args.script, named)
    for tree in named:
        problems = results[tree["name"]]
        report(not problems, tree["name"] + ("" if not problems else ": " + "; ".join(problems)))

    randoms = []
    for seed in range(args.seed, args.seed + args.random):
        size = np.random.default_rng(seed + 10 ** 6)
        randoms.append(random_tree(seed, int(size.integers(3, 400)), int(size.integers(0, 600))))
    if randoms:
        r_results, r_ties, _ = check_trees(args.script, randoms)
        bad = [(t["name"], r_results[t["name"]]) for t in randoms if r_results[t["name"]]]
        report(not bad, f"{len(randoms)} random trees (seeds {args.seed}-{args.seed + args.random - 1})"
               + ("" if not bad else f": {len(bad)} with problems, first {bad[0][0]}: {'; '.join(bad[0][1])}"))
        for k, v in ties.items():
            ties[k] = v + r_ties[k]
    report(True, "picks decided by a tie (step 08 breaks ties by sort order): "
           + ", ".join(f"{k} {v}" for k, v in ties.items()), info=True)

    n_long = 20000
    try:
        tree = dict(name="long chain", nodes=chain(n_long), pre=list(range(n_long - 100, n_long)), post=list(range(100)))
        t0 = time.time()
        got = f["calculate_node_SI"](as_neuron(tree, 1))
        middle = got[(got.node_id >= 100) & (got.node_id < n_long - 100)]
        ok = len(got) == n_long - 2 and bool((middle.SI == 1).all())
        report(ok, f"{n_long:,}-node chain: {len(got):,} candidates, SI = 1 at every cut between the groups "
                   f"({time.time() - t0:.1f} s)")
    except Exception as e:
        report(False, f"{n_long:,}-node chain: {e!r}")

    try:
        differ, compared = 0, 0
        for seed in range(30):
            neuron = as_neuron(random_tree(seed, 120, 200), 1)
            a = f["calculate_node_SI"](neuron).set_index("node_id")
            leaf = int(neuron.leafs.node_id.iloc[seed % len(neuron.leafs)])
            b = f["calculate_node_SI"](navis.reroot_skeleton(neuron, leaf, inplace=False)).set_index("node_id")
            for v in a.index.intersection(b.index):
                compared += 1
                x, y = tuple(a.loc[v, TABLE]), tuple(b.loc[v, TABLE])
                differ += not (x == y or x == (y[2], y[3], y[0], y[1])) or not same(a.at[v, "SI"], b.at[v, "SI"])
        report(differ == 0, f"re-rooting 30 trees: {compared} common candidates, {differ} with a different table or SI")
    except Exception as e:
        report(False, f"re-rooting: {e!r}")

    summary = f"\n{sum(l.startswith('PASS') for l in lines)} passed, {failures} failed ({time.time() - start:.0f} s)"
    print(summary)
    (args.out / "report.txt").write_text("\n".join(header + lines) + summary + "\n")
    print(f"Report: {args.out / 'report.txt'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    # No inputs needed: it checks the step 08 code in
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/processing/08_alternative_split_methods.py
    # on test trees it builds itself (named ones in this file and 300 random ones), and writes
    #   /Users/ohajyahia/PycharmProjects/mixed-polarity-paper/outputs/check/split_functions/report.txt
    # To test with more random trees, uncomment:
    # sys.argv[1:] = ["--random", "1000", "--seed", "1"]
    main()
