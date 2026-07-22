from __future__ import annotations

from pathlib import Path

import navis
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

from .config import Config


def attach_synapses(
    x,
    syn: pd.DataFrame,
    min_score: int = 30,
    neuropils: bool = False,
    batch_size: int = 10,
    progress: bool = True,
):
    """Attach synapse annotations to a neuron or neuron list."""
    if isinstance(x, navis.core.BaseNeuron):
        x = navis.NeuronList([x])

    if isinstance(x, navis.NeuronList):
        for n in x:
            presyn = pd.DataFrame([])
            postsyn = pd.DataFrame([])
            add_cols = ["neuropil"] if neuropils else []
            cols = ["pre_x", "pre_y", "pre_z", "post", "synapse_id"] + add_cols
            presyn = syn.loc[syn.pre == np.int64(n.id), cols].rename(
                {"pre_x": "x", "pre_y": "y", "pre_z": "z", "post": "partner_id"},
                axis=1,
            )
            presyn["type"] = "pre"

            cols = ["post_x", "post_y", "post_z", "pre", "synapse_id"] + add_cols
            postsyn = syn.loc[syn.post == np.int64(n.id), cols].rename(
                {"post_x": "x", "post_y": "y", "post_z": "z", "pre": "partner_id"},
                axis=1,
            )
            postsyn["type"] = "post"

            connectors = pd.concat((presyn, postsyn), axis=0, ignore_index=True)
            connectors["type"] = connectors["type"].astype("category")
            connectors["neuron"] = np.int64(n.id)

            if isinstance(n, navis.TreeNeuron):
                tree = navis.neuron2KDTree(n, data="nodes")
                dist, ix = tree.query(connectors[["x", "y", "z"]].values)

                too_far = dist > 10_000
                if any(too_far):
                    connectors = connectors[~too_far].copy()
                    ix = ix[~too_far]

                connectors["node_id"] = n.nodes.node_id.values[ix]
                connectors.insert(0, "connector_id", np.arange(connectors.shape[0]))

            n.connectors = connectors
    return x


def upload_swc(swc_id: int | str, allsynapses: pd.DataFrame, swc_dir: str | Path | None = None):
    """Load a neuron SWC file from the local project directory."""
    base_dir = Path(swc_dir or (Path(__file__).resolve().parent.parent / "data" / "swc")).resolve()
    swc_name = f"{swc_id}.swc"

    swc_path = None
    for candidate in base_dir.rglob(swc_name):
        if candidate.is_file():
            swc_path = candidate
            break

    if swc_path is None:
        raise FileNotFoundError(f"SWC file not found for neuron {swc_id} in {base_dir}")

    swc = navis.read_swc(swc_path)
    synapses = allsynapses[
        (allsynapses["pre"] == int(swc_id)) | (allsynapses["post"] == int(swc_id))
    ].copy()
    return swc, synapses


def heal_attach(item, allsynapses):
    try:
        healed_neurons = navis.heal_skeleton(item)
        return attach_synapses(healed_neurons, allsynapses)
    except Exception:
        return "heal or attach issue"


def merge_close_clusters_hybrid(df: pd.DataFrame, distance_threshold: int = 450) -> pd.DataFrame:
    """Merge nearby clusters using centroid filtering before a full distance check."""
    df = df.copy()
    second_threshold = 10_000
    merged = True
    while merged:
        merged = False
        centroids = df.groupby("cluster")[["x", "y", "z"]].mean()
        clusters = centroids.index.to_list()
        centroid_coords = centroids.values
        centroid_distances = cdist(centroid_coords, centroid_coords)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if centroid_distances[i, j] >= second_threshold:
                    continue

                c1 = df[df["cluster"] == clusters[i]][["x", "y", "z"]].values
                c2 = df[df["cluster"] == clusters[j]][["x", "y", "z"]].values
                real_distances = cdist(c1, c2)
                if real_distances.min() < distance_threshold:
                    df.loc[df["cluster"] == clusters[j], "cluster"] = clusters[i]
                    merged = True
                    break
            if merged:
                break

    unique_clusters = df["cluster"].unique()
    cluster_mapping = {old: new for new, old in enumerate(sorted(unique_clusters))}
    df["cluster"] = df["cluster"].map(cluster_mapping)
    return df


def prepare_clustering(allsynapses: pd.DataFrame, swc_id: int | str, config: Config | None = None):
    """Cluster the pre-synaptic connectors of a single neuron using DBSCAN on geodesic distances."""
    cfg = config or Config()
    swc_id = int(swc_id)
    synapses_subset = allsynapses[(allsynapses["pre"] == swc_id)]

    if len(synapses_subset) < 5:
        return None

    swc, synapses = upload_swc(swc_id, synapses_subset, swc_dir=cfg.swc_dir)
    swc_att = heal_attach(swc, synapses)
    if isinstance(swc_att, str):
        return None

    synapses = swc_att.connectors
    if synapses.empty:
        return None

    synapses = synapses[synapses["type"] == "pre"].copy()
    synapses["synapse_identifier"] = np.arange(len(synapses))

    geodesic_matrix = navis.geodesic_matrix(swc_att)
    synapse_indices = synapses["synapse_identifier"]
    node_ids = synapses["node_id"]
    matrix_expanded = geodesic_matrix.loc[node_ids, node_ids].copy()
    matrix_expanded.index = synapse_indices
    matrix_expanded.columns = synapse_indices

    db = DBSCAN(eps=cfg.dbscan_eps, min_samples=cfg.min_samples, metric="precomputed")
    db.fit(matrix_expanded)

    if all(label == -1 for label in db.labels_):
        return None

    synapse_to_cluster = {synapse: cluster for synapse, cluster in zip(matrix_expanded.index, db.labels_)}
    synapses["cluster"] = synapses["synapse_identifier"].map(synapse_to_cluster)
    synapses_clustered = synapses[synapses["cluster"] != -1]
    synapses_clustered = merge_close_clusters_hybrid(synapses_clustered, distance_threshold=cfg.merge_distance_threshold)
    synapses_clustered = synapses_clustered.sort_values(by="cluster").reset_index(drop=True)
    return synapses_clustered[["synapse_id", "node_id", "partner_id", "x", "y", "z", "neuron", "cluster"]]


def append_matched_synapses(
    cluster_info: pd.DataFrame,
    allsynapses: pd.DataFrame,
    distance_threshold: int = 450,
) -> pd.DataFrame:
    """Match unclustered synapses to the nearest clustered point for each neuron."""
    cluster_info = cluster_info.copy()
    cluster_info = cluster_info.drop(columns=[col for col in ("node_id",) if col in cluster_info.columns])

    valid_neurons = set(cluster_info["neuron"].dropna().astype(int))
    cluster_synapse_ids = set(cluster_info["synapse_id"].dropna().astype(int))

    mask_neuron = allsynapses["pre"].isin(valid_neurons)
    mask_synapse = ~allsynapses["synapse_id"].isin(cluster_synapse_ids)
    allsynapses1 = allsynapses.loc[mask_neuron & mask_synapse, ["synapse_id", "pre", "post", "pre_x", "pre_y", "pre_z"]].copy()
    allsynapses1 = allsynapses1.rename(
        columns={"pre": "neuron", "post": "partner_id", "pre_x": "x", "pre_y": "y", "pre_z": "z"}
    )

    new_rows = []
    for neuron, neuron_df in cluster_info.groupby("neuron", sort=False):
        try:
            coords_cluster = neuron_df[["x", "y", "z"]].values
            tree = cKDTree(coords_cluster)
            syn_df = allsynapses1[allsynapses1["neuron"] == neuron]
            if syn_df.empty:
                continue

            coords_syn = syn_df[["x", "y", "z"]].values
            dist, idx = tree.query(coords_syn, k=1)
            mask = dist < distance_threshold
            if not mask.any():
                continue

            close_syn = syn_df.loc[mask].copy()
            close_syn["cluster"] = neuron_df.iloc[idx[mask]]["cluster"].values
            new_rows.append(close_syn)
        except Exception:
            continue

    if new_rows:
        cluster_info = pd.concat([cluster_info] + new_rows, ignore_index=True)
        cluster_info = cluster_info.sort_values(by=["neuron", "cluster"]).reset_index(drop=True)

    return cluster_info
