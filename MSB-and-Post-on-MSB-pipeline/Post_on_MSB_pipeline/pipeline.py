# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 14:10:33 2025

@author: user

Run on the princeton synapses

"""
#%%
import sys, os, gc, traceback
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import navis
from scipy.spatial.distance import cdist
#%%
import traceback
#%%
# # === custom imports ===
# sys.path.append(r"C:\Users\user\Desktop\Majd\clustering")
# from HDBSCAN_methods import *
# sys.path.append(r"C:\Users\user\Desktop\Majd\filopodia")
# from filopodia_methods import *

import sys
from pathlib import Path

# Resolve parent root (MSB-and-Post-on-MSB-pipeline)
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

# Add root to sys.path so Python can find MSB_pipeline
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import directly from MSB_pipeline!
from MSB_pipeline.clustering import upload_swc, heal_attach
from MSB_pipeline.config import Config

# Import specific functions
from pipeline_methods import extract_nodes_edges, create_swc_graph, node_test2 

#%%

# Path to the shared data folder at the root (one level up)
data_dir = current_dir.parent / "data"

# === configuration ===
allsynapses = pd.read_feather(data_dir / "sample_synapses.ftr")
#%%
# ✅ enforce Int64 to preserve large IDs safely
allsynapses['pre'] = allsynapses['pre'].astype('Int64')
allsynapses['post'] = allsynapses['post'].astype('Int64')
# Load the file safely
cluster_info = pd.read_feather(data_dir / "cluster_info_output.ftr")

centroids = (
    cluster_info
    .groupby(['neuron', 'cluster'])[['x', 'y', 'z']]
    .mean()
    .reset_index()
)

#%%
# # === helper: first stage ===
# def process_first_part(swc_id: int, allsynapses: pd.DataFrame = allsynapses, centroids: pd.DataFrame = centroids, cluster_info: pd.DataFrame = cluster_info) -> pd.DataFrame:
#     """Compute filtered filopodia synapses for a single neuron (was first loop)."""
#     allsynapses2 = allsynapses[(allsynapses['pre'] == swc_id) | (allsynapses['post'] == swc_id)]
#     swc, syn_ = upload_swc(swc_id, allsynapses2)
#     # # === NEW: Skip large neurons ===
#     # if len(swc.nodes) > 80000:
#     #     print(f"⚠️ Skipping neuron {swc_id} — {len(swc.nodes)} nodes (> 80000)")
#     #     return pd.DataFrame()  # return empty so main loop can continue
#     swc_att = heal_attach(swc, syn_)
#     nodes = swc.nodes
#     synapses = swc_att.connectors

#     clust_centroids = centroids[centroids['neuron'] == str(swc_id)]
#     cluster_info2 = cluster_info[cluster_info['neuron'] == str(swc_id)]
    
#     node_id_info = synapses[['x','y','z','node_id']]
#     cluster_info2 = cluster_info2.merge(node_id_info, on=['x','y','z'], how='left')
#     # Find nearest nodes for centroids
#     centroids_xyz = clust_centroids[['x', 'y', 'z']].to_numpy()
#     nodes_xyz = nodes[['x', 'y', 'z']].to_numpy()
#     dists = np.linalg.norm(centroids_xyz[:, None, :] - nodes_xyz[None, :, :], axis=2)
#     closest_node_indices = np.argmin(dists, axis=1)
#     clust_centroids['node_id'] = nodes.iloc[closest_node_indices]['node_id'].values

#     presynapses = synapses[synapses['type'] == 'pre'].copy()
#     synapses = synapses[synapses['type'] == 'post'].copy()

#     m = navis.geodesic_matrix(swc_att)
#     syn_ids = synapses['node_id'].values
#     centroid_ids = clust_centroids['node_id'].values

#     dist_matrix = m.loc[syn_ids, centroid_ids].to_numpy()
#     min_dists = dist_matrix.min(axis=1)
#     closest_centroid_indices = dist_matrix.argmin(axis=1)

#     mask = min_dists < 5500
#     filtered_synapses = synapses[mask].copy()
#     closest_indices = closest_centroid_indices[mask]

#     filtered_synapses['center_x'] = clust_centroids.iloc[closest_indices]['x'].values
#     filtered_synapses['center_y'] = clust_centroids.iloc[closest_indices]['y'].values
#     filtered_synapses['center_z'] = clust_centroids.iloc[closest_indices]['z'].values
#     filtered_synapses['center_node_id'] = clust_centroids.iloc[closest_indices]['node_id'].values

#     # === merge logic ===
#     df1_coords = cluster_info2[['x', 'y', 'z']].copy()
#     df1_coords.columns = ['pre_x', 'pre_y', 'pre_z']
#     filtered_df2 = allsynapses2.merge(df1_coords.drop_duplicates(), on=['pre_x', 'pre_y', 'pre_z'])

#     df2_coords = filtered_synapses[['x', 'y', 'z']].copy()
#     df2_coords.columns = ['post_x', 'post_y', 'post_z']
#     filtered_df3 = allsynapses2.merge(df2_coords.drop_duplicates(), on=['post_x', 'post_y', 'post_z'])

#     df2_coords = filtered_synapses[['x', 'y', 'z', 'center_x', 'center_y', 'center_z', 'center_node_id']].copy()
#     df2_coords.columns = ['post_x', 'post_y', 'post_z', 'center_x', 'center_y', 'center_z', 'center_node_id']
#     filtered_df3 = filtered_df3.merge(df2_coords.drop_duplicates(), on=['post_x', 'post_y', 'post_z'], how='left')

#     cluster_info_merge = clust_centroids[['node_id', 'cluster']].copy()
#     cluster_info_merge.columns = ['center_node_id', 'cluster']
#     filtered_df3 = filtered_df3.merge(cluster_info_merge, on='center_node_id', how='left')

#     filtered_synapses2 = filtered_synapses[['node_id', 'x', 'y', 'z']].rename(
#         columns={'x': 'post_x', 'y': 'post_y', 'z': 'post_z'}
#     )
#     final = filtered_df3.merge(filtered_synapses2, on=['post_x', 'post_y', 'post_z'], how='left')

#     # filter out postsynapses closer to non-clustered presynapses
#     clustered_node_ids = set(cluster_info2['node_id'])
#     all_presynaptic_node_ids = set(presynapses['node_id'])
#     non_clustered_node_ids = np.array(list(all_presynaptic_node_ids - clustered_node_ids))

#     keep_mask = []
#     for _, row in final.iterrows():
#         center_id = row['center_node_id']
#         node_id = row['node_id']
#         try:
#             d_current = m.loc[center_id, node_id]
#             d_non_clustered = m.loc[center_id, non_clustered_node_ids]
#             keep_mask.append(np.all(d_non_clustered >= d_current))
#         except KeyError:
#             keep_mask.append(True)

#     final_filtered = final[keep_mask].reset_index(drop=True)
#     # ✅ enforce Int64 dtypes before returning
#     for col in ['pre', 'post']:
#         if col in final_filtered.columns:
#             final_filtered[col] = final_filtered[col].astype('Int64')

#     return final_filtered


# # === helper: second stage ===
# def process_second_part(swc_id: int, filopodia_info: pd.DataFrame):
#     """Refine filopodia for one neuron (was second loop)."""
#     filtered_nodes_final = pd.DataFrame()
#     allsynapses_local = allsynapses[(allsynapses['pre'] == swc_id) | (allsynapses['post'] == swc_id)]

#     try:
#         swc, synapses = upload_swc(swc_id, allsynapses_local)
#         healed_neuron = heal_attach(swc, synapses)
#         nodes_df, connectors_df = extract_nodes_edges(healed_neuron)
#         G = create_swc_graph(nodes_df, connectors_df)
#         nodes = filopodia_info[filopodia_info['post'] == swc_id]
#         nodes_continue, nodes_stop = node_test2(nodes, G)
#         nodes_stop_int = set(int(float(x)) for x in nodes_stop)
#         filtered_nodes = nodes[nodes['node_id'].isin(nodes_stop_int)]

#         G_undirected = G.to_undirected()
#         connectors_pre = set(connectors_df.loc[connectors_df['type'] == 'pre', 'node_id'])

#         def find_base_node(row):
#             try:
#                 source = f"{row['node_id']}.0"
#                 target = f"{row['center_node_id']}.0"
#                 path = nx.shortest_path(G_undirected, source=source, target=target)
#                 path_int = [int(float(n)) for n in path]
#                 base_node = None
#                 for i, node in enumerate(path_int):
#                     if node in connectors_pre:
#                         base_node = node if i == 0 else path_int[i - 1]
#                         break
#                 if base_node is None:
#                     base_node = row['center_node_id']
#                 base_node_str = f"{base_node}.0"
#                 if G_undirected.degree(base_node_str) > 2:
#                     idx = path_int.index(base_node)
#                     if idx > 0:
#                         base_node = path_int[idx - 1]
#                 return base_node
#             except (nx.NetworkXNoPath, nx.NodeNotFound):
#                 return None

#         filtered_nodes['base_node'] = filtered_nodes.apply(find_base_node, axis=1)
#         filtered_nodes = filtered_nodes.sort_values('base_node').reset_index(drop=True)
#         unique_bases = {b: idx for idx, b in enumerate(sorted(filtered_nodes['base_node'].unique()))}
#         filtered_nodes['base'] = filtered_nodes['base_node'].map(unique_bases)
#         filtered_nodes_final = pd.concat([filtered_nodes_final, filtered_nodes], ignore_index=True)
#         # ✅ enforce Int64 dtypes before returning
#         for col in ['pre', 'post']:
#             if col in filtered_nodes_final.columns:
#                 filtered_nodes_final[col] = filtered_nodes_final[col].astype('Int64')
        
#         return filtered_nodes_final
    
#     except Exception as e:
#         print(f"  ⚠️ Error in second stage for neuron {swc_id}: {e}")
#         traceback.print_exc()
#         return pd.DataFrame({c: pd.Series(dtype='Int64') for c in ['pre', 'post']})
#%%

# === helper: first stage ===
def process_first_part(swc_id: int, allsynapses: pd.DataFrame = allsynapses, centroids: pd.DataFrame = centroids, cluster_info: pd.DataFrame = cluster_info) -> pd.DataFrame:
    """Compute filtered filopodia synapses for a single neuron (was first loop)."""
    allsynapses2 = allsynapses[(allsynapses['pre'] == swc_id) | (allsynapses['post'] == swc_id)]
    swc, syn_ = upload_swc(swc_id, allsynapses2)
    
    # === Skip large neurons ===
    if len(swc.nodes) > 80000:
        print(f"⚠️ Skipping neuron {swc_id} — {len(swc.nodes)} nodes (> 80000)")
        return pd.DataFrame()  # return empty so main loop can continue

    swc_att = heal_attach(swc, syn_)
    nodes = swc.nodes
    # print(nodes.head())
    synapses = swc_att.connectors

    clust_centroids = centroids[centroids['neuron'] == swc_id]
    cluster_info2 = cluster_info[cluster_info['neuron'] == swc_id]
    
    # Include synapse_id if present in connectors/synapses table
    coord_cols = ['x', 'y', 'z', 'node_id']
    if 'synapse_id' in synapses.columns:
        coord_cols.append('synapse_id')
    node_id_info = synapses[coord_cols]

    cluster_info2 = cluster_info2.merge(node_id_info, on=['x', 'y', 'z'], how='left')
    
    # Find nearest nodes for centroids
    centroids_xyz = clust_centroids[['x', 'y', 'z']].to_numpy()
    nodes_xyz = nodes[['x', 'y', 'z']].to_numpy()
    # print("closest_node_indices", centroids_xyz)
    dists = np.linalg.norm(centroids_xyz[:, None, :] - nodes_xyz[None, :, :], axis=2)
    # print("closest_node_indices", dists)
    closest_node_indices = np.argmin(dists, axis=1)
    clust_centroids['node_id'] = nodes.iloc[closest_node_indices]['node_id'].values

    presynapses = synapses[synapses['type'] == 'pre'].copy()
    synapses = synapses[synapses['type'] == 'post'].copy()
    m = navis.geodesic_matrix(swc_att)
    syn_ids = synapses['node_id'].values
    
    centroid_ids = clust_centroids['node_id'].values
    # print(centroid_ids)
    dist_matrix = m.loc[syn_ids, centroid_ids].to_numpy()
    min_dists = dist_matrix.min(axis=1)
    closest_centroid_indices = dist_matrix.argmin(axis=1)

    mask = min_dists < 5500
    filtered_synapses = synapses[mask].copy()
    closest_indices = closest_centroid_indices[mask]

    filtered_synapses['center_x'] = clust_centroids.iloc[closest_indices]['x'].values
    filtered_synapses['center_y'] = clust_centroids.iloc[closest_indices]['y'].values
    filtered_synapses['center_z'] = clust_centroids.iloc[closest_indices]['z'].values
    filtered_synapses['center_node_id'] = clust_centroids.iloc[closest_indices]['node_id'].values

    # === merge logic ===
    df1_coords = cluster_info2[['x', 'y', 'z']].copy()
    df1_coords.columns = ['pre_x', 'pre_y', 'pre_z']
    filtered_df2 = allsynapses2.merge(df1_coords.drop_duplicates(), on=['pre_x', 'pre_y', 'pre_z'])

    df2_coords = filtered_synapses[['x', 'y', 'z', 'center_x', 'center_y', 'center_z', 'center_node_id']].copy()
    df2_coords.columns = ['post_x', 'post_y', 'post_z', 'center_x', 'center_y', 'center_z', 'center_node_id']
    filtered_df3 = allsynapses2.merge(df2_coords.drop_duplicates(), on=['post_x', 'post_y', 'post_z'], how='left')

    cluster_info_merge = clust_centroids[['node_id', 'cluster']].copy()
    cluster_info_merge.columns = ['center_node_id', 'cluster']
    filtered_df3 = filtered_df3.merge(cluster_info_merge, on='center_node_id', how='left')

    # Retain synapse_id cleanly during final coordinate merge
    sub_cols = ['node_id', 'x', 'y', 'z']
    if 'synapse_id' in filtered_synapses.columns:
        sub_cols.append('synapse_id')

    filtered_synapses2 = filtered_synapses[sub_cols].rename(
        columns={'x': 'post_x', 'y': 'post_y', 'z': 'post_z'}
    )
    
    # Merge while resolving potential duplicate synapse_id columns cleanly
    final = filtered_df3.merge(
        filtered_synapses2, 
        on=['post_x', 'post_y', 'post_z'], 
        how='left',
        suffixes=('', '_syn')
    )
    if 'synapse_id_syn' in final.columns:
        final['synapse_id'] = final['synapse_id'].fillna(final['synapse_id_syn'])
        final.drop(columns=['synapse_id_syn'], inplace=True)

    # Filter out postsynapses closer to non-clustered presynapses
    clustered_node_ids = set(cluster_info2['node_id'])
    all_presynaptic_node_ids = set(presynapses['node_id'])
    non_clustered_node_ids = np.array(list(all_presynaptic_node_ids - clustered_node_ids))

    keep_mask = []
    for _, row in final.iterrows():
        center_id = row['center_node_id']
        node_id = row['node_id']
        try:
            d_current = m.loc[center_id, node_id]
            d_non_clustered = m.loc[center_id, non_clustered_node_ids]
            keep_mask.append(np.all(d_non_clustered >= d_current))
        except KeyError:
            keep_mask.append(True)

    final_filtered = final[keep_mask].reset_index(drop=True)
    
    # ✅ Enforce Int64 dtypes for IDs including synapse_id before returning
    for col in ['pre', 'post', 'synapse_id']:
        if col in final_filtered.columns:
            final_filtered[col] = final_filtered[col].astype('Int64')

    return final_filtered


# === helper: second stage ===
def process_second_part(swc_id: int, filopodia_info: pd.DataFrame):
    """Refine filopodia for one neuron (was second loop)."""
    filtered_nodes_final = pd.DataFrame()
    allsynapses_local = allsynapses[(allsynapses['pre'] == swc_id) | (allsynapses['post'] == swc_id)]
    # print("allsynapses_local", allsynapses_local.head())
    try:
        swc, synapses = upload_swc(swc_id, allsynapses_local)
        healed_neuron = heal_attach(swc, synapses)
        nodes_df, connectors_df = extract_nodes_edges(healed_neuron)
        G = create_swc_graph(nodes_df, connectors_df)
        # Filter out any synapses/nodes missing a node_id
        nodes = filopodia_info[filopodia_info['post'] == swc_id].dropna(subset=['node_id']).copy()

        if nodes.empty:
            print(f"⚠️ Skipping second stage for neuron {swc_id} — no valid nodes found.")
            return pd.DataFrame({c: pd.Series(dtype='Int64') for c in ['pre', 'post', 'synapse_id']})

        nodes_continue, nodes_stop = node_test2(nodes, G)
        nodes_stop_int = set(int(float(x)) for x in nodes_stop)
        filtered_nodes = nodes[nodes['node_id'].isin(nodes_stop_int)].copy()

        G_undirected = G.to_undirected()
        connectors_pre = set(connectors_df.loc[connectors_df['type'] == 'pre', 'node_id'])
        # print(f"  Found {len(filtered_nodes)} filtered nodes for neuron {swc_id}")
        def find_base_node(row):
            try:
                source = f"{row['node_id']}.0"
                target = f"{row['center_node_id']}.0"
                path = nx.shortest_path(G_undirected, source=source, target=target)
                path_int = [int(float(n)) for n in path]
                base_node = None
                for i, node in enumerate(path_int):
                    if node in connectors_pre:
                        base_node = node if i == 0 else path_int[i - 1]
                        break
                if base_node is None:
                    base_node = row['center_node_id']
                base_node_str = f"{base_node}.0"
                if G_undirected.degree(base_node_str) > 2:
                    idx = path_int.index(base_node)
                    if idx > 0:
                        base_node = path_int[idx - 1]
                return base_node
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None

        filtered_nodes['base_node'] = filtered_nodes.apply(find_base_node, axis=1)
        filtered_nodes = filtered_nodes.sort_values('base_node').reset_index(drop=True)
        unique_bases = {b: idx for idx, b in enumerate(sorted(filtered_nodes['base_node'].unique()))}
        filtered_nodes['base'] = filtered_nodes['base_node'].map(unique_bases)
        filtered_nodes_final = pd.concat([filtered_nodes_final, filtered_nodes], ignore_index=True)
        
        # ✅ Enforce Int64 dtypes for pre, post, and synapse_id before returning
        for col in ['pre', 'post', 'synapse_id']:
            if col in filtered_nodes_final.columns:
                filtered_nodes_final[col] = filtered_nodes_final[col].astype('Int64')
        
        return filtered_nodes_final
    
    except Exception as e:
        print(f"  ⚠️ Error in second stage for neuron {swc_id}: {e}")
        traceback.print_exc()
        # Explicitly preserve synapse_id in empty return schema
        return pd.DataFrame({c: pd.Series(dtype='Int64') for c in ['pre', 'post', 'synapse_id']})

#%%
# === main loop ===
final_combined_df = pd.DataFrame()
swc_files = list(data_dir.glob('**/*.swc'))
total = len(swc_files)
#%%
# PART = 1  # 1 for first half, 2 for second half

# half = total // 2
# if PART == 1:
#     swc_files = swc_files[:half]
#     print(f"Running on FIRST half ({len(swc_files)} neurons)")
# else:
#     swc_files = swc_files[half:]
#     print(f"Running on SECOND half ({len(swc_files)} neurons)")
    
    
for i, swc_file in enumerate(swc_files, start=1):
    swc_id = int(swc_file.stem)
    try:
        print(f"\n=== [{i}/{total}] Neuron {swc_id} ===")
        final_filtered = process_first_part(swc_id)
        filtered_nodes = process_second_part(swc_id, final_filtered)
        filtered_nodes['neuron_id'] = swc_id
        final_combined_df = pd.concat([final_combined_df, filtered_nodes], ignore_index=True)

        # if i % save_every == 0:
        #     path = fr"C:\Users\user\Desktop\majd\filopodia_files_prince\partial_filopodia_{i}.ftr"
        #     final_combined_df.to_feather(path)
        #     print(f"💾 Saved partial result to {path}")
            
        #     # Clear accumulated data to free memory
        #     final_combined_df = pd.DataFrame()
        #     gc.collect()
            
    except Exception as e:
        print(f"⚠️ Error processing neuron {swc_id}: {e}")
        traceback.print_exc()
        continue




# import pandas as pd
# from pathlib import Path

# # folder where partial files are saved
# folder = Path(r"C:\Users\user\Desktop\majd\filopodia_files_prince")

# # find all partial feather files
# files = sorted(folder.glob("partial_filopodia_*.ftr"), key=lambda f: int(f.stem.split('_')[-1]))

# print(f"Found {len(files)} partial files to combine.")

# # read and combine
# dfs = []
# for f in files:
#     print(f"Reading {f.name} ...")
#     df = pd.read_feather(f)
#     dfs.append(df)

# # concatenate all
# combined = pd.concat(dfs, ignore_index=True)
# print(f"Combined shape: {combined.shape}")

# enforce safe integer type for large IDs
for col in ['pre', 'post']:
    if col in final_combined_df.columns:
        final_combined_df[col] = final_combined_df[col].astype('Int64')
#%%
# save final combined file post_on_MSB.ftr in the data directory
final_path = data_dir / "post_on_MSB.ftr"
final_combined_df.to_feather(final_path)
print(f"✅ Saved final combined file: {final_path}")

# # === final save ===
# final_path = r"C:\Users\user\Desktop\majd\all_filopodia_combined_final.ftr"
# final_combined_df.to_feather(final_path)
# print(f"\n✅ Finished. Final result saved to {final_path}")

# %%
