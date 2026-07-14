#%%
# import time
# import sys
import navis
import pandas as pd
import numpy as np
import os
# import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from pathlib import Path

# sys.path.append(r'C:\Users\majd_\Desktop\CS\5)Project\clustering')
# from ng_methods_v2 import *
# import random
# from HDBSCAN_methods import *
# import uuid
# edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
# import subprocess
# import gc
# import matplotlib.colors as mcolors
# import matplotlib.cm as cm
# import pickle
from scipy.spatial.distance import cdist
# def test():
#     print("test")
#%%
def attach_synapses(x,syn, min_score=30,neuropils=False,batch_size=10, progress=True):
   
    # Parse root IDs
        # Drop below threshold connections
        # if min_score:
        #     syn = syn[syn.cleft_score >= min_score]


        if isinstance(x, navis.core.BaseNeuron):
            x = navis.NeuronList([x])

        if isinstance(x, navis.NeuronList):
            for n in x:
                presyn = postsyn = pd.DataFrame([])
                add_cols = ['neuropil'] if neuropils else []
                cols = ['pre_x', 'pre_y', 'pre_z',
                            'post', 'synapse_id'] + add_cols
                presyn = syn.loc[syn.pre == np.int64(n.id), cols
                                     ].rename({'pre_x': 'x',
                                               'pre_y': 'y',
                                               'pre_z': 'z',
                                               'post': 'partner_id'},
                                              axis=1)
                presyn['type'] = 'pre'
                cols = ['post_x', 'post_y', 'post_z',
                            'pre', 'synapse_id'] + add_cols
                postsyn = syn.loc[syn.post == np.int64(n.id), cols
                                      ].rename({'post_x': 'x',
                                                'post_y': 'y',
                                                'post_z': 'z',
                                                'pre': 'partner_id'},
                                               axis=1)
                postsyn['type'] = 'post'

                connectors = pd.concat((presyn, postsyn), axis=0, ignore_index=True)

                # Turn type column into categorical to save memory
                connectors['type'] = connectors['type'].astype('category')

                # If TreeNeuron, map each synapse to a node
                if isinstance(n, navis.TreeNeuron):
                    tree = navis.neuron2KDTree(n, data='nodes')
                    dist, ix = tree.query(connectors[['x', 'y', 'z']].values)

                    too_far = dist > 10_000
                    if any(too_far):
                        connectors = connectors[~too_far].copy()
                        ix = ix[~too_far]

                    connectors['node_id'] = n.nodes.node_id.values[ix]

                    # Add an ID column for navis' sake
                    connectors.insert(0, 'connector_id', np.arange(connectors.shape[0]))

                n.connectors = connectors
        return x


# This will automatically build the correct absolute path, e.g.,
# C:\Users\majd_\Desktop\CS\5)Project\clustering\swc

def upload_swc(id_,allsynapses):
    # 1. Get the directory of the current script
    script_dir = Path(__file__).resolve().parent

    # 2. Point to the 'swc' folder inside it
    input_base_dir = script_dir / "swc"
    
    id_=str(id_)+'.swc'
    # Get the list of subdirectories in the input directory
    
    input_subfolders = {f.name for f in os.scandir(input_base_dir) if f.is_dir()}
    for folder_name in input_subfolders:
        input_folder = os.path.join(input_base_dir, folder_name)
        swc_path = os.path.join(input_folder, id_)
        try:
            swc = navis.read_swc(swc_path)
            allsynapses2 = allsynapses[allsynapses['pre'] == int(id_[:-4])]
            return swc,allsynapses2
        except:
            continue


def heal_attach(item,allsynapses):
    try:
        healed_neurons=navis.heal_skeleton(item)
        healed_neurons_att_syn=attach_synapses(healed_neurons,allsynapses)
        return healed_neurons_att_syn
    except:
        return 'heal or attach issue'


def merge_close_clusters_hybrid(df, distance_threshold=450):
    """
    Merges clusters in the DataFrame if any pair of points between two clusters 
    are within the distance_threshold. Uses centroids for fast filtering, 
    then confirms with pairwise distances.
    """
    second_threshold = 10000
    merged = True
    while merged:
        merged = False
        centroids = df.groupby('cluster')[['x', 'y', 'z']].mean()
        clusters = centroids.index.to_list()
        centroid_coords = centroids.values

        # Precompute centroid distances once per loop
        centroid_distances = cdist(centroid_coords, centroid_coords)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Skip if centroid distance too large
                if centroid_distances[i, j] >= second_threshold:
                    continue
                
                # Get actual points in each cluster
                c1 = df[df['cluster'] == clusters[i]][['x', 'y', 'z']].values
                c2 = df[df['cluster'] == clusters[j]][['x', 'y', 'z']].values

                # Do the expensive check only if centroid check passed
                real_distances = cdist(c1, c2)
                if real_distances.min() < distance_threshold:
                    # Merge j into i
                    df.loc[df['cluster'] == clusters[j], 'cluster'] = clusters[i]
                    merged = True
                    break
            if merged:
                break

    # Reassign cluster IDs to be consecutive integers
    unique_clusters = df['cluster'].unique()
    cluster_mapping = {old: new for new, old in enumerate(sorted(unique_clusters))}
    df['cluster'] = df['cluster'].map(cluster_mapping)

    return df


### original method
#%%
def prepare_clustering(allsynapses,swc_id):
    print(f"Preparing clustering for neuron {swc_id}...")
    allsynapses2 = allsynapses[(allsynapses['pre'] == int(swc_id))]
    # ignore neurons with less than 5 synapses
    if len(allsynapses2) < 5:
        return None
    
    swc,syn_=upload_swc(swc_id,allsynapses2)

    swc_att=heal_attach(swc,syn_)

    synapses = swc_att.connectors
    
    m = navis.geodesic_matrix(swc_att)
    # Ensure each synapse is treated independently
    synapses = synapses[synapses['type'] == 'pre'].copy()
    
    # # Create a unique synapse identifier
    synapses['synapse_identifier'] = np.arange(len(synapses))
    
    # Map each synapse to its node_id
    # synapse_to_node = dict(zip(synapses['synapse_identifier'], synapses['node_id']))
    
    # Create a new expanded geodesic matrix based on synapses, not nodes
    synapse_indices = synapses['synapse_identifier']
    node_ids = synapses['node_id']
    
    # Expand m_filtered by mapping each synapse_identifier to its node_id
    m_expanded = m.loc[node_ids, node_ids].copy()
    
    # Rename the rows and columns to use synapse IDs instead of node IDs
    m_expanded.index = synapse_indices
    m_expanded.columns = synapse_indices
    
    # hdb = HDBSCAN(min_cluster_size=4, min_samples=3, metric='precomputed', cluster_selection_epsilon=900)
    db = DBSCAN(eps=900, min_samples=3,metric='precomputed')
    db.fit(m_expanded)
    
    if all(label == -1 for label in db.labels_):
        return None
    
    # Create mapping from synapse_identifier to cluster label
    synapse_to_cluster = {synapse: cluster for synapse, cluster in zip(m_expanded.index, db.labels_)}
    
    # Assign clusters back to synapses
    synapses['cluster'] = synapses['synapse_identifier'].map(synapse_to_cluster)
    
    # Filter out noise (-1 cluster)
    synapses_clustered = synapses[synapses['cluster'] != -1]
    
    # synapses_clustered = merge_close_clusters(synapses_clustered, distance_threshold=500)
    
    synapses_clustered = merge_close_clusters_hybrid(synapses_clustered, distance_threshold=450)
    synapses_clustered = synapses_clustered.sort_values(by='cluster').reset_index(drop=True)
    synapses_clustered = synapses_clustered[['synapse_id','node_id','partner_id', 'x', 'y', 'z', 'neuron', 'cluster']]
    return synapses_clustered