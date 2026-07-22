# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 14:44:35 2025

@author: majd_

@description: filopodia algorithm methods 
"""

import networkx as nx
import pandas as pd
import numpy as np
import os

def extract_nodes_edges(healed_neuron):

    nodes_df = healed_neuron.nodes[['node_id', 'parent_id', 'x', 'y']]
    nodes_df['node_id']=nodes_df['node_id'].astype(float)
    nodes_df['parent_id']=nodes_df['parent_id'].astype(float)
    
    connectors_df = healed_neuron.connectors[['connector_id', 'node_id', 'type', 'x', 'y']]
    connectors_df['connector_id']=connectors_df['connector_id'].astype(float)
    connectors_df['node_id']=connectors_df['node_id'].astype(float)
    
    return nodes_df,connectors_df

def create_swc_graph(nodes_df,connectors_df):
# Create graph
    G = nx.DiGraph()
    skeleton_nodes_set = set()
    
    # Add skeleton nodes with 'n' prefix
    for _, row in nodes_df.iterrows():
        node_id = row['node_id']
        node_name = f'{node_id}'
        G.add_node(node_name, pos=(row['x'], row['y']))
        skeleton_nodes_set.add(node_id)
    
    
    # Add skeleton edges
    for _, row in nodes_df.iterrows():
        node = f'{row["node_id"]}'
        parent_id = row["parent_id"]
        if parent_id != -1 and parent_id in skeleton_nodes_set:
            parent = f'{parent_id}'
            G.add_edge(parent, node)
    
    for _, row in connectors_df.iterrows():
        connector_id = row['connector_id']
        skeleton_node_id = row['node_id']
        syn_type = row['type']
        
        if syn_type=='pre':
            
            connector_name = f'pre{connector_id}'
        else:
            
            connector_name = f'post{connector_id}'     
         
            
        skeleton_name = f'{skeleton_node_id}'
    
        # Add connector node
        G.add_node(connector_name, syn_type=syn_type, pos=(row['x'], row['y']))
    
        # Add edge to skeleton node
        G.add_edge(skeleton_name,connector_name)
        
    return G
def extract_graph_info(G):
    # Get positions of all nodes
    pos = nx.get_node_attributes(G, 'pos')
    nodes_with_pos = list(pos.keys())
    
    # Color map for nodes
    color_map = []
    for node in nodes_with_pos:
    
            syn_type = G.nodes[node].get('syn_type')
            color_map.append('red' if syn_type == 'pre' else 'blue' if syn_type == 'post' else 'gray')
         
    # Labels (node names)
    labels = {node: node for node in nodes_with_pos}
    return labels,color_map,pos,nodes_with_pos

def node_test2(nodes,G):
    
    leafs=[]
    nodes_continue=[]
    nodes_stop=[]
    for node in nodes.iterrows():
        #print(node[1][['node_id']][0])
       
        node=str(float(node[1][['node_id']][0]))
        dfv=pd.DataFrame(nx.descendants(G, node))
        
        if len(dfv)==0:
            leafs.append(node)
        else:
            dfv.columns=['node_ids']
            haspre=dfv[dfv['node_ids'].str.contains('pre')]
            if len(haspre)==0:
                nodes_stop.append(node)
            else:
                nodes_continue.append(node)
    return nodes_continue,nodes_stop

def extract_info_of_continued_nodes(nodes_stop,healed_neuron):
    n=healed_neuron.nodes[['node_id','x','y','z']]
    n['node_id']=n['node_id'].astype(float)
    nodes_stop=pd.DataFrame(nodes_stop)
    nodes_stop.columns=['node_id']
    nodes_stop['node_id']=nodes_stop['node_id'].astype(float)
    
    
    comb_nodes=n.merge(nodes_stop,on='node_id',how='right')
    
    comb_nodes['x']=(comb_nodes['x']/4).astype(np.int64)
    comb_nodes['y']=(comb_nodes['y']/4).astype(np.int64)
    comb_nodes['z']=(comb_nodes['z']/40).astype(np.int64)
    return comb_nodes
def extract_passed_test2_nodes_coord(healed_neuron,nodes_stop):
    nc=healed_neuron.connectors[['node_id','x','y','z']]
    nodes_stop=pd.DataFrame(nodes_stop)
    nodes_stop.columns=['node_id']
    nodes_stop['node_id']=nodes_stop['node_id'].astype(float)
    nodes_stop['node_id']=nodes_stop['node_id'].astype(np.int64)
    nc=nc.merge(nodes_stop, on='node_id',how='right')
    nc['x']=(nc['x']/4).astype(np.int64)
    nc['y']=(nc['y']/4).astype(np.int64)
    nc['z']=(nc['z']/40).astype(np.int64)
    return nc



def get_synapses(nid,syn_df,pos):
    return syn_df[syn_df[pos]==nid]

def get_points(syn_df,pos):
    return syn_df[[pos,pos+'_x',pos+'_y',pos+'_z']]

def turn_to_tuple(points_df):
    points_df=points_df.copy()
    points_df.columns=['neuron','x','y','z']
    points_df=points_df.drop(columns=['neuron'])
    points_df['xyz'] = points_df.apply(lambda row: (int(row['x']/4), int(row['y']/4), int(row['z']/40)), axis=1)
    return points_df['xyz']


