import networkx as nx
import pandas as pd

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
