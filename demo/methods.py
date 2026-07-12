import navis
import pandas as pd
import numpy as np

def attach_synapses_princeton_nonprocess(x,syn,neuropils=False,batch_size=100, progress=True):


        if isinstance(x, navis.core.BaseNeuron):
            x = navis.NeuronList([x])

        if isinstance(x, navis.NeuronList):
            for n in x:
                presyn = postsyn = pd.DataFrame([])
                add_cols = ['neuropil'] if neuropils else []
                cols = ['pre_x', 'pre_y', 'pre_z',
                            'post'] + add_cols
                presyn = syn.loc[syn.pre == np.int64(n.id), cols
                                     ].rename({'pre_x': 'x',
                                               'pre_y': 'y',
                                               'pre_z': 'z',
                                               'post': 'partner_id'},
                                              axis=1)
                presyn['type'] = 'pre'
                cols = ['post_x', 'post_y', 'post_z',
                            'pre'] + add_cols
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
def get_split(item,flow_thresh=1):
    try:
        split=navis.split_axon_dendrite(item, metric='synapse_flow_centrality',reroot_soma=True,flow_thresh=flow_thresh)
        return split
    except:
        return 'split issue'



def heal_attach(item,allsynapses):
    try:
        healed_neurons=navis.heal_skeleton(item)
        healed_neurons_att_syn=attach_synapses(healed_neurons,allsynapses)
        return healed_neurons_att_syn
    except:
        return 'heal or attach issue'