
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    METHODS_DIR, NEURON_TABLE_FTR, SYNAPSE_TABLE_FTR,
    RECIPROCITY_LIST_FTR, ALL_R_FULL_PKL, SELF_SDI_FTR,
    FULL_RECI_CONNECTIONS_FTR, CONNECTIONS_FILTERED_FTR,
    CONNECTIONS_RECI_FILTERED_FTR, CONNECTIONS_TABLE_FTR,
    DERIVED_DATA_DIR,
)
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import os
import seaborn as sns
import pickle
#%%  ---------------------shouldnt we do reci before keeping only AD/DD/AA/DA???!!!!!
#%%
#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
#%%
aaa=nodesG[nodesG['root_id']!=nodesG['neuron']]
#%%
# The synapse table (about 77 million rows) is counted with pyarrow: only the columns and the four types used here,
# one count per (pre, post, comp). The pandas groupby, pivot_table and networkx graph this replaces held several
# copies of it in memory.
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
syn=feather.read_table(SYNAPSE_TABLE_FTR, columns=['pre','post','comp'])
syn=syn.filter(pc.is_in(syn['comp'], value_set=pa.array(['AD','DA','DD','AA'])))
connections=(syn.group_by(['pre','post','comp']).aggregate([([], 'count_all')]).to_pandas()
             .rename(columns={'count_all':'synapse_count'}))
del syn
#%%
# One row per (pre, post) and one column per type, as pd.pivot_table makes it: rows and columns sorted, float values,
# NaN where a pair has none of a type
wide=connections.set_index(['pre','post','comp'])['synapse_count'].unstack('comp').sort_index().astype(float)
del connections
connections_p=wide.copy()
connections_p['synapse_count']=connections_p.sum(axis=1)   # the row sum that connections_p.T.sum() gave
connections_p=connections_p.reset_index()
#%%
# Reciprocal pairs: (A, B) with A < B where both A -> B and B -> A exist, sorted by (A, B), with the synapses of both
# directions added; this is what the networkx loop gave
assert (connections_p['pre'] != connections_p['post']).all()   # the old loop kept self pairs; this join would drop them
forward=connections_p[['pre','post','synapse_count']]
backward=forward.rename(columns={'pre':'post','post':'pre','synapse_count':'reverse_count'})
pairs=forward.merge(backward, on=['pre','post'], how='inner')
pairs=pairs[pairs['pre'] < pairs['post']].sort_values(['pre','post'])
reciprocal_edges_df=pd.DataFrame({'Node A': pairs['pre'].to_numpy(), 'Node B': pairs['post'].to_numpy(),
                                  'Synapse Count': (pairs['synapse_count'] + pairs['reverse_count']).to_numpy()})
del forward, backward, pairs
# (the blank lines below keep the commented-out block that follows at lines 69-171)





RECIPROCITY_LIST_FTR.parent.mkdir(parents=True, exist_ok=True)
reciprocal_edges_df.to_feather(RECIPROCITY_LIST_FTR)
#%%ר
reci_df=pd.read_feather(RECIPROCITY_LIST_FTR)
#%%%



#%%reci cal per neuron
r'''
allreci_neurons=pd.DataFrame(pd.concat([reci_df['Node A'],reci_df['Node B']]).unique(),columns=['neuron'])

import time
ii = 0

all_r = []
total_start_time = time.time() 
for n1_id in allreci_neurons['neuron'].iloc[0:]:
    if ii%100==0:
        print(ii)
        total_end_time = time.time()
        print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds")
    ii += 1
    start_time = time.time()  
    
    reci_partners1 = reci_df[reci_df['Node A'] == n1_id]
    reci_partners2 = reci_df[reci_df['Node B'] == n1_id]
    reci_partners = pd.DataFrame(
        pd.concat([reci_partners1['Node A'], reci_partners1['Node B'], reci_partners2['Node A'], reci_partners2['Node B']]),
        columns=['neuron']
    )
    reci_partners = reci_partners[reci_partners['neuron'] != n1_id]
    
    r_list = []
    swc1 = upload_swc(n1_id)
    allsynapses2 = allsynapses[(allsynapses['pre'] == n1_id) | (allsynapses['post'] == n1_id)]
    swc1 = heal_attach(swc1, allsynapses2)
    r_list.append(n1_id)
    
    for n2_id in reci_partners['neuron']:
        shared_synapses = allsynapses2[
            ((allsynapses2['pre'] == n1_id) & (allsynapses2['post'] == n2_id)) |
            ((allsynapses2['pre'] == n2_id) & (allsynapses2['post'] == n1_id))
        ]
        
        n1_id_pre = shared_synapses[shared_synapses['pre'] == n1_id][['pre_x', 'pre_y', 'pre_z']]
        n1_id_pre.columns = ['x', 'y', 'z']
        n1_id_post = shared_synapses[shared_synapses['post'] == n1_id][['post_x', 'post_y', 'post_z']]
        n1_id_post.columns = ['x', 'y', 'z']
        
        pre_syn = swc1.connectors.merge(n1_id_pre[['x', 'y', 'z']], on=['x', 'y', 'z'], how='inner')['node_id']
        post_syn = swc1.connectors.merge(n1_id_post[['x', 'y', 'z']], on=['x', 'y', 'z'], how='inner')['node_id']
        
        gm = navis.geodesic_matrix(swc1, from_=pre_syn)
        gm_expanded = gm.loc[pre_syn.values]
        gm_expanded = gm_expanded.loc[:, post_syn.values]
        
        r_list.append([n2_id, gm_expanded.mean().mean(), gm_expanded.shape])
    
    all_r.append(r_list)
    
    end_time = time.time()  # End time counter for this iteration
    
# Print total execution time

total_end_time = time.time()
print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds")
#%%
import pickle
file_path = r"C:\Users\user\organised_work\data\783\generated\reciprocity\calculations\partners\all_r_full_.pkl"

# Ensure the directory exists
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# Save the object
with open(file_path, "wb") as file:
    pickle.dump(all_r, file)

#%%
# Define the file path
file_path = r"C:\Users\user\organised_work\data\783\generated\reciprocity\calculations\partners\all_r_full_.pkl"

# Load the object
with open(file_path, "rb") as file:
    all_r = pickle.load(file)

all_dfs=[]
for i in all_r:
    neuron=i[0]
    partners=i[1:]
    df=pd.DataFrame(partners)
    df['neuron']=str(neuron)
    
    df.columns=['partner','dist','shape','neuron']
    all_dfs.append(df)

df_all=pd.DataFrame(pd.concat([i for i in all_dfs]))

    #%%
df_all['partner']=df_all['partner'].astype(np.int64)
df_all['neuron']=df_all['neuron'].astype(np.int64)


all_self=pd.read_feather(r'C:\Users\user\organised_work\data\783\generated\reciprocity\calculations\self_SDI.ftr')
#%%
all_self['neruon']=all_self['neruon'].astype(np.int64)
#%%
all_self_f = all_self[all_self['neruon'].isin(df_all['neuron'])]
all_self_f.columns=['neuron','SDI']
all_self_f['neuron']=all_self_f['neuron'].astype(np.int64)
#%%
'''
#------------------------------------------------------------------------------
#%%
#%%
# forward (A -> B)
final_df = reci_df.merge(
    connections_p,
    left_on=['Node A','Node B'],
    right_on=['pre','post'],
    how='left',
    suffixes=('', '_drop')
).fillna(0)

final_df = final_df.rename(columns={
    'pre': 'pre_AtoB',
    'post': 'post_AtoB',
    'AA': 'node_A_pre_AA',
    'AD': 'node_A_pre_AD',
    'DA': 'node_A_pre_DA',
    'DD': 'node_A_pre_DD',
    'synapse_count': 'A->B_synapse_count'
})

# backward (B -> A)
final_df = final_df.merge(
    connections_p,
    left_on=['Node B','Node A'],
    right_on=['pre','post'],
    how='left',
    suffixes=('', '_drop')
).fillna(0)

final_df = final_df.rename(columns={
    'pre': 'pre_BtoA',
    'post': 'post_BtoA',
    'AA': 'node_B_pre_AA',
    'AD': 'node_B_pre_AD',
    'DA': 'node_B_pre_DA',
    'DD': 'node_B_pre_DD',
    'synapse_count': 'B->A_synapse_count'
})

#%%
final_df['shape'] = list(zip(final_df['A->B_synapse_count'].astype(int),
                             final_df['B->A_synapse_count'].astype(int)))
final_df[['dim_A->B','dim_B->A']] = pd.DataFrame(final_df['shape'].tolist(),
                                                 index=final_df.index)
#%%

nodesG.columns

#%%
#------------------------------------------------
final_df=final_df.merge(nodesG[['neuron','primary_type','SI']],left_on='Node A',right_on='neuron',how='left')
#%%
final_df=final_df.merge(nodesG[['neuron','primary_type','SI']],left_on='Node B',right_on='neuron',how='left')


#%%
final_df=final_df[['Node A', 'Node B', 
       'node_A_pre_AA', 'node_A_pre_AD', 'node_A_pre_DA', 'node_A_pre_DD',
       'A->B_synapse_count',  'node_B_pre_AA',
       'node_B_pre_AD', 'node_B_pre_DA', 'node_B_pre_DD', 'B->A_synapse_count',
       'shape', 'dim_A->B', 'dim_B->A', 'primary_type_x', 'SI_x',
        'primary_type_y', 'SI_y']]
#%%
final_df.columns=['Node A', 'Node B', 'node_A_pre_AA', 'node_A_pre_AD', 'node_A_pre_DA',
       'node_A_pre_DD', 'A->B_synapse_count', 'node_B_pre_AA', 'node_B_pre_AD',
       'node_B_pre_DA', 'node_B_pre_DD', 'B->A_synapse_count', 'shape',
       'dim_A->B', 'dim_B->A', 'primary_type_A', 'SI_A', 'primary_type_B',
       'SI_B']

#%%

FULL_RECI_CONNECTIONS_FTR.parent.mkdir(parents=True, exist_ok=True)   # the four tables below share its folder
final_df.to_feather(FULL_RECI_CONNECTIONS_FTR)
#%%
#%%
# the same counts and pivot as above, not computed again
#%%

#%% five comnnection doesnt change synaptic type distribution
connection_pivot=wide; del wide   # so a re-run of the cell above stops instead of adding sum_syn twice

connection_pivot['sum_syn']=connection_pivot.sum(axis=1)   # the row sum that .T.sum() gave
connection_pivot=connection_pivot.fillna(0)
#%%
connection_pivot.to_feather(CONNECTIONS_FILTERED_FTR)
#%% creating reciprocal connections table
connections=pd.read_feather(CONNECTIONS_FILTERED_FTR).reset_index()
reci_A=final_df[['Node A','Node B']]
reci_B=final_df[['Node B','Node A']]
reci_B.columns=['Node A','Node B']
all_reci=pd.concat([reci_A,reci_B])

connections['reciprocal'] = connections.set_index(['pre', 'post']).index.isin(all_reci.set_index(['Node A', 'Node B']).index).astype(int)
#%%
connections.to_feather(CONNECTIONS_RECI_FILTERED_FTR)
#%%

connections=pd.read_feather(CONNECTIONS_RECI_FILTERED_FTR)
#%%
connections_types=connections.merge(nodesG[['neuron','primary_type']],left_on='pre',right_on='neuron',how='left')
connections_types=connections_types.merge(nodesG[['neuron','primary_type']],left_on='post',right_on='neuron',how='left')
#%%
connections_types['same_type']=(connections_types['primary_type_x']==connections_types['primary_type_y']).astype(int)
#%%
connections_types=connections_types[['pre', 'post', 'AA', 'AD',  'DA', 'DD', 
       'sum_syn', 'reciprocal', 'primary_type_x', 'primary_type_y', 'same_type']]
#%%
connections_types.to_feather(CONNECTIONS_TABLE_FTR)
#%%
connections5=connections.query('sum_syn>=5')