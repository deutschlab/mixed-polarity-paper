import sys
from pathlib import Path

# -- path setup ---------------------------------------------------------------
# Add repository root to sys.path so config.py and methods_all can be imported
# regardless of the working directory when the script is run.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import (
    METHODS_DIR,
    LARVA_DATA_DIR,
    LARVA_SWC_DIR,
    LARVA_SYNAPSE_TABLE_CSV,
    LARVA_NEURON_LIST_CSV,
    LARVA_CONNECTORS_PKL,
    LARVA_PRE_CONNECTORS_FTR,
    LARVA_POST_CONNECTORS_FTR,
    LARVA_SYNAPSES_PRE_MERGE_FTR,
    LARVA_SYNAPSES_FTR,
    LARVA_SI_FTR,
)

sys.path.insert(0, str(METHODS_DIR))
from methods_all import *

import os
import seaborn as sns
import pickle
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

from scipy.stats import gaussian_kde

# -- create output directories if needed -------------------------------------
LARVA_DATA_DIR.joinpath("output").mkdir(parents=True, exist_ok=True)
LARVA_DATA_DIR.joinpath("results").mkdir(parents=True, exist_ok=True)

#%%
# Pre-process SWC files: add metadata header line.
# Input:  LARVA_SWC_DIR / "neurons"      (raw SWC files from Winding et al. 2023)
# Output: LARVA_SWC_DIR / "neurons_fixed" (SWC files with metadata header added)
#
# Only run this block if starting from unfixed SWC files obtained directly from
# Winding et al. 2023. Skip if the fixed SWC files already exist in LARVA_SWC_DIR.
'''
import os
import pandas as pd

swc_directory = str(LARVA_SWC_DIR / "neurons")
output_directory = str(LARVA_SWC_DIR / "neurons_fixed")

os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(swc_directory):
    if filename.endswith('.swc') and not filename.endswith('_fixed.swc'):
        swc_path = os.path.join(swc_directory, filename)
        df = pd.read_csv(swc_path)
        df.columns = ['id', 'type', 'x', 'y', 'z', 'r', 'parent_id']
        swc_formatted = df[['id', 'type', 'x', 'y', 'z', 'r', 'parent_id']]
        file_id = filename.rsplit('.', 1)[0]
        metadata_line = f"# Meta: {{'id': '{file_id}'}}\n"
        output_path = os.path.join(output_directory, f'{file_id}_fixed.swc')
        with open(output_path, 'w') as f:
            f.write(metadata_line)
            swc_formatted.to_csv(f, sep=' ', index=False, header=False, mode='a')
        print(f"Processed: {filename} -> {output_path}")
'''

#%%
# Define the directory containing the SWC files
directory_path = str(LARVA_SWC_DIR)

# List all SWC files in the directory
swc_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.swc')]

# Load the SWC files using navis
swc_list = []
for swc_file in swc_files:
    try:
        # Load each SWC file as a TreeNeuron object
        neuron = navis.read_swc(swc_file)
        swc_list.append(neuron)
    except Exception as e:
        print(f"Failed to load {swc_file}: {e}")
#%%


#%%

def get_split(item,flow_thresh=1):
    try:    
        split=navis.split_axon_dendrite(item, metric='synapse_flow_centrality',reroot_soma=True,flow_thresh=flow_thresh)
        return split
    except:
        return 'split issue'
def heal_attach(item,allsynapses):
    try:
        #ealed_neurons=navis.heal_skeleton(item)
        healed_neurons_att_syn=attach_synapses_larva2(item,allsynapses)
        return healed_neurons_att_syn
    except:
        return 'heal or attach issue'

#%%
allsynapses=pd.read_csv(LARVA_SYNAPSE_TABLE_CSV)
#%%


#%%
allsynapses.columns=['rowid', 'pre_skeleton_node_id', 'connector_id',
       'pre_x', 'pre_y', 'pre_z',
       'connector_x', 'connector_y', 'connector_z', 'post_skeleton_node_id',
       'post_x', 'post_y', 'post_z',
       'pre', 'post']
#%%
allsynapses['cleft_score']=50
#%%
pre=allsynapses['pre'].unique()
post=allsynapses['post'].unique()

#%%
aa=allsynapses.head()
#%%
nlist=pd.read_csv(LARVA_NEURON_LIST_CSV)

#%%
nlist.dtypes

#%%
#allsynapses=allsynapses[allsynapses['pre'].isin(nlist['neuron'])]
#allsynapses=allsynapses[allsynapses['post'].isin(nlist['neuron'])]

#%%
#allsynapses=allsynapses[(allsynapses['pre']!=-1)&(allsynapses['post']!=-1)]

#%% 

#%%
swc_list_to_analyse = navis.NeuronList(i for i in swc_list if int(i.id) in nlist['neuron'].values)
#%%
#swc_list_to_analyse = navis.NeuronList(i for i in swc_list)


#%%
healed_attached_neurons_list = heal_attach(swc_list_to_analyse,allsynapses) 
#%%


#%%
issues=[]
issues_s=[]
# Determine the number of items to process
num_items = len(healed_attached_neurons_list)
all_connectors=pd.DataFrame(columns=['connector_id', 'x', 'y', 'z', 'cleft_score', 'partner_id', 'type',
       'node_id', 'compartment', 'neuron'])
# Start the time measurement
start_time = time.time()
SI_list=[]
aord = []
fragments_data = []
# Perform your operations for each neuron
ddd=0
for i in range(0, num_items):
    ddd+=1
    print(ddd)
    

    print(f"--------------------------------{i}--------------------")
    n = healed_attached_neurons_list[i]
    
    #if len(n.nodes)>80000:
    #    trees.append(f'neruon too big: {n.id}')
    #    continue
    axon_swc=None
    dend_swc=None
    linker_swc=None
    axon=None
    dend=None
    linker=False
    split=get_split(n,flow_thresh=1)
    if len(split) ==3:
        if split[0].compartment == 'dendrite' and split[2].compartment == 'axon':
            
            
            axon_swc=split[2]
            linker_swc=split[1]
            dend_swc=split[0]
            linker=True
        else:
            issues.append([i])
            print("---------------issue---------------")
            continue
    elif len(split) == 2:
        if split[0].compartment=='linker' or split[1].compartment=='linker':
            issues_s.append([i])
            print("---------------issue_solved---------------")
            continue
        if split[0].compartment=='axon':
            issues.append([i])
            print("---------------issue_opposite---------------")
            axon_swc = split[0]
            dend_swc = split[1]
        else:
            axon_swc = split[1]
            dend_swc = split[0]

    else:
        print('split issue compartments number')
        print(len(split))
        continue
    

    SI,IG=SI_calc(['',(len(axon_swc.presynapses),len(axon_swc.postsynapses)),(len(dend_swc.presynapses),len(dend_swc.postsynapses))])
    SI_list.append([n.id,SI,linker])

    axon_connectors=axon_swc.connectors
    axon_connectors['compartment']='A'
    dend_connectors=dend_swc.connectors
    dend_connectors['compartment']='D'
    
    nid=n.id
    if linker:
            
        linker_connectors=linker_swc.connectors
        linker_connectors['compartment']='L'
        neuron_connectors=pd.concat([axon_connectors,dend_connectors,linker_connectors])
    else:
        neuron_connectors=pd.concat([axon_connectors,dend_connectors])
    neuron_connectors['neuron']=nid
    neuron_connectors=neuron_connectors.reset_index(drop=True)
    all_connectors=pd.concat([all_connectors,neuron_connectors])
    
# End the time measurement
end_time = time.time()
elapsed_time = end_time - start_time
#%%


with open(LARVA_CONNECTORS_PKL, 'wb') as f:
    pickle.dump(all_connectors, f)
#%%

#%%

SI_df=pd.DataFrame(SI_list)
SI_df.to_feather(str(LARVA_SI_FTR))

#%%
# Define the base directories

#%%

#%%

with open(LARVA_CONNECTORS_PKL, 'rb') as f:
    all_connectors_list = pickle.load(f)
   

#%%    
    
#%%
all_connectors=all_connectors_list.copy()
        #%%
del all_connectors_list        
    #%%
   
pre_connectors=all_connectors[all_connectors['type']=='pre'] 
#%% 
pre_connectors.columns=['pre_connector_id','pre_x','pre_y','pre_z','cleft_score','post','type','pre_node_id','pre_compartment','pre','rowid']
pre_connectors=pre_connectors.drop(columns=['type'])
#%%
pre_connectors.reset_index(drop=True).to_feather(str(LARVA_PRE_CONNECTORS_FTR))
#%%
del pre_connectors
#%%


post_connectors=all_connectors[all_connectors['type']=='post']    
post_connectors.columns=['post_connector_id','post_x','post_y','post_z','cleft_score','pre','type','post_node_id','post_compartment','post','rowid']
post_connectors=post_connectors.drop(columns=['type'])
#%%
post_connectors.reset_index(drop=True).to_feather(str(LARVA_POST_CONNECTORS_FTR))
#%%



#%%
pre_connectors=pd.read_feather(str(LARVA_PRE_CONNECTORS_FTR))
#%%
'''
allsynapses2=allsynapses.copy()
allsynapses=allsynapses.drop_duplicates(subset=['pre', 'post', 'cleft_score', 'pre_x', 'pre_y', 'pre_z','post_x', 'post_y', 'post_z'])
allsynapses=allsynapses.drop_duplicates(subset=['pre', 'post', 'cleft_score', 'pre_x', 'pre_y', 'pre_z'])
allsynapses=allsynapses.drop_duplicates(subset=['pre', 'post', 'cleft_score', 'post_x', 'post_y', 'post_z'])

all_duplicates = allsynapses2[allsynapses2.duplicated(subset=['pre', 'post', 'cleft_score', 'pre_x', 'pre_y', 'pre_z', 'post_x', 'post_y', 'post_z'], keep=False)]
'''
#%%
pre_connectors['pre'] = pre_connectors['pre'].astype('int64')
pre_connectors['post'] = pre_connectors['post'].astype('int64')
#%%
pre_connectors=pre_connectors[['pre_compartment','rowid']]
# Step 2: Merge based on the specified columns
# Assuming you want to merge pre_connectors with itself
merged_connectors = allsynapses.merge(
    pre_connectors,
    on='rowid',
    how='left'  # Adjust 'how' as necessary (e.g., 'inner', 'left', etc.)
)


#%%
merged_connectors.reset_index(drop=True).to_feather(str(LARVA_SYNAPSES_PRE_MERGE_FTR))


#%%
#%%
merged_connectors=pd.read_feather(str(LARVA_SYNAPSES_PRE_MERGE_FTR))
#%%
post_connectors=pd.read_feather(str(LARVA_POST_CONNECTORS_FTR))
#%%
post_connectors['pre'] = post_connectors['pre'].astype('int64')
post_connectors['post'] = post_connectors['post'].astype('int64')


#%%
post_connectors=post_connectors[['post_compartment','rowid']]

# Step 2: Merge based on the specified columns
# Assuming you want to merge pre_connectors with itself
merged_connectors_f = merged_connectors.merge(
    post_connectors,
    on=['rowid'],
    how='left'  # Adjust 'how' as necessary (e.g., 'inner', 'left', etc.)
)

#%%
merged_connectors_f=merged_connectors_f.dropna()


#%%
del merged_connectors
#%%
merged_connectors_f.reset_index(drop=True).to_feather(str(LARVA_SYNAPSES_FTR))
#%%

merged_connectors_f=merged_connectors_f[(merged_connectors_f['pre']!=-1)&(merged_connectors_f['post']!=-1)]
#%%
merged_connectors_f['comp']=merged_connectors_f['pre_compartment']+merged_connectors_f['post_compartment']
#%%
SI_df=pd.read_feather(str(LARVA_SI_FTR))


#%%
SI_df.columns=['neuron','SI','linker']
SI_df['neuron']=SI_df['neuron'].astype(np.int64)
merged_connectors_f_p=merged_connectors_f.merge(SI_df,left_on='pre',right_on='neuron',how='left')
merged_connectors_f_p=merged_connectors_f_p.merge(SI_df,left_on='post',right_on='neuron',how='left')

#%%
sns.kdeplot(SI_df[['SI']])
plt.xlim(0,1)
#%%
results=merged_connectors_f.groupby(by='comp').size().reset_index()
#%%
results.columns=['syn_type','size']
#%%
results=results[results['syn_type'].isin(['AD','AA','DD','DA'])]
#%%
sns.barplot(results,x='syn_type',y='size')
#%%
# Calculate proportions
results['proportion'] = results['size'] / results['size'].sum()

# Create the bar plot with proportions
sns.barplot(data=results, x='syn_type', y='proportion')
#%%


merged_connectors_f2=merged_connectors_f_p[(merged_connectors_f_p['linker_x']==1)&(merged_connectors_f_p['linker_y']==1)]
#%%
aaaa=pd.concat([merged_connectors_f2['pre'],merged_connectors_f2['post']]).unique()

#%%
results=merged_connectors_f2.groupby(by='comp').size().reset_index()
#%%
results.columns=['syn_type','size']
#%%
results=results[results['syn_type'].isin(['AD','AA','DD','DA'])]
#%%
sns.barplot(results,x='syn_type',y='size')
#%%
SI_df.groupby('linker').size()

#%%
# Calculate proportions
results['proportion'] = results['size'] / results['size'].sum()

# Create the bar plot with proportions
sns.barplot(data=results, x='syn_type', y='proportion')

#%%

merged_connectors_f=merged_connectors_f_p[(merged_connectors_f_p['SI_x']>=0.1)&(merged_connectors_f_p['SI_y']>=0.1)]
#%%
results=merged_connectors_f.groupby(by='comp').size().reset_index()
#%%
results.columns=['syn_type','size']
#%%
results=results[results['syn_type'].isin(['AD','AA','DD','DA'])]
#%%
sns.barplot(results,x='syn_type',y='size')
#%%
# Calculate proportions
results['proportion'] = results['size'] / results['size'].sum()

# Create the bar plot with proportions
sns.barplot(data=results, x='syn_type', y='proportion')

#%%
x = navis.example_neurons(1)
#%%
split = navis.split_axon_dendrite(x, metric='synapse_flow_centrality',
                                   reroot_soma=True)

#%%
split
