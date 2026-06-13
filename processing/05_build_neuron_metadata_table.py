
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    METHODS_DIR, NEURON_ANNOTATIONS_CSV, CELL_STATS_CSV,
    PROCESSED_SWC_DIR, PROCESSED_BIG_NEURONS_DIR,
    SI_UPDATED_FTR, SWC_DATA_FTR, SYNAPSE_TABLE_FTR,
    NEURON_TABLE_FTR, RANKS_DIR, NEURONS_CSV,
    DERIVED_DATA_DIR,
)
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import os
import seaborn as sns
import pickle
#%%
#%%
nodesG=pd.read_csv(NEURON_ANNOTATIONS_CSV)
#%%
nodesG=nodesG[['root_id','super_class','top_nt','cell_type']]
cell_stats=pd.read_csv(CELL_STATS_CSV)
nodesG=nodesG.merge(cell_stats,on='root_id',how='left')
#%%
nodesG=nodesG.rename(columns={'top_nt':'nt_type'})
#%%SI and linker
import os
import pickle
import pandas as pd

# Path containing all subfolders
base_dir = PROCESSED_SWC_DIR

# Empty lists to collect data
all_SI_list = []
linker_list = []

# Loop over each subfolder
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)

    # Check if it's a directory
    if os.path.isdir(folder_path):
        try:
            # Load all_SI
            with open(os.path.join(folder_path, 'all_SI.pkl'), 'rb') as f:
                all_SI = pickle.load(f)
            all_SI_list.append(all_SI)

            # Load linker
            with open(os.path.join(folder_path, 'linker.pkl'), 'rb') as f:
                linker = pickle.load(f)
            linker_list.append(linker)

        except FileNotFoundError:
            print(f"Missing files in folder: {folder_name}")
        except Exception as e:
            print(f"Error loading data in folder {folder_name}: {e}")
# Flatten the nested list structure first
flat_all_SI_list = [item for sublist in all_SI_list for item in sublist]
flat_linker_list = [item for sublist in linker_list for item in sublist]
# Then convert to DataFrame
all_SI_df = pd.DataFrame(flat_all_SI_list)
linker_df = pd.DataFrame(flat_linker_list)

all_SI_df.columns=['root_id','SI']
linker_df.columns=['root_id']
all_SI_df['root_id']=all_SI_df['root_id'].astype(np.int64)
linker_df['root_id']=all_SI_df['root_id'].astype(np.int64)
linker_df['linker']=1
#%%
dfs = []
base_dir = PROCESSED_BIG_NEURONS_DIR

for fname in os.listdir(base_dir):
    if fname.startswith("linker_"):
        path = os.path.join(base_dir, fname)

        obj = pd.read_pickle(path)  # this is a list
        sid = fname.split("_", 1)[1].split(".", 1)[0]

        for v in obj:
            dfs.append({"source_id": sid, "value": v})

linker_df2 = pd.DataFrame(dfs)
linker_df2.columns=['root_id','a']
linker_df2=linker_df2['root_id']
linker_df2['linker']=1

#%%

linker_df=pd.concat([linker_df,linker_df2])
#%%bign addition
#%%big n afddition *******************
dfs = []
base_dir = PROCESSED_BIG_NEURONS_DIR

for fname in os.listdir(base_dir):
    if fname.startswith("all_SI_"):
        path = os.path.join(base_dir, fname)

        obj = pd.read_pickle(path)  # this is a list
        sid = fname.split("_", 1)[1].split(".", 1)[0]

        for v in obj:
            dfs.append({"source_id": sid, "value": v})

all_SI_df_big = pd.DataFrame(dfs)
#%%
out = pd.DataFrame({
    "root_id": all_SI_df_big["value"].apply(lambda x: int(x[0])),
    "SI":  all_SI_df_big["value"].apply(lambda x: float(x[1]))
})
#%%


all_SI_df=pd.concat([all_SI_df,out])

#%%
DERIVED_DATA_DIR.mkdir(parents=True, exist_ok=True)
all_SI_df.to_feather(SI_UPDATED_FTR)


#%%
nodesG=nodesG.merge(all_SI_df,on='root_id',how='left')

nodesG=nodesG.merge(linker_df,on='root_id',how='left')
nodesG['linker']=nodesG['linker'].fillna(0)
nodesG=nodesG.dropna(subset='SI')
sns.histplot(nodesG,hue='linker',x='SI')
#%%

#%%nt
#nt_df=pd.read_csv(NEURONS_CSV)
#nodesG=nodesG.merge(nt_df[['root_id','nt_type','nt_type_score']],on='root_id',how='left')
#%%
r'''
# Define the base input directory
input_base_dir = r"C:\Users\user\organised_work\data\783\recieved\swc\783"

# Get the list of subdirectories in the input directory
input_subfolders = [f.name for f in os.scandir(input_base_dir) if f.is_dir()]

# Define your global variables (if needed)

# Initialize a global DataFrame (optional, if you need to aggregate results)
global_results = []
vv=0
# Start timing
start_time_global = time.time()

# Loop through each folder in the input directory
for folder_name in input_subfolders:
    print(f"Processing folder: {folder_name}")
    
    # Define the input folder path
    input_folder = os.path.join(input_base_dir, folder_name)
    
    try:
        # Load SWC files for the current folder
        swc_list = load_swc(input_folder)  # Ensure this function is defined
        
        # Perform healing and attaching neurons

        # Optionally, process results here (e.g., store in a DataFrame)
        # Append results to the global_results list if needed
        for swc in swc_list:
            id_=swc.id
            nodes=swc.n_nodes
            leafs=swc.n_leafs
            n_branches=swc.n_branches
            cable_length=swc.cable_length
            global_results.append([id_,nodes,leafs,n_branches,cable_length])

    except Exception as e:
        print(f"Error processing folder {folder_name}: {e}")
        continue  # Skip to the next folder if an error occurs
    del swc_list
    gc.collect()
    print(vv)
    vv+=1

    
df=pd.DataFrame(global_results)
df.columns=['root_id','nodes','leafs','branches','cable_length']
df.to_feather(r'C:\Users\user\organised_work\data\783\generated\post_processing_data\article\swc_data.ftr')



'''

#%%
df=pd.read_feather(SWC_DATA_FTR)
df['root_id']=df['root_id'].astype(np.int64)
nodesG=nodesG.merge(df,on='root_id',how='left')
#%%
#%%
allsynapses=pd.read_feather(SYNAPSE_TABLE_FTR)
#%%


#%%


#%%
out_synapses=pd.DataFrame(allsynapses.groupby('pre').size()).rename(columns={0:'out_synapses_count'})
in_synapses=pd.DataFrame(allsynapses.groupby('post').size()).rename(columns={0:'in_synapses_count'})

#%%
max_degrees=pd.DataFrame(allsynapses.groupby(by=['pre','post']).size()).reset_index().rename(columns={0:'degree'})

max_out_degree=max_degrees.groupby(by='pre')['degree'].max()
max_in_degree=max_degrees.groupby(by='post')['degree'].max()

mean_out_degree=max_degrees.groupby(by='pre')['degree'].mean()
mean_in_degree=max_degrees.groupby(by='post')['degree'].mean()

in_partners=max_degrees.groupby(by='post')['degree'].count()
out_partners=max_degrees.groupby(by='pre')['degree'].count()
#%%

#%%correct_pos
unique_neurons = pd.unique(allsynapses[['pre', 'post']].values.ravel())
result_df = pd.DataFrame({'neuron': unique_neurons})
#%%
# Define prefixes and components
prefixes = ['pre', 'post']
components = ['A', 'D', 'L']

# Initialize the result DataFrame with columns for each combination
for prefix in prefixes:
    for component in components:
        result_df[f"{prefix}_{component}"] = 0

# Extract the first character from the comp column
allsynapses['first_comp'] = allsynapses['comp'].str[0]

# For each component, count occurrences for pre and post neurons
# Process both characters in 'comp' for both pre and post neurons
for component in components:
    # Create masks for both pre and post based on the first and second characters of 'comp'
    pre_mask = allsynapses['comp'].str[0] == component
    post_mask = allsynapses['comp'].str[1] == component

    # Pre counts: Count occurrences where the neuron is in `pre` and the first character matches
    pre_counts = allsynapses.loc[pre_mask].groupby('pre').size()
    result_df[f"pre_{component}"] += result_df['neuron'].map(pre_counts).fillna(0).astype(int)

    # Post counts: Count occurrences where the neuron is in `post` and the second character matches
    post_counts = allsynapses.loc[post_mask].groupby('post').size()
    result_df[f"post_{component}"] += result_df['neuron'].map(post_counts).fillna(0).astype(int)

        #%%
result_df['axon_correct']=result_df['pre_A']/(result_df['pre_A']+result_df['post_A'])

result_df['dend_correct']=result_df['post_D']/(result_df['pre_D']+result_df['post_D'])

#%%

#%%
df=result_df.merge(nodesG,left_on='neuron',right_on='root_id',how='left')
#%%
# Ensure max_out_degree and max_in_degree are DataFrames with proper column names for merging
max_out_degree = max_out_degree.reset_index().rename(columns={'pre': 'neuron', 'degree': 'max_out_degree'})
max_in_degree = max_in_degree.reset_index().rename(columns={'post': 'neuron', 'degree': 'max_in_degree'})


mean_out_degree = mean_out_degree.reset_index().rename(columns={'pre': 'neuron', 'degree': 'mean_out_degree'})
mean_in_degree = mean_in_degree.reset_index().rename(columns={'post': 'neuron', 'degree': 'mean_in_degree'})

# Ensure in_partners and out_partners are DataFrames with proper column names for merging
out_partners = out_partners.reset_index().rename(columns={'pre': 'neuron', 'degree': 'out_partners'})
in_partners = in_partners.reset_index().rename(columns={'post': 'neuron', 'degree': 'in_partners'})
# Ensure out_synapses and in_synapses have neuron as the key
out_synapses = out_synapses.reset_index().rename(columns={'pre': 'neuron', 0: 'out_synapses_count'})
in_synapses = in_synapses.reset_index().rename(columns={'post': 'neuron', 0: 'in_synapses_count'})

# Merge all the data into the main DataFrame `df`
df = df.merge(out_synapses, on='neuron', how='left')
df = df.merge(in_synapses, on='neuron', how='left')
df = df.merge(max_out_degree, on='neuron', how='left')
df = df.merge(max_in_degree, on='neuron', how='left')
df = df.merge(mean_in_degree, on='neuron', how='left')
df = df.merge(mean_out_degree, on='neuron', how='left')

df = df.merge(out_partners, on='neuron', how='left')
df = df.merge(in_partners, on='neuron', how='left')
#%%

#df['comp_correct_ratio']=df['axon_correct']/df['dend_correct']
df['partners_ratio']=df['in_partners']/df['out_partners']
df['max_degree_ratio']=df['max_in_degree']/df['max_out_degree']
df['mean_degree_ratio']=df['mean_in_degree']/df['mean_out_degree']
df['mean_out_synapses_per_partner']=df['mean_out_degree']/df['out_partners']
df['mean_in_synapses_per_partner']=df['mean_in_degree']/df['in_partners']
df['mean_synapses_per_partner_ratio']=df['mean_in_synapses_per_partner']/df['mean_out_synapses_per_partner']
#%%
df=df[df['neuron']==df['root_id']]

#%%
NEURON_TABLE_FTR.parent.mkdir(parents=True, exist_ok=True)
df.to_feather(NEURON_TABLE_FTR)
#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
#%%
import os
import pandas as pd

# Define the folder path
folder_path = RANKS_DIR

# List all .ftr files
ftr_files = [f for f in os.listdir(folder_path) if f.endswith('.feather')]

# Initialize list to store renamed DataFrames
df_list = []

for file in ftr_files:
    # Extract part between last '-' before '-10000' and the next '-'
    base_name = os.path.splitext(file)[0]
    try:
        prefix = base_name.split('-10000')[0]
        extracted = prefix.split('-')[-1]  # Get the last segment before -10000
    except IndexError:
        extracted = 'unknown'

    # Read file and rename columns
    df = pd.read_feather(os.path.join(folder_path, file))
    df.columns = [f"{extracted}_{col}" for col in df.columns]
    df_list.append(df)

# Concatenate all DataFrames along columns
df_combined = pd.concat(df_list, axis=1)

# Preview the result

#%%
df_combined=df_combined[['allinputs_root_id','allinputs_layer_mean', 
       'allsensory_novisual_layer_mean',
      'ascending_layer_mean',
      'gustatory_layer_mean',
       'hygrosensory_layer_mean',
      
       'mechanosensory_layer_mean',
       'auditory_layer_mean',
    'jo_layer_mean',
       'peg_layer_mean', 'ocellar_layer_mean',
      'olfactory_layer_mean', 
   'thermosensory_layer_mean',
       'visual_projection_layer_mean']]
df_combined=df_combined.rename(columns={'allinputs_root_id':'neuron'})
#%%
#%%
df_combined['neuron']=df_combined['neuron'].astype(np.int64)
#%%
nodesG2=nodesG.merge(df_combined,on='neuron',how='left')
#%%
nodesG2.to_feather(NEURON_TABLE_FTR)
#%%
nodesG2=pd.read_feather(NEURON_TABLE_FTR)
nodesG2=nodesG2.rename(columns={'cell_type':'primary_type'})
nodesG2.to_feather(NEURON_TABLE_FTR)

#%%
df=pd.read_feather(NEURON_TABLE_FTR)
#%%
nt_df=pd.read_csv(NEURONS_CSV)
#%%
nt_df=nt_df[['root_id','nt_type']]
nt_df.columns=['neuron','nt_type']
df=df.merge(nt_df,how='left',on='neuron')
#%%
df=df.rename(columns={'nt_type_x':'nt_type_new','nt_type_y':'nt_type'})
#%%
aa=df.head(1000)

#%%
NEURON_TABLE_FTR.parent.mkdir(parents=True, exist_ok=True)
df.to_feather(NEURON_TABLE_FTR)



#%%

# Keep only rows where 'neuron' from dff are not in nodesG2
diff = dff[~dff['neuron'].isin(nodesG2['neuron'])]