# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:00:36 2024

@author: user
"""



from nglui.statebuilder import ChainedStateBuilder


import matplotlib.pyplot as plt
import numpy as np
from fafbseg import flywire
import navis
import pandas as pd

import time
import datetime
import os
import sys
from pathlib import Path
# Make repo root importable so config.py can be found from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import METHODS_DIR, PRINCETON_SYNAPSE_CSV, SWC_DIR, PROCESSED_SWC_DIR, refuse_published_run
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
import random
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pickle
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
refuse_published_run("Step 01")
#allsynapses=pd.read_feather(r'C:\Users\user\organised_work\data\connections\synapses')
#allsynapses=pd.read_feather(r'C:\Users\user\organised_work\data\783\generated\preprocessing\synapses\flywire_synapses_783_sclass_filter_processed.feather')
#%% Shards: N copies of this step at once, each on its own folders
# N_SHARDS = 1 runs every folder in one process. With N_SHARDS = N, the copy with SHARD = k (0 to N-1) runs the folders
# sorted(all folders under SWC_DIR)[k::N] that are not done yet, and keeps from the CSV only the rows whose pre or
# post is a cell of those folders: the rows, synapse_id values and dtypes those cells get from a full read. The
# folders are dealt from the full sorted list, never from the folders still to do, so a copy that is restarted
# gets the same folders again and never takes one of another copy's. STEP01_N_SHARDS and STEP01_SHARD in the
# environment override the two settings, to launch the copies from a shell.
N_SHARDS = 1
SHARD = 0
N_SHARDS = int(os.environ.get('STEP01_N_SHARDS', N_SHARDS))
SHARD = int(os.environ.get('STEP01_SHARD', SHARD))
if not (N_SHARDS >= 1 and 0 <= SHARD < N_SHARDS):
    raise SystemExit(f'SHARD must be 0 to N_SHARDS - 1 (N_SHARDS = {N_SHARDS}, SHARD = {SHARD})')
if N_SHARDS > 1:
    shard_folders = set(sorted(f.name for f in os.scandir(SWC_DIR) if f.is_dir())[SHARD::N_SHARDS])
    # the cells of this copy's folders that are not done yet (done: linker.pkl is there, as in the loop below)
    shard_cells = np.array(sorted({np.int64(f.name[:-4]) for d in shard_folders
                                   if not os.path.exists(os.path.join(PROCESSED_SWC_DIR, d, 'linker.pkl'))
                                   for f in os.scandir(os.path.join(SWC_DIR, d)) if f.name.endswith('.swc')}),
                           dtype=np.int64)
    print(f'Shard {SHARD} of {N_SHARDS}: folders {sorted(shard_folders)}, {len(shard_cells):,} cells not done yet')
#%%
# Read only the columns this step uses (ctr_x/y/z and neuropil are never used here), in chunks.
# In each chunk: rebuild the full root IDs, drop self-synapses, and keep coordinates and size as int32
# (checked to fit). Each folder's synapses are cast back to int64 in the loop below, so the neurons get
# the same values and dtypes as with a full read. The index is the CSV row number, which becomes synapse_id.
# A blank or decimal value in these columns stops the read here, where a full read would have loaded it as float.
syn_cols=['pre_x', 'pre_y', 'pre_z', 'post_x', 'post_y', 'post_z', 'size',
          'pre_root_id_720575940', 'post_root_id_720575940']
small_cols=['pre_x', 'pre_y', 'pre_z', 'post_x', 'post_y', 'post_z', 'size']
chunks=[]
for chunk in pd.read_csv(PRINCETON_SYNAPSE_CSV, usecols=syn_cols, dtype={c: np.int64 for c in syn_cols},
                         chunksize=5_000_000):
    # Full root ID = 720575940 * 10**(number of digits of the truncated id) + truncated id
    for c in ['pre_root_id_720575940', 'post_root_id_720575940']:
        v=chunk[c].to_numpy()
        if (v<0).any():
            raise ValueError(f'{c} has negative values')
        n_digits=np.ones(len(v), dtype=np.int64)
        for k in range(1, 19):
            n_digits+=v>=10**k
        chunk[c]=(720575940*10**n_digits+v).astype(np.int64)
    chunk=chunk.rename(columns={'pre_root_id_720575940':'pre', 'post_root_id_720575940':'post'})
    chunk=chunk[chunk['pre']!=chunk['post']]
    if N_SHARDS > 1:   # only the rows of this copy's cells
        chunk=chunk[chunk['pre'].isin(shard_cells)|chunk['post'].isin(shard_cells)]
    for c in small_cols:
        if chunk[c].min()<np.iinfo(np.int32).min or chunk[c].max()>np.iinfo(np.int32).max:
            raise ValueError(f'{c} does not fit in int32')
    chunk=chunk.astype({c: np.int32 for c in small_cols})
    chunks.append(chunk)
allsynapses=pd.concat(chunks)
del chunks, chunk, v, n_digits
#%%
allsynapses.columns

#%%
#duplicates_df = allsynapses[allsynapses.duplicated(keep=False)]
from flow_centrality import prepare_for_split   # the flow column without navis's dense matrix
#%%
def get_split(item,flow_thresh=1):
    try:    
        split=navis.split_axon_dendrite(prepare_for_split(item), metric='synapse_flow_centrality',reroot_soma=True,flow_thresh=flow_thresh)
        return split
    except:
        return 'split issue'
def heal_attach_princeton(item,allsynapses):
    try:
        healed_neurons=navis.heal_skeleton(item)
        healed_neurons_att_syn=attach_synapses_princeton(healed_neurons,allsynapses)
        return healed_neurons_att_syn
    except:
        return 'heal or attach issue'

#%%
#allsynapses=allsynapses.drop_duplicates()???

#%%
import os
import time
import pickle
import navis  # Assuming you have navis library available for NeuronList


# Define the base directories
input_base_dir = SWC_DIR
output_base_dir = PROCESSED_SWC_DIR
output_base_dir.mkdir(parents=True, exist_ok=True)


# Get the list of subdirectories in the input directory
input_subfolders = {f.name for f in os.scandir(input_base_dir) if f.is_dir()}
# A folder counts as done only when its last pickle (linker.pkl) is there; a folder left empty or
# half-written by an interrupted run is processed again
output_subfolders = {f.name for f in os.scandir(output_base_dir)
                     if f.is_dir() and os.path.exists(os.path.join(f.path, 'linker.pkl'))}

# Identify subfolders that are in the input but not in the output
folders_to_process = input_subfolders - output_subfolders
if N_SHARDS > 1:   # this copy's folders, dealt from the full sorted list (see the Shards cell)
    folders_to_process = folders_to_process & shard_folders


# Loop through each subfolder that needs to be processed

#%%
# Get the list of subdirectories in the input directory
start_time_global = time.time()
start_time_global = time.time()


# Loop through each subfolder
for folder_name in sorted(folders_to_process):

    issues_in_comp = []
    linker_neurons=[]
    all_SI=[]
    all_non_succsessfull_connectors=pd.DataFrame(columns=['connector_id', 'x', 'y', 'z', 'partner_id', 'type',
           'node_id', 'compartment', 'neuron','synapse_id','size'])
    all_connectors=pd.DataFrame(columns=['connector_id', 'x', 'y', 'z', 'partner_id', 'type',
           'node_id', 'compartment', 'neuron','synapse_id','size'])
    connector_frames=[]   # each cell's connectors, joined to all_connectors once after the loop (a join per cell is quadratic)
    input_folder = os.path.join(input_base_dir, folder_name)
    output_folder = os.path.join(output_base_dir, folder_name)
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load SWC files for the current folder
    swc_list = load_swc(input_folder)  
    
    swc_list_id=[np.int64(i.id) for i in swc_list]
    if N_SHARDS > 1 and not np.isin(swc_list_id, shard_cells).all():
        raise SystemExit(f'Folder {folder_name} has cells whose synapses this copy did not read; run it again')

    allsynapses2=allsynapses[(allsynapses['pre'].isin(swc_list_id))|(allsynapses['post'].isin(swc_list_id))]
    # Back to the layout and dtypes of the full read: synapse_id (the CSV row number) first, every column int64
    allsynapses2=allsynapses2.astype(np.int64)
    allsynapses2.insert(0, 'synapse_id', allsynapses2.index.to_numpy(dtype=np.int64))
    
    swc_list_to_analyse = navis.NeuronList(swc_list[0:])  # Adjust slicing if needed
    healed_attached_neurons_list = heal_attach_princeton(swc_list_to_analyse,allsynapses2)  # Ensure this function is defined

    # Determine the number of items to process
    num_items = len(healed_attached_neurons_list)

    # Start the time measurement
    start_time = time.time()


    # Perform your operations for each neuron
    for i in range(0, num_items):
        

        print(f"--------------------------------{i}--------------------")
        n = healed_attached_neurons_list[i]
        
        if len(n.nodes)>80000:
            issues_in_comp.append([n.id,len(n.nodes)])
            continue
        axon_swc=None
        dend_swc=None
        linker_swc=None
        axon=None
        dend=None
        linker=False
        split=get_split(n)
        if len(split) ==3:
            if split[0].compartment=='dendrite' and split[2].compartment=='axon':
                axon=parents_check_on_comp(split[2])
                dend=parents_check_on_comp(split[0])
                axon_swc=split[2]
                linker_swc=split[1]
                dend_swc=split[0]
                linker=True
                linker_neurons.append(n.id)
            else:
                if split[2].compartment=='dendrite' and split[0].compartment=='axon':
                    axon=parents_check_on_comp(split[0])
                    dend=parents_check_on_comp(split[2])
                    axon_swc=split[0]
                    linker_swc=split[1]
                    dend_swc=split[2]
                    linker=True
                    linker_neurons.append(n.id) 
                else:
                    issues_in_comp.append([n.id,'issue1'])
                    all_non_succsessfull_connectors=pd.concat([all_non_succsessfull_connectors,split.connectors])

                    continue
        elif len(split) == 2:
            if split[0].compartment=='dendrite' and split[1].compartment=='axon':
        
                axon = parents_check_on_comp(split[1])
                dend = parents_check_on_comp(split[0])
                axon_swc = split[1]
                dend_swc = split[0]
            else:
                if split[1].compartment=='dendrite' and split[0].compartment=='axon':
            
                    axon = parents_check_on_comp(split[0])
                    dend = parents_check_on_comp(split[1])
                    axon_swc = split[0]
                    dend_swc = split[1]
                else:
                    issues_in_comp.append([n.id,'issue2'])
                    all_non_succsessfull_connectors=pd.concat([all_non_succsessfull_connectors,split.connectors])

                    continue
        else:

            issues_in_comp.append([n.id,'issue3'])


            continue
        
        nid=n.id

        SI,IG=SI_calc(['',(len(axon_swc.presynapses),len(axon_swc.postsynapses)),(len(dend_swc.presynapses),len(dend_swc.postsynapses))])
        #print(SI)
        all_SI.append([nid,SI])
        axon_connectors=axon_swc.connectors
        axon_connectors['compartment']='A'
        dend_connectors=dend_swc.connectors
        dend_connectors['compartment']='D'
        
        if linker:
                
            linker_connectors=linker_swc.connectors
            linker_connectors['compartment']='L'
            neuron_connectors=pd.concat([axon_connectors,dend_connectors,linker_connectors])
        else:
            neuron_connectors=pd.concat([axon_connectors,dend_connectors])
        neuron_connectors['neuron']=nid
        neuron_connectors=neuron_connectors.reset_index(drop=True)
        connector_frames.append(neuron_connectors)
        
    if connector_frames:
        all_connectors=pd.concat([all_connectors]+connector_frames)
    del connector_frames

    # End the time measurement
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    
    end_time_c_g = time.time()
    elapsed_time = end_time_c_g - start_time_global
    print(f"Elapsed time for folder {folder_name}: {elapsed_time:.2f} seconds")
    print(f"Elapsed time so far: {elapsed_time:.2f} seconds")


    # Each pickle is written under a temporary name holding the process id, then renamed, so no file is ever left
    # half-written under its real name. Two copies that happen to do the same folder write the same bytes, and the
    # last rename leaves a whole file. linker.pkl, which marks the folder as done, comes last, so a run stopped while
    # writing leaves the folder to be redone. The later steps read these files by their exact names, so a leftover
    # temporary file is never read.
    for name, contents in (('connectors.pkl', all_connectors), ('all_SI.pkl', all_SI),
                           ('issues.pkl', issues_in_comp), ('linker.pkl', linker_neurons)):
        tmp = os.path.join(output_folder, f'{name}.{os.getpid()}.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(contents, f)
        os.replace(tmp, os.path.join(output_folder, name))

#%%
end_time_c_g = time.time()
elapsed_time = end_time_c_g - start_time_global
print(f"Elapsed time global: {elapsed_time:.2f} seconds")

print("Processing complete for all folders.")
     #%%4
