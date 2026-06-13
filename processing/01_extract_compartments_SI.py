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
from config import METHODS_DIR, PRINCETON_SYNAPSE_CSV, SWC_DIR, PROCESSED_SWC_DIR
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
import random
client = CAVEclient('flywire_fafb_production')
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pickle
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
#%%
#allsynapses=pd.read_feather(r'C:\Users\user\organised_work\data\connections\synapses')
#allsynapses=pd.read_feather(r'C:\Users\user\organised_work\data\783\generated\preprocessing\synapses\flywire_synapses_783_sclass_filter_processed.feather')
#%%
allsynapses=pd.read_csv(PRINCETON_SYNAPSE_CSV)
#%%
allsynapses.columns
#%%

#%%
allsynapses=allsynapses.reset_index(drop=False)
#%%
allsynapses=allsynapses.drop(columns=['ctr_x', 'ctr_y', 'ctr_z'])
#%%
allsynapses["pre_root_id_720575940"] = (
    720575940 * 10**allsynapses["pre_root_id_720575940"].astype(str).str.len()
    + allsynapses["pre_root_id_720575940"].astype(int)
).astype(np.int64)

#%%
allsynapses["post_root_id_720575940"] = (
    720575940 * 10**allsynapses["post_root_id_720575940"].astype(str).str.len()
    + allsynapses["post_root_id_720575940"].astype(int)
).astype(np.int64)
#%%


#%%
allsynapses.columns

#%%
#duplicates_df = allsynapses[allsynapses.duplicated(keep=False)]

#%%
allsynapses.columns=['synapse_id','pre_x', 'pre_y', 'pre_z', 'post_x',
       'post_y', 'post_z', 'size', 'pre',
       'post', 'neuropil']

#%%
def get_split(item,flow_thresh=1):
    try:    
        split=navis.split_axon_dendrite(item, metric='synapse_flow_centrality',reroot_soma=True,flow_thresh=flow_thresh)
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

allsynapses=allsynapses[allsynapses['pre']!=allsynapses['post']]
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


# Get the list of subdirectories in the input directory
input_subfolders = {f.name for f in os.scandir(input_base_dir) if f.is_dir()}
output_subfolders = {f.name for f in os.scandir(output_base_dir) if f.is_dir()}

# Identify subfolders that are in the input but not in the output
folders_to_process = input_subfolders - output_subfolders


# Loop through each subfolder that needs to be processed

#%%
# Get the list of subdirectories in the input directory
start_time_global = time.time()
start_time_global = time.time()


# Loop through each subfolder
for folder_name in folders_to_process:

    issues_in_comp = []
    linker_neurons=[]
    all_SI=[]
    all_non_succsessfull_connectors=pd.DataFrame(columns=['connector_id', 'x', 'y', 'z', 'partner_id', 'type',
           'node_id', 'compartment', 'neuron','synapse_id','size'])
    all_connectors=pd.DataFrame(columns=['connector_id', 'x', 'y', 'z', 'partner_id', 'type',
           'node_id', 'compartment', 'neuron','synapse_id','size'])
    input_folder = os.path.join(input_base_dir, folder_name)
    output_folder = os.path.join(output_base_dir, folder_name)
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load SWC files for the current folder
    swc_list = load_swc(input_folder)  
    
    swc_list_id=[np.int64(i.id) for i in swc_list]

    allsynapses2=allsynapses[(allsynapses['pre'].isin(swc_list_id))|(allsynapses['post'].isin(swc_list_id))]
    
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
        all_connectors=pd.concat([all_connectors,neuron_connectors])
        
    # End the time measurement
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    
    end_time_c_g = time.time()
    elapsed_time = end_time_c_g - start_time_global
    print(f"Elapsed time for folder {folder_name}: {elapsed_time:.2f} seconds")
    print(f"Elapsed time so far: {elapsed_time:.2f} seconds")


    with open(os.path.join(output_folder, 'connectors.pkl'), 'wb') as f:
        pickle.dump(all_connectors, f)
    with open(os.path.join(output_folder, 'all_SI.pkl'), 'wb') as f:
        pickle.dump(all_SI, f)
    with open(os.path.join(output_folder, 'issues.pkl'), 'wb') as f:
        pickle.dump(issues_in_comp, f)
    with open(os.path.join(output_folder, 'linker.pkl'), 'wb') as f:
        pickle.dump(linker_neurons, f)

#%%
end_time_c_g = time.time()
elapsed_time = end_time_c_g - start_time_global
print(f"Elapsed time global: {elapsed_time:.2f} seconds")

print("Processing complete for all folders.")
     #%%4
