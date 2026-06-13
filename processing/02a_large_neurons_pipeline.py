


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
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    METHODS_DIR, PRINCETON_SYNAPSE_CSV,
    PROCESSED_BIG_NEURONS_DIR, CONNECTORS_DIR,
    PRE_CONNECTORS_BIGN_FTR, POST_CONNECTORS_BIGN_FTR,
)
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
allsynapses.columns

#%%
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
#allsynapses=allsynapses.drop_duplicates()???

#%%

output_base_dir = PROCESSED_BIG_NEURONS_DIR
output_folder = os.path.join(output_base_dir, folder_name)
os.makedirs(output_folder, exist_ok=True)
#%%
lista=['720575940620156283', '720575940630006521', '720575940623444044', '720575940623499132', '720575940628731140', '720575940628762280', '720575940628908548', '720575940611379569', '720575940611563310', '720575940604088288', '720575940604569824', '720575940615160239', '720575940615255603', '720575940640062005', '720575940640263989', '720575940640736371', '720575940624377224', '720575940624394790', '720575940624402173', '720575940624449318', '720575940624547622', '720575940627796298', '720575940633228103', '720575940619055617', '720575940619148829', '720575940619245016', '720575940627190556', '720575940627247802', '720575940627250238', '720575940627253604', '720575940627264062', '720575940627349860', '720575940623000858', '720575940622309726', '720575940622313398', '720575940622331830', '720575940627996285', '720575940628038808', '720575940628069501', '720575940635684762', '720575940635945919', '720575940625525740', '720575940625550823', '720575940625733351', '720575940618366843', '720575940618444473', '720575940618554041', '720575940605758142', '720575940605899692', '720575940606149321', '720575940610584184', '720575940638720233', '720575940638900713', '720575940639209956', '720575940633294796', '720575940633305681', '720575940633573836', '720575940633626337', '720575940633631396', '720575940635089945', '720575940635119723', '720575940635202204', '720575940623888397', '720575940626783065', '720575940626403917', '720575940621226405', '720575940621371045', '720575940621393172', '720575940626979621', '720575940627053861', '720575940621522624', '720575940621551854', '720575940621638534', '720575940621640363', '720575940621684358', '720575940621689727', '720575940623113752', '720575940623133097', '720575940625357547', '720575940625926693', '720575940625934094', '720575940625952755', '720575940626000056', '720575940626039996', '720575940626044942', '720575940633787309', '720575940633893401', '720575940630810306', '720575940631306284', '720575940624973831', '720575940625049781', '720575940625102224', '720575940614710509', '720575940614956072', '720575940615028143', '720575940606648608', '720575940606692542', '720575940606756030', '720575940622516852', '720575940622523508', '720575940628307026', '720575940620827595', '720575940620975696', '720575940621040737', '720575940621103239', '720575940621116807', '720575940631180599', '720575940631190016', '720575940631942341', '720575940631945815', '720575940617257985', '720575940630407388', '720575940650416505', '720575940650929273', '720575940650935673', '720575940611719378', '720575940611772514', '720575940618165393', '720575940618203707', '720575940618229051', '720575940627497244', '720575940627594568', '720575940640803200', '720575940640834211', '720575940640851363', '720575940640978048', '720575940608545219', '720575940609282825', '720575940612264817', '720575940634545890', '720575940634612194', '720575940634707863', '720575940634748385', '720575940613635737', '720575940613686698', '720575940613783277', '720575940613891442', '720575940628406538', '720575940628592968', '720575940632777320', '720575940632795532', '720575940632821352', '720575940622160705', '720575940619811638', '720575940619991862', '720575940620008758', '720575940620023686', '720575940620065694', '720575940636479668', '720575940636543668', '720575940636574388', '720575940636576948', '720575940621801819', '720575940621875821', '720575940630460919', '720575940630610042', '720575940612976497', '720575940613065130', '720575940624225741', '720575940616169578', '720575940617691734', '720575940623636701', '720575940623788040', '720575940629382150', '720575940629513222', '720575940629513478', '720575940632403986', '720575940632504874', '720575940614137809', '720575940613455986', '720575940613583001', '720575940613588246', '720575940607635913', '720575940608235186', '720575940631584876', '720575940631608313', '720575940631758393', '720575940631842360', '720575940644632087', '720575940638128474', '720575940615607207', '720575940615743650', '720575940637611365', '720575940637633678', '720575940637656974', '720575940637689486']        #%%
lista=[np.int64(i) for i in lista]
#%%
allsynapses = allsynapses[
    (allsynapses['pre'].isin(lista)) | (allsynapses['post'].isin(lista))
]

#%%
'''
import time

start_time = time.time()
prev_time = start_time

log_list = [] 
for i, id_ in enumerate(lista):
    try:
        linker_neurons=[]
        issues_in_comp=[]
        all_SI=[]

        iter_start = time.time()
    
        swc1 = upload_swc(id_)
        allsynapses2 = allsynapses[
            (allsynapses['pre'] == id_) | (allsynapses['post'] == id_)
        ]
    
        healed_attached_neurons_list = heal_attach_princeton(
            swc1, allsynapses2
        )
        split = get_split(healed_attached_neurons_list)
    
        iter_end = time.time()
        iter_time = iter_end - prev_time
        prev_time = iter_end
    
        # collect info
        iter_info = {
            "iteration": i,
            "neuron_id": id_,
            "n_nodes": len(healed_attached_neurons_list.nodes),
            "n_synapses": len(split.connectors),
            "iteration_time_sec": iter_time
        }
        log_list.append(iter_info)
    
        # print nicely
        print("-" * 60)
        print(f"Iteration       : {i}")
        print(f"Neuron ID       : {id_}")
        print(f"# Nodes         : {iter_info['n_nodes']}")
        print(f"Synapses        : {iter_info['n_synapses']}")
        print(f"Iteration time  : {iter_time:.2f} sec")
        n = healed_attached_neurons_list
        
        axon_swc=None
        dend_swc=None
        linker_swc=None
        axon=None
        dend=None
        linker=False
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
        

    
        with open(os.path.join(output_folder, 'connectors.pkl'), 'wb') as f:
            pickle.dump(neuron_connectors, f)
        with open(os.path.join(output_folder, 'all_SI.pkl'), 'wb') as f:
            pickle.dump(all_SI, f)
        with open(os.path.join(output_folder, 'issues.pkl'), 'wb') as f:
            pickle.dump(issues_in_comp, f)
        with open(os.path.join(output_folder, 'linker.pkl'), 'wb') as f:
            pickle.dump(linker_neurons, f)
    except Exception as e:
        issues_in_comp.append([n.id,'issue3'])
        print(e)
        with open(os.path.join(output_folder, 'issues.pkl'), 'wb') as f:
            pickle.dump(issues_in_comp, f)
'''

#%%
output_folder = PROCESSED_BIG_NEURONS_DIR
os.makedirs(output_folder, exist_ok=True)
#%%
# --- loop ---
start_time = time.time()
prev_time = start_time

log_list = []
for i, id_ in enumerate(lista):
    try:
        linker_neurons = []
        issues_in_comp = []
        all_SI = []

        # (re-added) per-neuron non-success connectors DF


        iter_start = time.time()

        swc1 = upload_swc(id_)
        allsynapses2 = allsynapses[
            (allsynapses['pre'] == id_) | (allsynapses['post'] == id_)
        ]

        healed_attached_neurons_list = heal_attach_princeton(
            swc1, allsynapses2
        )
        split = get_split(healed_attached_neurons_list)

        iter_end = time.time()
        iter_time = iter_end - prev_time
        prev_time = iter_end

        iter_info = {
            "iteration": i,
            "neuron_id": id_,
            "n_nodes": len(healed_attached_neurons_list.nodes),
            "n_synapses": len(split.connectors),
            "iteration_time_sec": iter_time
        }
        log_list.append(iter_info)

        print("-" * 60)
        print(f"Iteration       : {i}")
        print(f"Neuron ID       : {id_}")
        print(f"# Nodes         : {iter_info['n_nodes']}")
        print(f"Synapses        : {iter_info['n_synapses']}")
        print(f"Iteration time  : {iter_time:.2f} sec")

        n = healed_attached_neurons_list

        axon_swc = None
        dend_swc = None
        linker_swc = None
        axon = None
        dend = None
        linker = False

        if len(split) == 3:
            if split[0].compartment == 'dendrite' and split[2].compartment == 'axon':
                axon = parents_check_on_comp(split[2])
                dend = parents_check_on_comp(split[0])
                axon_swc = split[2]
                linker_swc = split[1]
                dend_swc = split[0]
                linker = True
                linker_neurons.append(n.id)
            else:
                if split[2].compartment == 'dendrite' and split[0].compartment == 'axon':
                    axon = parents_check_on_comp(split[0])
                    dend = parents_check_on_comp(split[2])
                    axon_swc = split[0]
                    linker_swc = split[1]
                    dend_swc = split[2]
                    linker = True
                    linker_neurons.append(n.id)
                else:
                    issues_in_comp.append([n.id, 'issue1'])
  

                continue

        elif len(split) == 2:
            if split[0].compartment == 'dendrite' and split[1].compartment == 'axon':
                axon = parents_check_on_comp(split[1])
                dend = parents_check_on_comp(split[0])
                axon_swc = split[1]
                dend_swc = split[0]
            else:
                if split[1].compartment == 'dendrite' and split[0].compartment == 'axon':
                    axon = parents_check_on_comp(split[0])
                    dend = parents_check_on_comp(split[1])
                    axon_swc = split[0]
                    dend_swc = split[1]
                else:
                    issues_in_comp.append([n.id, 'issue2'])
                    all_non_succsessfull_connectors = split.connectors
                    with open(os.path.join(output_folder, f'non_success_connectors_{swcid}.pkl'), 'wb') as f:
                        pickle.dump(all_non_succsessfull_connectors, f)

                continue

        else:
            issues_in_comp.append([n.id, 'issue3'])
            continue

        nid = n.id

        SI, IG = SI_calc(['', (len(axon_swc.presynapses), len(axon_swc.postsynapses)),
                          (len(dend_swc.presynapses), len(dend_swc.postsynapses))])
        all_SI.append([nid, SI])

        axon_connectors = axon_swc.connectors
        axon_connectors['compartment'] = 'A'
        dend_connectors = dend_swc.connectors
        dend_connectors['compartment'] = 'D'

        if linker:
            linker_connectors = linker_swc.connectors
            linker_connectors['compartment'] = 'L'
            neuron_connectors = pd.concat([axon_connectors, dend_connectors, linker_connectors])
        else:
            neuron_connectors = pd.concat([axon_connectors, dend_connectors])

        neuron_connectors[['neuron']] = nid
        neuron_connectors = neuron_connectors.reset_index(drop=True)

        # --- per-neuron filenames (suffix _<swcid>) ---
        swcid = swc1.id

        with open(os.path.join(output_folder, f'connectors_{swcid}.pkl'), 'wb') as f:
            pickle.dump(neuron_connectors, f)

        with open(os.path.join(output_folder, f'all_SI_{swcid}.pkl'), 'wb') as f:
            pickle.dump(all_SI, f)

        with open(os.path.join(output_folder, f'issues_{swcid}.pkl'), 'wb') as f:
            pickle.dump(issues_in_comp, f)

        with open(os.path.join(output_folder, f'linker_{swcid}.pkl'), 'wb') as f:
            pickle.dump(linker_neurons, f)

        # (added) save non-success connectors per neuron


    except Exception as e:
        # swc1.id if available, otherwise fallback to id_
        swcid = (swc1.id if 'swc1' in locals() else id_)
        issues_in_comp.append([swcid, 'issue3'])
        print(e)
        with open(os.path.join(output_folder, f'issues_{swcid}.pkl'), 'wb') as f:
            pickle.dump(issues_in_comp, f)
            
            
#%%

    #%%
    

# directory where files are stored
base_dir = PROCESSED_BIG_NEURONS_DIR

dfs = []

for fname in os.listdir(base_dir):
    if fname.startswith("connectors_"):
        path = os.path.join(base_dir, fname)

        if fname.endswith(".pkl"):
            df = pd.read_pickle(path)
        elif fname.endswith(".ftr"):
            df = pd.read_feather(path)
        elif fname.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            continue

        # keep source ID (very important)
        dfs.append(df)

connectors_df = pd.concat(dfs, ignore_index=True)

#%%
dfs = []

for fname in os.listdir(base_dir):
    if fname.startswith("all_SI_"):
        path = os.path.join(base_dir, fname)

        obj = pd.read_pickle(path)  # this is a list
        sid = fname.split("_", 1)[1].split(".", 1)[0]

        for v in obj:
            dfs.append({"source_id": sid, "value": v})

all_SI_df = pd.DataFrame(dfs)
#%%
out = pd.DataFrame({
    "nid": all_SI_df["value"].apply(lambda x: int(x[0])),
    "SI":  all_SI_df["value"].apply(lambda x: float(x[1]))
})
#%%

#%%
dfs = []

for fname in os.listdir(base_dir):
    if fname.startswith("issues_"):
        path = os.path.join(base_dir, fname)

        obj = pd.read_pickle(path)  # this is a list
        sid = fname.split("_", 1)[1].split(".", 1)[0]

        for v in obj:
            dfs.append({"source_id": sid, "value": v})

issues_df = pd.DataFrame(dfs)
#%%
dfs = []

for fname in os.listdir(base_dir):
    if fname.startswith("linker_"):
        path = os.path.join(base_dir, fname)

        obj = pd.read_pickle(path)  # this is a list
        sid = fname.split("_", 1)[1].split(".", 1)[0]

        for v in obj:
            dfs.append({"source_id": sid, "value": v})

linker_df = pd.DataFrame(dfs)
#%%
#TDL:
'''
    how mamy skeleton? how many classification neurons? how many we processed and how many finished processing?
    check neuron diff:
        1) big
        2) small

    how many nodes and synapses in small?

    add big
    add linker
    re do analyses with new ptypes
'''

#%%
missing_in_nodesG = df.loc[~df['root_id'].isin(nodesG['root_id'])]
#%%

base_dir = PROCESSED_BIG_NEURONS_DIR

dfs = []

for fname in os.listdir(base_dir):
    if fname.startswith("connectors_"):
        path = os.path.join(base_dir, fname)

        if fname.endswith(".pkl"):
            df = pd.read_pickle(path)
        elif fname.endswith(".ftr"):
            df = pd.read_feather(path)
        elif fname.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            continue

        # keep source ID (very important)
        dfs.append(df)

all_connectors = pd.concat(dfs, ignore_index=True)
#%%
all_connectors=all_connectors.drop(columns=['connector_id','node_id'])

#%%


#%%

pre_connectors=all_connectors[all_connectors['type']=='pre']
#%%

aaa=pre_connectors.head()
 
#%% 
pre_connectors.columns=['pre_x','pre_y','pre_z','post','synapse_id','size','type','pre_compartment','neuron']
pre_connectors=pre_connectors.drop(columns=['type'])

#%%
CONNECTORS_DIR.mkdir(parents=True, exist_ok=True)
pre_connectors.reset_index(drop=True).to_feather(PRE_CONNECTORS_BIGN_FTR)
#%%
del pre_connectors
#%%

post_connectors=all_connectors[all_connectors['type']=='post']

post_connectors.columns=['post_x','post_y','post_z','pre','synapse_id','size','type','post_compartment','neuron']
#%%
post_connectors.reset_index(drop=True).to_feather(POST_CONNECTORS_BIGN_FTR)
#%%






