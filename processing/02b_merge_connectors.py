

x=5
y=10
z=x+y

list_=[x,y,z]
#%%
#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import time
import datetime
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROCESSED_SWC_DIR, CONNECTORS_DIR,
    PRE_CONNECTORS_FTR, POST_CONNECTORS_FTR,
)
import networkx as nx

from sklearn.cluster import AgglomerativeClustering
import random
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pickle

#%%
#allsynapses=pd.read_feather(r'C:\Users\user\organised_work\data\connections\synapses')

#%%

#%%
# Define the base directories
input_base_dir = PROCESSED_SWC_DIR
#output_base_dir = r"C:\Users\user\organised_work\data\783\generated\783\trees\all_trees"


# Get the list of subdirectories in the input directory
input_subfolders = {f.name for f in os.scandir(input_base_dir) if f.is_dir() }

#%%

#%%

all_connectors_list = []
i = 0

for folder_name in input_subfolders:
    folder_name = str(folder_name)

    folder_path = os.path.join(input_base_dir, folder_name)

    connectors_data_file_path = os.path.join(folder_path, 'connectors.pkl')

    with open(connectors_data_file_path, 'rb') as f:
        connectors_data = pickle.load(f)
        print(len(connectors_data))
        connectors_data=connectors_data.drop(columns=['connector_id','node_id'])
        all_connectors_list.append(connectors_data)

    i += 1
    #%%
all_connectors = pd.concat(all_connectors_list).reset_index(drop=True)

        #%%
del all_connectors_list        
    #%%
#%%    
pre_connectors=all_connectors[all_connectors['type']=='pre'] 
#%% 
pre_connectors.columns=['pre_x','pre_y','pre_z','post','type','pre_compartment','pre','synapse_id','size']
pre_connectors=pre_connectors.drop(columns=['type'])
#%%
CONNECTORS_DIR.mkdir(parents=True, exist_ok=True)
pre_connectors.reset_index(drop=True).to_feather(PRE_CONNECTORS_FTR)
#%%
del pre_connectors
#%%


post_connectors=all_connectors[all_connectors['type']=='post']    
post_connectors.columns=['post_x','post_y','post_z','pre','type','post_compartment','post','synapse_id','size']
post_connectors=post_connectors.drop(columns=['type'])
#%%
post_connectors.reset_index(drop=True).to_feather(POST_CONNECTORS_FTR)
#%%
