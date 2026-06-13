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
    SYNAPSE_NON_PROCESSED_FTR, PROCESSED_SWC_DIR,
    PROCESSED_BIG_NEURONS_DIR, SYNAPSE_TABLE_FTR,
    DERIVED_DATA_DIR,
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

syn_df=pd.read_feather(SYNAPSE_NON_PROCESSED_FTR)
#%%

#%%
syn_df.dropna(subset=['post_compartment','pre_compartment'],inplace=True)
#%%
#syn_df=syn_df.drop_duplicates()???

#%%
aaa=syn_df.head()

#%%

#%%
null_counts = syn_df.isnull().sum()

print(null_counts)
#%%
none_total = (syn_df == None).sum().sum()
print("Total None values in DataFrame:", none_total)

#%%
dup_count = syn_df.duplicated().sum()
print("Number of duplicate rows:", dup_count)
#%%
dup_count = syn_df.duplicated(subset=['pre','post','pre_x','pre_y','pre_z']).sum()
print("Number of duplicates based on these columns:", dup_count)
#%%
dup_count = syn_df.duplicated(subset=['pre','post','post_x','post_y','post_z']).sum()
print("Number of duplicates based on these columns:", dup_count)
#%%
duplicates = syn_df[syn_df.duplicated( keep=False)]

#%%
#%%add_si!
import os
import pickle
# to make sure u take the right iteraiton! Majd
#base_dir = r'C:\Users\user\organised_work\data\783\generated\783\trees\all_trees_princeton'
base_dir = PROCESSED_SWC_DIR

all_si_list = []

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):
        pkl_path = os.path.join(folder_path, 'all_SI.pkl')
        if os.path.exists(pkl_path):
            SI = pd.read_pickle(pkl_path)
            si_df = pd.DataFrame(SI)
            si_df.columns = ['nid', 'SI']
            all_si_list.append(si_df)
#%%
SI_df = pd.concat(all_si_list, ignore_index=True)
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

all_SI_df = pd.DataFrame(dfs)
#%%
out = pd.DataFrame({
    "nid": all_SI_df["value"].apply(lambda x: int(x[0])),
    "SI":  all_SI_df["value"].apply(lambda x: float(x[1]))
})
#%%


SI_df=pd.concat([SI_df,out])
#%%
SI_df['nid']=SI_df['nid'].astype(np.int64)
#%%
syn_df_SI_pre=syn_df.merge(SI_df,left_on='pre',right_on='nid',how='left')
#%%

del syn_df
#%%
syn_df_SI=syn_df_SI_pre.merge(SI_df,left_on='post',right_on='nid',how='left')
#%%
del syn_df_SI_pre

#%%

#%%

syn_df_SI2=syn_df_SI.dropna()
#%%
del syn_df_SI
#%%
#%%
syn_df_SI2['comp']=syn_df_SI2['pre_compartment']+syn_df_SI2['post_compartment']
#%%
syn_df_SI2.columns

#%%
syn_df_SI2=syn_df_SI2[['synapse_id','pre', 'post', 
       'post_x', 'post_y', 'post_z', 'pre_x', 'pre_y', 'pre_z',
       'neuropil', 'comp','SI_x','SI_y']]

#%%
syn_df_SI2.columns=['synapse_id', 'pre', 'post',
       'post_x', 'post_y', 'post_z', 'pre_x', 'pre_y', 'pre_z', 
       'npil', 'comp','SI_pre','SI_post']
#%%

a=syn_df_SI2.head(1000)

#%%
DERIVED_DATA_DIR.mkdir(parents=True, exist_ok=True)
syn_df_SI2.reset_index(drop=True).to_feather(SYNAPSE_TABLE_FTR)
#%%
