# -*- coding: utf-8 -*-
"""
Builds PC1_table.csv: one row per connection between intrinsic neurons, with the
PC1 score, super-class, NT and SI of the pre- and postsynaptic neuron and the
AA/AD/DA/DD synapse counts.

Run it before figures/fig4/syntype_x_pc1.py, which reads this table and adds the
model predictions.
"""
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

import matplotlib.patches as mpatches

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (NEURON_TABLE_FTR, PCA_TABLE_FTR, CONNECTIONS_TABLE_FTR,
                    PC1_TABLE_CSV, METHODS_DIR)
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import os
import seaborn as sns
import pickle

from scipy.stats import gaussian_kde


#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
#%%

#%%
nodesG=nodesG[['neuron','super_class','SI','nt_type' ]]
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]

#%%

pcadf=pd.read_feather(PCA_TABLE_FTR)
#%%

nodesG=nodesG.merge(pcadf,on='neuron',how='left')
#%%



#%%
connections=pd.read_feather(CONNECTIONS_TABLE_FTR)

#%%
connections=connections[['pre', 'post', 'AA', 'AD', 'DA', 'DD', 
       'sum_syn', 'reciprocal', 
       'same_type']]


#%%
#%%

#%%
connections=connections.merge(nodesG[['neuron','SI','super_class','nt_type','PC1']],left_on='pre',right_on='neuron',how='left')
connections=connections.merge(nodesG[['neuron','SI','super_class','nt_type','PC1']],left_on='post',right_on='neuron',how='left')
#%%
a=connections.head()
#%%
connections = connections[connections['super_class_x'].isin(['central', 'optic', 'visual_centrifugal', 'visual_projection'])]
connections = connections[connections['super_class_y'].isin(['central', 'optic', 'visual_centrifugal', 'visual_projection'])]



#%%
connections2=connections[['pre', 'post', 'AA', 'AD', 'DA', 'DD', 'reciprocal',
       'same_type', 'SI_x', 'super_class_x', 'nt_type_x',
       'SI_y', 'super_class_y', 'nt_type_y', 'neuron_x','PC1_x',
       'neuron_y', 'PC1_y']]
#%%
connections2.columns=['pre', 'post', 'AA', 'AD', 'DA', 'DD', 'reciprocal',
       'same_type', 'SI_pre', 'super_class_pre', 'nt_type_pre',
       'SI_post', 'super_class_post', 'nt_type_post', 'neuron_pre', 'PC1_pre',
       'neuron_post', 'PC1_post']
#%%
connections2=connections2[['PC1_pre','PC1_post','pre','post','super_class_pre','super_class_post','nt_type_pre','nt_type_post','SI_pre','SI_post','AA', 'AD', 'DA', 'DD','reciprocal','same_type']]
#%%
connections2[['pre']]=connections2[['pre']].astype(str)
connections2[['post']]=connections2[['post']].astype(str)


#%%
connections2.to_csv(PC1_TABLE_CSV)
