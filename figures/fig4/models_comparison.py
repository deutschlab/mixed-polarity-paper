import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

import os
import time
import pickle
import navis 
import sys
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import NEURON_TABLE_FTR, SYNAPSE_TABLE_FTR, CONNECTIONS_TABLE_FTR, OUTPUT_DIR, METHODS_DIR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *

#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
#%%

nodesG=nodesG[['neuron','super_class','primary_type','SI' ]]
#%%

nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
#%%
syn_df=pd.read_feather(SYNAPSE_TABLE_FTR)
#%%

connections=pd.read_feather(CONNECTIONS_TABLE_FTR)
#%%
connections=connections.merge(nodesG[['neuron','super_class','SI']],left_on='pre',right_on='neuron',how='left')
connections=connections.merge(nodesG[['neuron','super_class','SI']],left_on='post',right_on='neuron',how='left').dropna()
#%%
aa=connections.head()
#%%
connections=connections[(connections['SI_x']>=0.1)&(connections['SI_y']>=0.1)]

#%%
connections=connections[['AA', 'AD', 'DA', 'DD', 'sum_syn', 'reciprocal',
       'primary_type_x', 'primary_type_y']]
#%%

connections['sum_syn'] = connections[['AA', 'AD', 'DA', 'DD']].sum(axis=1)

#%%
connections['max_syn'] = connections[['AA', 'AD', 'DA', 'DD']].max(axis=1)
#%%
result_per_conneciton=connections['max_syn'].sum()/connections['sum_syn'].sum()
#%%

connections_agg=connections.groupby(['primary_type_x', 'primary_type_y'])[['AA', 'AD', 'DA', 'DD']].sum()
connections_agg['sum_syn']=connections_agg.T.sum()
connections_agg['max_syn'] = connections_agg[['AA', 'AD', 'DA', 'DD']].max(axis=1)
result_per_conneciton_primary_type=connections_agg['max_syn'].sum()/connections_agg['sum_syn'].sum()
#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Data
x = ['per neuron', 'per primary_type', 'all features', 'pc1_x/pc1_y']
y = [result_per_conneciton*100, 100*result_per_conneciton_primary_type, 79.353, 69.245]

# Create DataFrame
df = pd.DataFrame({'x': x, 'y': y})

# Create the plot
plt.figure(figsize=(5, 4))
sns.pointplot(data=df, x='x', y='y', color='black', join=False, markers='o', scale=1.3)

# Formatting
plt.ylabel('Accuracy (%)')
plt.xlabel('')
plt.ylim(65, 100)
plt.title('Model Accuracy by Feature Type', pad=15)
plt.grid(False)
plt.tight_layout()
_out_dir = OUTPUT_DIR / "fig4" / "models_comparison"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "accuracy_values.svg")
sns.despine(top=True, right=True)
plt.show()
#%%
