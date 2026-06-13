# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 17:29:54 2026

@author: user
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import METHODS_DIR, NEURON_TABLE_FTR, OUTPUT_DIR, NEUROPIL_SYNAPSE_CSV
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import os
import seaborn as sns
import pickle

from scipy.stats import gaussian_kde
import matplotlib as mpl

#%%
path = NEUROPIL_SYNAPSE_CSV

allsynapses = pd.read_csv(
    path,
    usecols=[1,2,3,7,8,9,10,11,12,13]
)
#%%7720575940610757204

allsynapses.columns=['pre_x','pre_y','pre_z','post_x','post_y','post_z','size','pre','post','npil']
#%%
allsynapses=allsynapses[allsynapses['pre']!=allsynapses['post']]

#%%
aa=aa.reset_index()

#%%
#%%
allsynapses=allsynapses.reset_index()
#%%
allsynapses=allsynapses.rename(columns={'index':'synapse_id'})
#%%
allsynapses.columns=['synapse_id','pre_x','pre_y','pre_z','post_x','post_y','post_z','size','pre','post','npil']

#%%

nodesG=pd.read_feather(NEURON_TABLE_FTR)


#%%
vneuron=720575940620797269
#%%
SI_list=[]
random_neurons=nodesG.sample(10000)['neuron']

#%%
ii=0
for vneuron in random_neurons:
    print(ii)
    ii+=1
    try:
        swc = upload_swc(vneuron)
        healed_neurons=navis.heal_skeleton(swc)
        allsynapses2=allsynapses[(allsynapses['pre']==vneuron)|(allsynapses['post']==vneuron)]
        healed_neurons_att_syn=attach_synapses_princeton(healed_neurons,allsynapses2)
        split = get_split(healed_neurons_att_syn)
        SI=navis.segregation_index(split)
        SI_list.append([vneuron,SI])
    except Exception as e:
        print(e)
#%%
SI_df=pd.DataFrame(SI_list)

SI_df.columns=['neuron','SI_new']

SI_df=SI_df.merge(nodesG[['neuron','SI']],how='left',on='neuron')
#%%
ax = sns.scatterplot(data=SI_df, x='SI', y='SI_new')

corr = SI_df[['SI_new', 'SI']].corr().iloc[0, 1]

ax.text(
    0.05, 0.95,
    f'r = {corr:.3f}',
    transform=ax.transAxes,
    ha='left',
    va='top'
)

_out_dir = OUTPUT_DIR / "fig2" / "SI_x_twigs"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "SI_twigs.svg")
#%%
print(f'corr =\n{SI_df[["SI_new", "SI"]].corr()}')

#%%
print(f'corr =\n{SI_df[["SI_new", "SI"]].corr()}')
#%%
