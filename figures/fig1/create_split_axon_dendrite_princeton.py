
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import METHODS_DIR, NEURON_TABLE_FTR, SYNAPSE_TABLE_RAW_FTR, OUTPUT_DIR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import os
import seaborn as sns
import pickle
#%%
#need to add to scripts here:
    #t1_reci_per_neuron
    #reci_running_neurons_on_self
#%%
#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
#%%
a=nodesG.head()

df=nodesG[nodesG['super_class'].isin(['central','visual_projection','visual_centrifugal','optic'])]
#%%

df=df[['pre_A','post_A','pre_D','post_D']].sum()


#%%raw?


allsynapses=pd.read_feather(SYNAPSE_TABLE_RAW_FTR)
#%%
c=nodesG[nodesG['super_class']=='central']
c=c[c['SI']<0.1]

vp=nodesG[nodesG['super_class']=='visual_projection']
vp=vp[vp['SI']<0.1]

vc=nodesG[nodesG['super_class']=='visual_centrifugal']
vc=vc[vc['SI']<0.1]

op=nodesG[nodesG['super_class']=='optic']
op=op[op['SI']<0.1]
#%%
r=nodesG[nodesG['neuron']==720575940627236874]
#%%
for i in c['neuron'].iloc[10000:10050]:
    print(i)
#%%
ex=nodesG[nodesG['SI']>=0.2]
#%%
gg=create_links_many_neurons([720575940627236874],allsynapses)
shorthen_and_open_links(gg[0])
#%%
#129 1.71867
#720575940621448357 1.90444
id_=720575940620605776  #-0.704227
swc1=upload_swc(id_)

allsynapses2=allsynapses[(allsynapses['pre']==id_)|(allsynapses['post']==id_)]

neuron=heal_attach_princeton_non_process(swc1,allsynapses2)
split1=get_split(neuron,flow_thresh = 1)
#%%
navis.segregation_index(navis.NeuronList([split1[0],split1[2]]))
#%%	720575940641074829	720575940627938122


id_=720575940620451974 #-0.818067
swc1=upload_swc(id_)

allsynapses2=allsynapses[(allsynapses['pre']==id_)|(allsynapses['post']==id_)]

neuron=heal_attach_princeton_non_process(swc1,allsynapses2)
split2=get_split(neuron)
#%%
aa=nodesG[nodesG['neuron']==720575940625630212]
#%%
print(f"percent is: {(aa['pre_A'] / (aa['pre_A'] + aa['post_A'])).iloc[0]:.3f}, axon is {aa['pre_A'].iloc[0]}, and total is {(aa['pre_A'] + aa['post_A']).iloc[0]}")
print(f"percent is: {(aa['post_D'] / (aa['post_D'] + aa['pre_D'])).iloc[0]:.3f}, dend is {aa['post_D'].iloc[0]}, and total is {(aa['pre_D'] + aa['post_D']).iloc[0]}")


#%%720575940616741121,720575940621832397

# Grab your neuron
# Plot
fig, ax = navis.plot2d(split1,connectors=False,method='3d',colors=['#fbb040','g','#92278f'],cn_size=10,linewidth=0.5)
ax.elev = -94
ax.azim = 88
ax.roll = 170
# Change background color
ax.set_facecolor('white')  

# Make background patch visible again
fig.patch.set_alpha(1)
ax.patch.set_alpha(1)
#%%fbb040
fig, ax = navis.plot2d([split2,split1],connectors=False,method='3d',colors=['#fbb040','g','#92278f','#fbb040','g','#92278f'],cn_size=30,linewidth=0.5)
ax.elev = 40
ax.azim = 8
ax.roll = -90
# Change background color
ax.set_facecolor('white')  

# Make background patch visible again
fig.patch.set_alpha(1)
ax.patch.set_alpha(1)
    #%%
_out_dir = OUTPUT_DIR / "fig3" / "pc1_example"
_out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(
    _out_dir / "pc1_example_AA1.svg",
    format='svg',
    bbox_inches='tight',
    transparent=True
)
