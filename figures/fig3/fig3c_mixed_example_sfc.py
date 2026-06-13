
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import NEURON_TABLE_FTR, SYNAPSE_TABLE_FTR, OUTPUT_DIR, METHODS_DIR
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
allsynapses=pd.read_feather(SYNAPSE_TABLE_FTR)
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
r=nodesG[nodesG['neuron']==720575940639598909]
#%%
for i in c['neuron'].iloc[10000:10050]:
    print(i)
#%%
ex=nodesG[nodesG['SI']>=0.2]
#%%
gg=create_links_many_neurons([720575940621470028],allsynapses)
shorthen_and_open_links(gg[0])
#%%


id_=720575940616113514
swc1=upload_swc(id_)

allsynapses2=allsynapses[(allsynapses['pre']==id_)|(allsynapses['post']==id_)]

neuron=heal_attach_princeton_non_process(swc1,allsynapses2)
split1=get_split(neuron)

#%%
aa=nodesG[nodesG['neuron']==720575940616113514]
#%%
print(f"percent is: {(aa['pre_A'] / (aa['pre_A'] + aa['post_A'])).iloc[0]:.3f}, axon is {aa['pre_A'].iloc[0]}, and total is {(aa['pre_A'] + aa['post_A']).iloc[0]}")
print(f"percent is: {(aa['post_D'] / (aa['post_D'] + aa['pre_D'])).iloc[0]:.3f}, dend is {aa['post_D'].iloc[0]}, and total is {(aa['pre_D'] + aa['post_D']).iloc[0]}")


#%%

# Grab your neuron
# Plot
fig, ax = navis.plot2d(split1,connectors=False,method='3d',colors=['#fbb040','g','#92278f'],cn_size=30,linewidth=1)
ax.elev = 80
ax.azim = 80
ax.roll = -10
# Change background color
ax.set_facecolor('white')  

# Make background patch visible again
fig.patch.set_alpha(1)
ax.patch.set_alpha(1)
    #%%
_out_dir = OUTPUT_DIR / "fig3" / "mixed_example"
_out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(
    _out_dir / "axon_dendrite_example_720575940616113514.svg",
    format='svg',
    bbox_inches='tight',
    transparent=True
)
