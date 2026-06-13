
from nglui.statebuilder import ChainedStateBuilder
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'



import matplotlib.pyplot as plt
import numpy as np
from fafbseg import flywire
import navis
import pandas as pd

import time
import datetime
import os
import sys
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import NEURON_TABLE_FTR, NEURON_ANNOTATIONS_CSV, OUTPUT_DIR, METHODS_DIR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
client = CAVEclient('flywire_fafb_production')
import seaborn as sns
import matplotlib.pyplot as plt

import os
import time
import pickle
import navis

import matplotlib.colors as mcolors
import matplotlib.cm as cm



#%%
#allsynapses=pd.read_feather(r'C:\Users\user\organised_work\data\783\generated\post_processing_data\synapses_783_final.ftr')
#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
nodesG=nodesG.dropna(subset=['axon_correct','dend_correct','primary_type','super_class'])
#%%

n=pd.read_csv(NEURON_ANNOTATIONS_CSV)
#%%
n=n.rename(columns={'root_id':'neuron'})
#%%
nodesG=nodesG.merge(n[['neuron','cell_sub_class']],how='left',on='neuron')

#%%
des=nodesG[nodesG['super_class']=='descending']
l=des[des['SI']<=0.35]
h=des[des['SI']>0.35]

#%%
allowed_values = ['optic', 'central', 'visual_projection','visual_centrifugal']

# Filter the DataFrame
nodesG = nodesG[nodesG['super_class'].isin(allowed_values)]

#%%
ring_neurons=nodesG[nodesG['cell_sub_class']=='ring neuron']
kc_neurons = nodesG[nodesG['primary_type'].str.startswith('KC', na=False)]
dm_neurons = nodesG[nodesG['primary_type'].str.startswith('Dm', na=False)]
lca_neurons = nodesG[nodesG['primary_type'].str.startswith('LC10a', na=False)]
lcb_neurons = nodesG[nodesG['primary_type'].str.startswith('LC10b', na=False)]
lcc_neurons = nodesG[nodesG['primary_type'].str.startswith('LC10c', na=False)]
lcd_neurons = nodesG[nodesG['primary_type'].str.startswith('LC10d', na=False)]

T4a_neurons = nodesG[nodesG['primary_type'].str.startswith('T4a', na=False)]
T4b_neurons = nodesG[nodesG['primary_type'].str.startswith('T4b', na=False)]
T4c_neurons = nodesG[nodesG['primary_type'].str.startswith('T4c', na=False)]
T4d_neurons = nodesG[nodesG['primary_type'].str.startswith('T4d', na=False)]



#%%
lc_neurons=pd.concat([lca_neurons,lcb_neurons,lcc_neurons,lcd_neurons])
T4_neurons=pd.concat([T4b_neurons,T4a_neurons,T4c_neurons,T4d_neurons])

#%%
kc_g = list(kc_neurons.groupby(by='primary_type').size().reset_index(name='count').itertuples(index=False, name=None))
dm_g = list(dm_neurons.groupby(by='primary_type').size().reset_index(name='count').itertuples(index=False, name=None))
lc_g = list(lc_neurons.groupby(by='primary_type').size().reset_index(name='count').itertuples(index=False, name=None))
ring_g = list(ring_neurons.groupby(by='primary_type').size().reset_index(name='count').itertuples(index=False, name=None))
T4_g = list(T4_neurons.groupby(by='primary_type').size().reset_index(name='count').itertuples(index=False, name=None))

#%%
# Plot histograms with different colors
sns.histplot(kc_neurons['SI'], fill=True, stat='density', color='green', label='DM Neurons')
#%%
sns.histplot(dm_neurons, fill=True, stat='density', color='green', label='DM Neurons')
sns.histplot(lc_neurons, fill=True, stat='density', color='red', label='lc Neurons')
sns.histplot(ring_neurons, fill=True, stat='density', color='yellow', label='ring Neurons')

# Add legend
plt.legend(title='Neuron Type')

# Show the plot
plt.show()

#%%
# Plot histograms with different colors
sns.histplot(T4_neurons['SI'], fill=True, stat='density', color='blue', label='KC Neurons')
sns.histplot(dm_neurons['SI'], fill=True, stat='density', color='green', label='DM Neurons')
sns.histplot(ring_neurons['SI'], fill=True, stat='density', color='red', label='Ring Neurons')
sns.histplot(lc_neurons['SI'], fill=True, stat='density', color='orange', label='Lc Neurons')

# Add legend
plt.legend(title='Neuron Type')

# Show the plot
plt.show()
#%%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(1.7,1.7))

# Plot the KDE curves
sns.kdeplot(T4_neurons['SI'], fill=True, color='blue', linewidth=0.5)
sns.kdeplot(dm_neurons['SI'], fill=True, color='green', linewidth=0.5)
sns.kdeplot(ring_neurons['SI'], fill=True, color='red', linewidth=0.5)
sns.kdeplot(lc_neurons['SI'], fill=True, color='orange', linewidth=0.5)

# Add text labels stacked in the top-right corner
x_pos = 0.35
y_start = 29
line_spacing = 3.5

plt.text(x_pos, y_start - 0 * line_spacing, f'T4 Neurons, n = {len(T4_neurons)}', color='blue', fontsize=8)
plt.text(x_pos, y_start - 1 * line_spacing, f'DM Neurons, n = {len(dm_neurons)}', color='green', fontsize=8)
plt.text(x_pos, y_start - 2 * line_spacing, f'ring Neurons, n = {len(ring_neurons)}', color='red', fontsize=8)
plt.text(x_pos, y_start - 3 * line_spacing, f'LC Neurons, n = {len(lc_neurons)}', color='orange', fontsize=8)

# Labels and styling
plt.xlabel('SI', size=8)
plt.ylabel('Density', size=8)
plt.xlim(0, 0.5)
plt.ylim(0, 30)
plt.xticks([0,0.1,0.2,0.4,0.6], size=4)  # explicitly add 0.1 as a tick
plt.axvline(x=0.1, lw=0.1)

# Get current axis and color the 0.1 tick blue
ax = plt.gca()
for label in ax.get_xticklabels():
    if label.get_text() == '0.1':
        label.set_color('blue')

sns.despine(top=True, right=True)

# Save and show
_out_dir = OUTPUT_DIR / "fig3" / "si_x_neuron_types"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "SI_ptypes_inline_labels_princeton.svg")
plt.show()

#%%
counts_kc, bin_edges_kc = np.histogram(kc_neurons['SI'], density=False)
# Compute histogram for ring neurons
counts_ring, bin_edges_ring = np.histogram(ring_neurons['SI'], density=False)
counts_dm, bin_edges_dm = np.histogram(dm_neurons['SI'],density=False)

# Print Bin Counts and Edges for Ring Neurons
print("Bin Counts (Ring):", counts_ring)
print("Bin Edges (Ring):", bin_edges_ring)
print("Bin Counts:", counts_kc)
print("Bin Edges:", bin_edges_kc)



print("Bin Counts:", counts_dm)
print("Bin Edges:", bin_edges_dm)
# Calculate bin midpoints for ring neurons
bin_midpoints_ring = 0.5 * (bin_edges_ring[:-1] + bin_edges_ring[1:])

# Normalize counts to proportions for ring neurons
bin_midpoints_kc = 0.5 * (bin_edges_kc[:-1] + bin_edges_kc[1:])
bin_midpoints_dm = 0.5 * (bin_edges_dm[:-1] + bin_edges_dm[1:])
# Normalize counts to proportions
proportions_kc = counts_kc / counts_kc.sum()
proportions_dm = counts_dm / counts_dm.sum()
proportions_ring = counts_ring / counts_ring.sum()

# Plot line plots for KC and DM neurons

# Plot line plots for KC, DM, and Ring neurons
plt.plot(bin_midpoints_kc, proportions_kc, label='KC Neurons', marker='o', linestyle='-', markersize=2, color='blue')
plt.plot(bin_midpoints_dm, proportions_dm, label='DM Neurons', marker='o', linestyle='-', markersize=2, color='green')
plt.plot(bin_midpoints_ring, proportions_ring, label='Ring Neurons', marker='o', linestyle='-', markersize=2, color='red')

# Add labels and title
plt.xlabel('SI (Bin Midpoints)')
plt.ylabel('Proportions')
plt.title('Neuron SI Distribution (Proportions)')
plt.legend()
plt.xlim(0, 0.4)  # Limit the x-axis range
plt.grid(True)  # Add grid for better readability

# Show the plot
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate bins dynamically
def calculate_bins(data):
    return int(np.sqrt(len(data)))  # You can swap with other methods if needed

# Dynamically determine bins for each class
bins_kc = calculate_bins(kc_neurons['SI'])
bins_dm = calculate_bins(dm_neurons['SI'])
bins_ring = calculate_bins(ring_neurons['SI'])

# Compute histograms with dynamic bins
counts_kc, bin_edges_kc = np.histogram(kc_neurons['SI'], bins=bins_kc, density=False)
counts_dm, bin_edges_dm = np.histogram(dm_neurons['SI'], bins=bins_dm, density=False)
counts_ring, bin_edges_ring = np.histogram(ring_neurons['SI'], bins=bins_ring, density=False)

# Calculate bin midpoints
bin_midpoints_kc = 0.5 * (bin_edges_kc[:-1] + bin_edges_kc[1:])
bin_midpoints_dm = 0.5 * (bin_edges_dm[:-1] + bin_edges_dm[1:])
bin_midpoints_ring = 0.5 * (bin_edges_ring[:-1] + bin_edges_ring[1:])

# Normalize counts to proportions
proportions_kc = counts_kc / counts_kc.sum()
proportions_dm = counts_dm / counts_dm.sum()
proportions_ring = counts_ring / counts_ring.sum()

# Plot line plots for KC, DM, and Ring neurons
plt.plot(bin_midpoints_kc, proportions_kc, label='KC Neurons', marker='o', linestyle='-', markersize=2, color='blue')
plt.plot(bin_midpoints_dm, proportions_dm, label='DM Neurons', marker='o', linestyle='-', markersize=2, color='green')
plt.plot(bin_midpoints_ring, proportions_ring, label='Ring Neurons', marker='o', linestyle='-', markersize=2, color='red')

# Add labels and title
plt.xlabel('SI (Bin Midpoints)')
plt.ylabel('Proportions')
plt.title('Neuron SI Distribution (Proportions)')
plt.legend()
plt.xlim(0, 0.4)  # Limit the x-axis range
plt.grid(True)

# Show the plot
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate bins using Freedman-Diaconis Rule
def calculate_bins_iqr(data):
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / len(data)**(1/3)
    return max(1, int(np.ceil((data.max() - data.min()) / bin_width)))  # Ensure at least 1 bin

# Dynamically determine bins for each class using IQR
bins_kc = calculate_bins_iqr(kc_neurons['SI'])
bins_dm = calculate_bins_iqr(dm_neurons['SI'])
bins_ring = calculate_bins_iqr(ring_neurons['SI'])

# Compute histograms with dynamic bins
counts_kc, bin_edges_kc = np.histogram(kc_neurons['SI'], bins=bins_kc, density=False)
counts_dm, bin_edges_dm = np.histogram(dm_neurons['SI'], bins=bins_dm, density=False)
counts_ring, bin_edges_ring = np.histogram(ring_neurons['SI'], bins=bins_ring, density=False)

# Calculate bin midpoints
bin_midpoints_kc = 0.5 * (bin_edges_kc[:-1] + bin_edges_kc[1:])
bin_midpoints_dm = 0.5 * (bin_edges_dm[:-1] + bin_edges_dm[1:])
bin_midpoints_ring = 0.5 * (bin_edges_ring[:-1] + bin_edges_ring[1:])

# Normalize counts to proportions
proportions_kc = counts_kc / counts_kc.sum()
proportions_dm = counts_dm / counts_dm.sum()
proportions_ring = counts_ring / counts_ring.sum()

# Plot line plots for KC, DM, and Ring neurons
plt.plot(bin_midpoints_kc, proportions_kc, label='KC Neurons', marker='o', linestyle='-', markersize=2, color='blue')
plt.plot(bin_midpoints_dm, proportions_dm, label='DM Neurons', marker='o', linestyle='-', markersize=2, color='green')
plt.plot(bin_midpoints_ring, proportions_ring, label='Ring Neurons', marker='o', linestyle='-', markersize=2, color='red')

# Add labels and title
plt.xlabel('SI (Bin Midpoints)')
plt.ylabel('Proportions')
plt.title('Neuron SI Distribution (Proportions)')
plt.legend()
plt.xlim(0, 0.4)  # Limit the x-axis range
plt.grid(True)

# Show the plot
plt.show()

