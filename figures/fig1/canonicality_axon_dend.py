import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import METHODS_DIR, NEURON_TABLE_FTR, OUTPUT_DIR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import os
import seaborn as sns
import pickle
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'  # or 'Helvetica' or any standard system font
from scipy.stats import gaussian_kde
#%%

# Define the Freedman–Diaconis rule to calculate bin width
def freedman_diaconis(data):
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25  # Interquartile range
    bin_width = 2 * iqr / (len(data) ** (1 / 3))  # Freedman–Diaconis rule
    return bin_width

# Define a custom color palette for the super_class
custom_palette = {
    'visual_projection': '#D5A848',
    'central': '#F9574E',
    'optic': '#F4D826',
    'visual_centrifugal': '#44733B'
}
#%%
df=pd.read_feather(NEURON_TABLE_FTR)
#%%
df=df.dropna(subset=['dend_correct','axon_correct','super_class','primary_type'])
#%%
dfg=df.groupby(by=('super_class')).size()

#%%
df['axon_correct']=df['axon_correct']*100
df['dend_correct']=df['dend_correct']*100
#%%

#%%
allowed_values = ['optic', 'central', 'visual_projection','visual_centrifugal']

# Filter the DataFrame
df = df[df['super_class'].isin(allowed_values)]
#%%

#%%
# Define a custom color palette for the super_class
custom_palette = {
    'visual_projection': '#D5A848',
    'central': '#F9574E',
    'optic': '#F4D826',
    'visual_centrifugal': '#44733B'
}


#%%
import numpy as np
import matplotlib.pyplot as plt


plt.figure(figsize=(2.7,2 ))

# Compute and plot for 'axon_correct'
axon_data = df['axon_correct'].dropna()
axon_bin_width = freedman_diaconis(axon_data)
axon_bins = int((axon_data.max() - axon_data.min()) / axon_bin_width) if axon_bin_width > 0 else 1
axon_bin_edges = np.linspace(axon_data.min(), axon_data.max(), axon_bins + 1)
axon_densities, _ = np.histogram(axon_data, bins=axon_bin_edges, density=True)
axon_bin_centers = 0.5 * (axon_bin_edges[:-1] + axon_bin_edges[1:])

# Line plot and fill for 'axon_correct'
plt.plot(axon_bin_centers, axon_densities, color='#92278f', label='Axon',linewidth=0.7)
plt.fill_between(axon_bin_centers, axon_densities, color='#92278f', alpha=0.2)

# Compute and plot for 'dend_correct'
dend_data = df['dend_correct'].dropna()
dend_bin_width = freedman_diaconis(dend_data)
dend_bins = int((dend_data.max() - dend_data.min()) / dend_bin_width) if dend_bin_width > 0 else 1
dend_bin_edges = np.linspace(dend_data.min(), dend_data.max(), dend_bins + 1)
dend_densities, _ = np.histogram(dend_data, bins=dend_bin_edges, density=True)
dend_bin_centers = 0.5 * (dend_bin_edges[:-1] + dend_bin_edges[1:])

# Line plot and fill for 'dend_correct'
plt.plot(dend_bin_centers, dend_densities, color='#fbb042', label='Dendrite',linewidth=0.7)
plt.fill_between(dend_bin_centers, dend_densities, color='#fbb042', alpha=0.2)
plt.rc('axes', titlesize=8)         # Title size
plt.rc('axes', labelsize=8)         # X and Y label size
plt.rc('xtick', labelsize=8)        # X tick label size
plt.rc('ytick', labelsize=8)        # Y tick label size
plt.rc('legend', fontsize=8)        # Legend text size
plt.rc('legend', title_fontsize=8) 
# Add legend and labels
plt.xticks(np.arange(0, 120, 20))

plt.legend(title="Compartment")
plt.xlabel("Cannonical Percent")
plt.ylabel("Density")
plt.tight_layout()
sns.despine()
file_name = "Axon_Dend.svg"  
_out_dir = OUTPUT_DIR / "fig1" / "canonicality_axon_dend"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "axon_dendrite_cannonical_percent.svg")

plt.show()
#%%
# Percentiles for axon_data
axon_data.quantile([0.25, 0.5, 0.75])
#%%
# Percentiles for dend_data
dend_data.quantile([0.25, 0.5, 0.75])
#%%
df[['axon_correct','dend_correct']].corr()

#%%
from scipy.stats import levene

# Define the classes of interest
classes = ['central', 'visual_projection', 'optic', 'visual_centrifugal']

# Create groups for axon_correct
axon_groups = [df[df['super_class'] == c]['axon_correct'].dropna() for c in classes]

# Create groups for dend_correct
dend_groups = [df[df['super_class'] == c]['dend_correct'].dropna() for c in classes]

print([len(g) for g in axon_groups])  # sample sizes per class
print([len(g) for g in dend_groups])  # sample sizes per class
#%%
from scipy.stats import levene

# Levene test for axon_correct
stat_ax, p_ax = levene(*axon_groups)
print("Levene test (axon_correct):", stat_ax, p_ax)

# Levene test for dend_correct
stat_dend, p_dend = levene(*dend_groups)
print("Levene test (dend_correct):", stat_dend, p_dend)
#%%

#%%
