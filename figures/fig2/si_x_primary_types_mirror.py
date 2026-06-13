
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

import matplotlib.patches as mpatches

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import METHODS_DIR, NEURON_TABLE_FTR, OUTPUT_DIR, SI_UPDATED_FTR, NEURON_ANNOTATIONS_CSV
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import os
import seaborn as sns
import pickle

nodesG=pd.read_feather(NEURON_TABLE_FTR)
#%%
nodesG=nodesG.drop(columns=['SI'])
all_SI_df=pd.read_feather(SI_UPDATED_FTR)
all_SI_df=all_SI_df.rename(columns={'root_id':'neuron'})
nodesG=nodesG.merge(all_SI_df,on='neuron',how='left').dropna(subset='SI')#%%
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
nodesG=nodesG.dropna(subset=['dend_correct','axon_correct','super_class','primary_type'])
#%%
nodesG[nodesG['neuron'].isin([720575940628378378,720575940612792830])]['SI']

#%%720575940637687605,720575940615226086
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
nodesG=nodesG.dropna(subset=['dend_correct','axon_correct','super_class','primary_type'])
#%%
nodesGog=pd.read_csv(NEURON_ANNOTATIONS_CSV)

nodesGog=nodesGog.rename(columns={'root_id':'neuron'})

nodesG=nodesG.merge(nodesGog[['neuron','side']],on='neuron',how='left')

       #%%
nodesG=nodesG[nodesG['side']!='center']
#%%
nodesG=nodesG.dropna(subset='primary_type')
#%%
nodesG['p_type_count']=nodesG.groupby(by='primary_type')['primary_type'].transform('count')
#%%

ptypes=nodesG.groupby(by=['primary_type'])
#%%
two_p_types=nodesG[nodesG['p_type_count']==2]

two_p_types=two_p_types[[ 'SI', 'primary_type',
       'side']]


#%%

pivot_df=pd.pivot_table(two_p_types,values='SI',index='primary_type',columns='side')
clean_df = pivot_df.dropna(subset=['left', 'right'])

#%%

custom_palette = {
    'ascending': '#6EB6F6',
    'visual_centrifugal': '#44733B',
    'descending': '#803D3D',
    'endocrine': '#8973B2',
    'motor': '#B48667',
    'optic': '#F4D826',
    'sensory': '#848484',
    'central': '#F9574E',
    'visual_projection': '#D5A848'
}
clean_df = pivot_df.dropna(subset=['left', 'right']).reset_index()

# Merge 'super_class' from nodesG based on 'primary_type'
super_class_map = nodesG[['primary_type', 'super_class']].drop_duplicates()
clean_df = clean_df.merge(super_class_map, on='primary_type', how='left')



#%%
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
clean_df=clean_df[clean_df['super_class'].isin(['visual_centrifugal','optic','central','visual_projection'])]
#%%

# Calculate Pearson R
r_value, p_value = pearsonr(clean_df['left'], clean_df['right'])

# Scatter plot with hue and your palette
plt.figure(figsize=(2, 2))
sns.scatterplot(
    data=clean_df,
    x='left',
    y='right',
    hue='super_class',  # replace with your actual column name
    palette=custom_palette,  # use your existing palette
    s=2  # adjust point size if you want
)
plt.text(0.98, 0.97, f'n ={len(clean_df)}', 
         fontsize=8, color='black', ha='right')
# Add identity line x = y
min_val = min(clean_df['left'].min(), clean_df['right'].min())
max_val = max(clean_df['left'].max(), clean_df['right'].max())
plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1, label='x = y')

# Annotate Pearson R
plt.text(0.05, 0.95, f'R = {r_value:.2f}', fontsize=8, transform=plt.gca().transAxes)
plt.xticks([0,0.2,0.4,0.6,0.8,1],size=4)
plt.yticks([0,0.2,0.4,0.6,0.8,1],size=4)

# Labels and style
plt.xlabel('SI Left', size=8)
plt.ylabel('SI Right', size=8)
plt.xticks(size=8)
plt.yticks(size=8)
plt.title('SI Left vs Right', size=8)
plt.legend(title='Super Class', bbox_to_anchor=(0.5, 1.3), loc='center', ncol=3)
sns.despine(right=True, top=True)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()

# Save and show
_out_dir = OUTPUT_DIR / "fig2" / "SI_x_primary_types_mirror"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "SI_corr_sides_princeton_v2.svg")
plt.show()
#%%
pivot_df['dif']=abs(pivot_df['left']-pivot_df['right'])
#%%
from scipy.stats import ttest_rel
import numpy as np

# Drop NaNs to ensure pairs are valid
paired_df = clean_df.dropna(subset=['left', 'right'])

# Run paired t-test
t_stat, p_val = ttest_rel(paired_df['left'], paired_df['right'])
mean_diff = np.mean(paired_df['left'] - paired_df['right'])

print(f"Paired t-test:")
print(f"  t = {t_stat:.3f}")
print(f"  p = {p_val:.3e}")
print(f"  Mean difference (Left - Right) = {mean_diff:.4f}")

from scipy.stats import wilcoxon

w_stat, p_wilcoxon = wilcoxon(paired_df['left'], paired_df['right'])
print(f"Wilcoxon signed-rank test: W = {w_stat}, p = {p_wilcoxon:.3e}")
import numpy as np

# Compute Cohen’s d for paired samples
mean_diff = clean_df['left'].mean() - clean_df['right'].mean()
sd_diff = np.std(clean_df['left'] - clean_df['right'], ddof=1)
cohen_d = mean_diff / sd_diff
print(f"Cohen's d (paired) = {cohen_d:.3f}")

#%%
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Initialize threshold range (SI from 0.05 to 0.6 with small increments)
si_thresholds = np.arange(0.05, 0.65, 0.05)

# Prepare a list to store results
results = []

# Iterate through thresholds
for threshold in si_thresholds:
    # Subset for SI below the threshold
    subset_low = clean_df[clean_df['left'] < threshold]
    # Subset for SI equal to or above the threshold
    subset_high = clean_df[clean_df['left'] >= threshold]
    
    # Calculate correlations for both subsets
    if len(subset_low) > 1:
        r_low, _ = pearsonr(subset_low['left'], subset_low['right'])
    else:
        r_low = np.nan  # Not enough data for correlation
    
    if len(subset_high) > 1:
        r_high, _ = pearsonr(subset_high['left'], subset_high['right'])
    else:
        r_high = np.nan  # Not enough data for correlation
    
    # Store the results
    results.append({
        'SI_threshold': threshold,
        'r_low (SI < threshold)': r_low,
        'r_high (SI >= threshold)': r_high
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Display results
print(results_df)
#%%


