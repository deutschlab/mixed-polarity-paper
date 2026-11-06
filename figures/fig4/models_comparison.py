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
from methods.figure_text import use_arial_8
use_arial_8()   # every text in Arial 8 pt when a figure is saved
from config import NEURON_TABLE_FTR, SYNAPSE_TABLE_FTR, CONNECTIONS_TABLE_FTR, OUTPUT_DIR, METHODS_DIR, FIG4_HELDOUT_MODELS_FTR, FIG4_HELDOUT_CONFUSION_FTR
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

# The neurons model can only be scored on the synapses it was counted from: a neuron it never saw has no pairs to
# look up. As the other models, it is scored by balanced accuracy (the mean of the four types' recalls): each
# connection predicts its most common type for all its synapses.
TYPES = ['AA', 'AD', 'DA', 'DD']
_pred = connections[TYPES].idxmax(axis=1)
recall_per_neuron = {t: connections.loc[_pred == t, t].sum() / connections[t].sum() for t in TYPES}
balanced_per_neuron = float(np.mean(list(recall_per_neuron.values())))

# The other three models, on neurons they never saw (figures/fig4/fig4_heldout_models.py, five cell splits)
heldout = pd.read_feather(FIG4_HELDOUT_MODELS_FTR)
held = heldout.groupby('model')['balanced_accuracy'].agg(['mean', 'min', 'max'])

x = ['neurons', 'cell types', 'features', 'PC1']
y = [100 * balanced_per_neuron] + [100 * held.at[m, 'mean'] for m in ('cell_types', 'features', 'pc1')]
lo = [np.nan] + [100 * held.at[m, 'min'] for m in ('cell_types', 'features', 'pc1')]
hi = [np.nan] + [100 * held.at[m, 'max'] for m in ('cell_types', 'features', 'pc1')]

# The plot, at its size in the figure: neurons as an open circle (in-sample), set apart by a thin line; the other three
# filled, at the mean over the five splits, each with a line from the lowest to the highest split (the spread between
# splits, not a confidence interval); the y-axis starts at chance (25%)
fig, ax = plt.subplots(figsize=(1.7, 1.8))
ax.axvline(0.5, color='0.8', lw=0.5, zorder=0)
ax.errorbar(range(1, 4), y[1:], yerr=[np.subtract(y[1:], lo[1:]), np.subtract(hi[1:], y[1:])], fmt='none',
            ecolor='black', elinewidth=0.8, capsize=1.5, zorder=1)
ax.plot([0], y[:1], 'o', mfc='white', mec='black', mew=0.8, ms=4, zorder=2)
ax.plot(range(1, 4), y[1:], 'o', color='black', ms=3, zorder=2)
ax.set_xticks(range(4), x, rotation=30, ha='right')
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(25, 100)   # starts at chance
ax.set_yticks([25, 50, 75, 100])
ax.set_ylabel('Balanced accuracy (%)')
sns.despine(ax=ax, top=True, right=True)
plt.tight_layout()
_out_dir = OUTPUT_DIR / "fig4" / "models_comparison"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "accuracy_values.svg")
plt.show()
#%% 4A: the PC1 model's confusion matrix on neurons it never saw, over the five splits (figures/fig4/fig4_heldout_models.py)
LABELS = ['AD', 'AA', 'DD', 'DA']   # the published panel's order; rows and columns are taken by name
counts = pd.read_feather(FIG4_HELDOUT_CONFUSION_FTR).pivot(index='true', columns='predicted', values='synapses')
share = counts.loc[LABELS, LABELS].div(counts.loc[LABELS].sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(2.6, 2.6))
ax.imshow(share.to_numpy(), cmap='Blues', vmin=0, vmax=100)   # row percentages, so 100 is the ceiling
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{share.iat[i, j]:.1f}", ha='center', va='center',
                color='white' if share.iat[i, j] > 55 else 'black')
ax.set_xticks(range(4), LABELS)
ax.set_yticks(range(4), LABELS)
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
plt.tight_layout()
plt.savefig(_out_dir / "confusion_matrix_pc1_unseen_neurons.svg", dpi=300, bbox_inches='tight')
plt.show()
print('4A diagonal (recall per type, %):', {t: round(share.at[t, t], 1) for t in LABELS})

#%% n values for reporting

# majority label per directed neuron-neuron connection (the 4B 'neurons' dot)
n_synapses_connection_model = int(connections['sum_syn'].sum())
n_connections = len(connections)

# majority label per primary-type pair, counted and scored on the same synapses (not drawn; 4B's cell-type bar is
# the held-out model from fig4_heldout_models.py)
n_synapses_primary_type_model = int(connections_agg['sum_syn'].sum())
n_primary_type_pairs = len(connections_agg)


print('--- N values ---')
print(f'Neuron-pair / connection majority label:')
print(f'  n synapses = {n_synapses_connection_model:,}')
print(f'  n directed connections = {n_connections:,}')
print(f'  accuracy = {result_per_conneciton * 100:.1f}%; balanced accuracy = {balanced_per_neuron * 100:.1f}% (the dot)')

print(f'\nPrimary-type pair majority label (not drawn in 4B):')
print(f'  n synapses = {n_synapses_primary_type_model:,}')
print(f'  n primary-type pairs = {n_primary_type_pairs:,}')
print(f'  accuracy = {result_per_conneciton_primary_type * 100:.1f}% (counted and scored on the same synapses)')

print('\nOn neurons the models never saw (balanced accuracy over five cell splits: mean, lowest, highest):')
for m in ('cell_types', 'features', 'pc1', 'pc1_superclass'):
    r = heldout[heldout.model == m]
    print(f'  {m}: {100 * held.at[m, "mean"]:.1f}% ({100 * held.at[m, "min"]:.1f} to {100 * held.at[m, "max"]:.1f}); '
          f'plain accuracy {100 * r.accuracy.mean():.1f}%')
r = heldout[heldout.model == 'pc1']
print(f'  per split: test synapses {int(r.n_test_synapses.min()):,} to {int(r.n_test_synapses.max()):,}, test cells '
      f'{int(r.n_test_cells.min()):,} to {int(r.n_test_cells.max()):,}, their cell types {int(r.n_test_cell_types.min()):,} '
      f'to {int(r.n_test_cell_types.max()):,}; with a cell type pair seen in training {r.share_with_type_pair.min():.3f} '
      f'to {r.share_with_type_pair.max():.3f}; with a PC1 for both cells {r.share_with_pc1.min():.4f} to '
      f'{r.share_with_pc1.max():.4f}')
print(f'4A pools the five splits ({int(counts.to_numpy().sum()):,} test synapses); 4B shows the mean of the splits')
