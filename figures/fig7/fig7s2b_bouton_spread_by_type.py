# Supp 7-S2 B: the same per cell type, coloured by why a type was excluded from the bouton analysis.
# Ported from the authors' original script fig7S2b.py.
# Changes: paths via config.py; figure saves switched on and placed before plt.show().
# Fix: the cell-type table names its id column root_id; renamed to neuron so the merge runs (the original stopped with a KeyError).
# Not changed: the excluded-type lists write 'R1–6' with an en dash, while the table has 'R1-6', so R1-6 is coloured 'included'.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BOUTON_INERTIA_FTR, CONSOLIDATED_CELL_TYPES_CSV, NEURON_DATA_FTR, OUTPUT_DIR

_out_dir = OUTPUT_DIR / "fig7" / "fig7s2b_bouton_spread_by_type"
_out_dir.mkdir(parents=True, exist_ok=True)

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
neuron_quality=pd.read_feather(BOUTON_INERTIA_FTR)
neuron_types = pd.read_csv(CONSOLIDATED_CELL_TYPES_CSV)
neuron_types = neuron_types.rename(columns={'root_id': 'neuron'})
neuron_data = pd.read_feather(NEURON_DATA_FTR) 
neuron_data = neuron_data.rename(columns={'root_id':'neuron'}) 
#%%% 
neuron_data=neuron_data[["neuron","sub_class"]] 
neuron_data.dtypes

neuron_subclass = neuron_data.set_index('neuron')['sub_class']

plot_df = neuron_quality.merge(
    neuron_types[['neuron', 'primary_type']],
    on='neuron',
    how='left'
)

plot_df = plot_df.merge(
    neuron_subclass,
    on='neuron',
    how='left'
)

#%%

type_stats = plot_df.groupby('primary_type').agg(
    mean_rms=('mean_rms', 'mean'),
    mean_centroid_dist=('mean_centroid_dist', 'mean'),
    n_neurons=('neuron', 'nunique'),
    uniglomerular_fraction=('sub_class', lambda x: (x == 'uniglomerular').mean())
).reset_index()

exclude_types = {
    'L1','L2','L3','L4','L5',
    'Mi1','Mi4','Mi9',
    'T1','T2','T2a','T3',
    'R1–6','R7','R8',
    'Tm1','Tm2','Tm3','Tm4',
    'Tm5a','Tm5f','Tm6','Tm9','Tm21',
    'LCNOp','LCNOpm',
    'AN_AVLP_10','AN_AVLP_18','AN_AVLP_29',
    'AN_AVLP_40','AN_AVLP_45','AN_AVLP_6',
    'AN_GNG_55','AN_GNG_VES_5',
    'AN_multi_2',
    'CB0268','CB0419',
    'Dm8',
    'DNa16',
    'DNge175',
    'GLNO',
    'JO-A','JO-E',
    'Lai',
    'LTe26',
    'MTe44',
    'PLP163',
    'PS083a','PS280',
    'PVLP061','PVLP141',
    'SA_VTV_6',
    'VP3+_vPN',
    'VS8'
}

type_stats['excluded'] = (
    type_stats['primary_type'].isin(exclude_types)
    |
    type_stats['primary_type'].str.startswith('TuBu', na=False)
    |
    type_stats['primary_type'].str.startswith('T4', na=False)
    |
    type_stats['primary_type'].str.startswith('T5', na=False)
    |
    (type_stats['uniglomerular_fraction'] == 1.0)
)




#%%

# Types excluded specifically due to high RMS
rms_excluded_types = {
    'AN_AVLP_10','AN_AVLP_18','AN_AVLP_29',
    'AN_AVLP_40','AN_AVLP_45','AN_AVLP_6',
    'AN_GNG_55','AN_GNG_VES_5',
    'AN_multi_2',
    'CB0268','CB0419',
    'Dm8',
    'DNa16',
    'DNge175',
    'GLNO',
    'JO-A','JO-E',
    'Lai',
    'LTe26',
    'Mi9',
    'MTe44',
    'PLP163',
    'PS083a','PS280',
    'PVLP061','PVLP141',
    'SA_VTV_6',
    'VP3+_vPN',
    'VS8',
    'T2','T2a','T3'
}

# Other excluded types
other_excluded_types = {
    'L1','L2','L3','L4','L5',
    'Mi1','Mi4',
    'T1',
    'R1–6','R7','R8',
    'Tm1','Tm2','Tm3','Tm4',
    'Tm5a','Tm5f','Tm6','Tm9','Tm21',
    'LCNOp','LCNOpm'
}

#%%

def assign_group(row):

    pt = row['primary_type']
    uniglom = row['uniglomerular_fraction']

    # uniglomerular
    if uniglom == 1.0:
        return 'uniglomerular'

    # excluded due to RMS
    if (
        pt in rms_excluded_types
        or str(pt).startswith('T4')
        or str(pt).startswith('T5')
    ):
        return 'rms_excluded'

    # other excluded
    if (
        pt in other_excluded_types
        or str(pt).startswith('TuBu')
    ):
        return 'other_excluded'

    # included
    return 'included'


type_stats['group'] = type_stats.apply(assign_group, axis=1)

#%%

color_map = {
    'included': 'steelblue',
    'uniglomerular': 'darkgoldenrod',
    'rms_excluded': 'purple',
    'other_excluded': 'red'
}

fig, ax = plt.subplots(figsize=(6,6))

for group_name, group_df in type_stats.groupby('group'):

    ax.scatter(
        group_df['mean_rms'],
        group_df['mean_centroid_dist'],
        color=color_map[group_name],
        alpha=0.85,
        label=group_name.replace('_', ' ')
    )

ax.set_xlabel('Mean RMS')
ax.set_ylabel('Mean Centroid Distance (Separation)')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(frameon=False)

plt.savefig(_out_dir / 'inertia_grouped.svg', format='svg')
plt.show()
#%%