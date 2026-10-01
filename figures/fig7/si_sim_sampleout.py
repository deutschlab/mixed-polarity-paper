# Fig 7J input, sampleout condition: SI recomputed per neuron with control: as many non-filopodium incoming synapses removed at random as there are filopodium ones.
# Ported from Amit's Fig 7/filapodia_extraction/SI_sim/si_sim_sampleout.py.
# Changes: paths via config.py; unused imports removed (nglui, fafbseg, navis, networkx, sklearn,
# ng_methods_v2, Amit's colour module, the CAVEclient connection and others). SI_calc came from
# altsi_methods_v2 in Amit's setup; the copy in methods/methods_all.py is identical.
# Amit's copy read "syn_bouton_filopodia(new).ftr"; SYN_BOUTON_FTR names the file without "(new)".
# Writes filopodia_nodes_sampleout (FILOPODIA_NODES_SAMPLEOUT_FTR), which figures/fig7/si_comb_analysis.py reads.
# The compartments are not re-split: SI is recomputed on the existing A/D labels.
# Also added: svg.fonttype='none' and Arial (repository convention). Amit's saved SVGs drew text as
# DejaVu Sans glyph paths, so text and spacing differ from his files; plotted values do not.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    METHODS_DIR, NEURON_TABLE_FTR, SYNAPSE_TABLE_FTR, SYN_BOUTON_FTR, FILOPODIA_NODES_SAMPLEOUT_FTR, OUTPUT_DIR,
)
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

_out_dir = OUTPUT_DIR / "fig7" / "si_sim_sampleout"
_out_dir.mkdir(parents=True, exist_ok=True)
FILOPODIA_NODES_SAMPLEOUT_FTR.parent.mkdir(parents=True, exist_ok=True)
#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
#%%
nodesG['SI'].mean()
#%%
nodesG['SI'].median()

#%%

df=pd.read_feather(SYN_BOUTON_FTR)
#%%
#%%
allsynapses=pd.read_feather(SYNAPSE_TABLE_FTR)
#%%
allsynapses=allsynapses[['synapse_id', 'pre', 'post', 'comp']]
#%%
allsynapses=allsynapses[allsynapses['comp'].isin(['AA','AD','DA','DD'])]
#%%

allsynapses=allsynapses.merge(df,on='synapse_id',how='left')
#%%
del df
#%%
allsynapses['pre_comp']=allsynapses['comp'].str[0]
allsynapses['post_comp']=allsynapses['comp'].str[1]

#%%
def extract_comp_syntype(allsynapses_pre,allsynapses_post):
    
    
    neuron_as_pre=allsynapses_pre.groupby(by=['pre','pre_comp']).size().reset_index()
    neuron_as_post=allsynapses_post.groupby(by=['post','post_comp']).size().reset_index()
    neuron_as_pre.columns=['neuron','comp_on_pre','count_pre']
    neuron_as_post.columns=['neuron','comp_on_post','count_post']
    
    neuron_as_pre_piv=pd.pivot_table(neuron_as_pre,columns='comp_on_pre',index='neuron',values='count_pre').fillna(0)
    neuron_as_post_piv=pd.pivot_table(neuron_as_post,columns='comp_on_post',index='neuron',values='count_post').fillna(0)
    
    neuron_as_pre_piv.columns=['pre_synapses_on_axon','pre_synapses_on_dend']
    neuron_as_post_piv.columns=['post_synapses_on_axon','post_synapses_on_dend']
    
    
    comb_df = neuron_as_pre_piv.merge(
        neuron_as_post_piv,
        left_index=True,
        right_index=True,
        how="outer"
    ).fillna(0)
    
    
    return comb_df

def calculate_SI(df):
    
    cols = [
        "pre_synapses_on_axon", "post_synapses_on_axon",
        "pre_synapses_on_dend", "post_synapses_on_dend"
    ]
    
    
    
    
    comp_syntype = df.apply(
        lambda x: SI_calc([
            "",
            (x["pre_synapses_on_axon"],  x["post_synapses_on_axon"]),
            (x["pre_synapses_on_dend"],  x["post_synapses_on_dend"])
        ])[0],
        axis=1
    )
    
    return comp_syntype.reset_index()
#%% no filopodia

filopodia_sum=allsynapses['filopodia_on_post'].sum()
total_post=len(allsynapses)
#%%   baseline1
'''
baseline_comp_syntype=extract_comp_syntype(allsynapses,allsynapses)
baseline_SI=calculate_SI(baseline_comp_syntype)
'''
#%%   baseline2

synapses_to_remove= (
    allsynapses[allsynapses["filopodia_on_post"] == 0]
    .sample(n=filopodia_sum, random_state=0)
    .copy()['synapse_id']
)

allsynapses_sampleout = allsynapses.loc[
    ~allsynapses["synapse_id"].isin(synapses_to_remove)
].copy()

#%%
sampleout_comp_syntype=extract_comp_syntype(allsynapses,allsynapses_sampleout)
sampleout_SI=calculate_SI(sampleout_comp_syntype)


#%%
sampleout_SI.columns=['neuron','sampleout_SI']

#%%
del allsynapses
#%%
#%%

allsynapses_sampleout=allsynapses_sampleout.merge(sampleout_SI,left_on='pre',right_on='neuron',how='left')
allsynapses_sampleout=allsynapses_sampleout.merge(sampleout_SI,left_on='post',right_on='neuron',how='left')
allsynapses_sampleout=allsynapses_sampleout[['synapse_id', 'pre', 'post', 'comp', 'sampleout_SI_x', 'sampleout_SI_y']]
allsynapses_sampleout=allsynapses_sampleout.rename(columns={'sampleout_SI_x':'sampleout_SI_pre','sampleout_SI_y':'sampleout_SI_post'})

#%%
import matplotlib.pyplot as plt

def pie_comp(df, title):
    counts = df["comp"].value_counts(dropna=False)

    plt.figure(figsize=(5.5, 5.5))
    plt.pie(
        counts.values,
        labels=counts.index.astype(str),
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(_out_dir / "sampleout_pie.svg")
    plt.show()
#%%
pie_comp(allsynapses_sampleout.query('sampleout_SI_pre>=0.1 and sampleout_SI_post>=0.1'), "sampleout synapses, SI>=0.1")


#%%correct percentages

sampleout_comp_syntype['axon_correct']=sampleout_comp_syntype['pre_synapses_on_axon']/(sampleout_comp_syntype['pre_synapses_on_axon']+sampleout_comp_syntype['post_synapses_on_axon'])
sampleout_comp_syntype['dend_correct']=sampleout_comp_syntype['post_synapses_on_dend']/(sampleout_comp_syntype['pre_synapses_on_dend']+sampleout_comp_syntype['post_synapses_on_dend'])

#%%
sampleout_SI=sampleout_SI.merge(sampleout_comp_syntype.reset_index()[['neuron','axon_correct',
                                                                   'dend_correct']],on='neuron')
#%%
# sort baseline
si_col_nofilopodia = sampleout_SI.columns[1]
nofilopodia_SI_sorted = sampleout_SI.sort_values(si_col_nofilopodia, ascending=True).reset_index(drop=True)


#%%import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

custom_palette = {
    'AA': '#8B2BE2',
    'DD': '#F4B95A',
    'AD': '#9F4800',
    'DA': '#B3B3B3'
}

def plot_roll_ax_dend(df_sorted, si_col, title="", window=500, x_thr=0.1):
    d = df_sorted[[si_col, "axon_correct", "dend_correct"]].dropna().copy()

    d["axon_roll"] = d["axon_correct"].rolling(window=window, center=True, min_periods=10).mean()
    d["dend_roll"] = d["dend_correct"].rolling(window=window, center=True, min_periods=10).mean()

    idx = (d[si_col] - x_thr).abs().idxmin()
    ax_val = float(d.loc[idx, "axon_roll"])
    de_val = float(d.loc[idx, "dend_roll"])

    plt.figure(figsize=(8, 5))
    plt.plot(d[si_col], d["axon_roll"], linewidth=2, label="axon_correct (rolling mean)", color=custom_palette["AA"])
    plt.plot(d[si_col], d["dend_roll"], linewidth=2, label="dend_correct (rolling mean)", color=custom_palette["DD"])

    plt.axvline(x_thr, linestyle="--", linewidth=1.5, color="black")
    plt.text(x_thr, 0.98, f"axon={ax_val:.3f}\ndend={de_val:.3f}", ha="left", va="top", fontsize=10)

    plt.xlabel("SI")
    plt.ylabel("Correct (0–1)")
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(_out_dir / "comp_correct_SI_sampleout.svg")

    plt.show()

    return ax_val*100, de_val*100


axon_b, dend_b = plot_roll_ax_dend(nofilopodia_SI_sorted, si_col_nofilopodia, title="Baseline", window=400, x_thr=0.1)

print("no_filopodia:", axon_b, dend_b)
#%%
nodesG_baseline=sampleout_SI.copy()
threshold = axon_b
# Create a new column based on comparison
nodesG_baseline['axon_label'] = nodesG_baseline['axon_correct'].apply(lambda x: 'A' if x*100 > threshold else 'M')

# Verify result
print(nodesG_baseline['axon_label'].value_counts())


# Use the previously computed axon_val threshold from SI=0.1
threshold = dend_b  # from earlier step

# Create a new column based on comparison
nodesG_baseline['dend_label'] = nodesG_baseline['dend_correct'].apply(lambda x: 'D' if x*100 > threshold else 'M')

#%%
nodesG_baseline.to_feather(FILOPODIA_NODES_SAMPLEOUT_FTR)
#%%baseline
