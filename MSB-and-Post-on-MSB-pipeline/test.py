#%%
import os

import pandas as pd
import numpy as np

#%%
allsynapses=pd.read_parquet(r"C:\Users\majd_\Desktop\CS\5)Project\filopodia\princeton_py\synapses.parquet")

#%%
allsynapses.dtypes
# %%
# keep in allsynapses only the rows where pre is one of the names of the swc files in the folder C:\Users\majd_\Desktop\mixed-polarity-paper\MSB-and-Post-on-MSB-pipeline\swc
swc_folder = r"C:\Users\majd_\Desktop\mixed-polarity-paper\MSB-and-Post-on-MSB-pipeline\swc"
swc_files = [f for f in os.listdir(swc_folder) if f.endswith('.swc')]
swc_ids = [int(f.split(".")[0]) for f in swc_files]
#%%
synapses = allsynapses[allsynapses['pre'].isin(swc_ids) | allsynapses['post'].isin(swc_ids)]


# %%

neuron_types = pd.read_csv(r"C:\Users\majd_\Desktop\CS\5)Project\consolidated_cell_types.csv")

#%%
# all the above types are excluded from the analysis. I will create a new column in the neuron_types dataframe called 'excluded' which is True if the primary_type is in the exclude_types list which is the above commented lines
exclude_types = [
    "AN_AVLP_6",
    "AN_AVLP_10",
    "AN_AVLP_18",
    "AN_AVLP_29",
    "AN_AVLP_40",
    "AN_AVLP_45",
    "AN_GNG_55",
    "AN_GNG_VES_5",
    "AN_multi_2",
    "CB0268",
    "CB0419",
    "D_adPN",
    "DA1_lPN",
    "DA2_lPN",
    "DA3_adPN",
    "DA4l_adPN",
    "DA4m_adPN",
    "DC1_adPN",
    "DC2_adPN",
    "DC3_adPN",
    "DC4_adPN",
    "DL1_adPN",
    "DL2d_adPN",
    "DL2v_adPN",
    "DL3_lPN",
    "DL4_adPN",
    "DL5_adPN",
    "DM1_lPN",
    "DM2_lPN",
    "DM3_adPN",
    "DM4_adPN",
    "DM5_lPN",
    "DM6_adPN",
    "Dm8",
    "DNa16",
    "DP1l_adPN",
    "DP1m_adPN",
    "GLNO",
    "JO-A",
    "JO-E",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "Lai",
    "LCNOp",
    "LCNOpm",
    "LTe26",
    "Mi1",
    "Mi4",
    "Mi9",
    "MTe44",
    "PLP163",
    "PS083a",
    "PS280",
    "PVLP061",
    "PVLP141",
    "R1-6",
    "R7",
    "R8",
    "SA_VTV_6",
    "T1",
    "T2",
    "T2a",
    "T3",
    "T4a",
    "T4b",
    "T4c",
    "T4d",
    "T5a",
    "T5b",
    "T5c",
    "T5d",
    "Tm1",
    "Tm2",
    "Tm21",
    "Tm3",
    "Tm4",
    "Tm5a",
    "Tm5f",
    "Tm6",
    "Tm9",
    "TuBu01a",
    "TuBu01b",
    "TuBu02",
    "TuBu03",
    "TuBu06a",
    "TuBu06b",
    "TuBu07",
    "TuBu08",
    "TuBu09",
    "V_ilPN",
    "V_l2PN",
    "VA1d_adPN",
    "VA1v_adPN",
    "VA2_adPN",
    "VA3_adPN",
    "VA4_lPN",
    "VA5_lPN",
    "VA6_adPN",
    "VA7l_adPN",
    "VA7m_lPN",
    "VC1_lPN",
    "VC2_lPN",
    "VC3_adPN",
    "VC4_adPN",
    "VC5_lvPN",
    "VL1_ilPN",
    "VL2a_adPN",
    "VL2p_adPN",
    "VM1_lPN",
    "VM2_adPN",
    "VM3_adPN",
    "VM4_adPN",
    "VM4_lvPN",
    "VM5d_adPN",
    "VM5v_adPN",
    "VM6_adPN",
    "VM7d_adPN",
    "VM7v_adPN",
    "VP1d_il2PN",
    "VP1m_l2PN",
    "VP2_adPN",
    "VP2_l2PN",
    "VP3+_vPN",
    "VS8"
]
# %%
# Keep in synapses only the rows where pre is not in the exclude_types list
synapses2 = synapses[~synapses['pre'].isin(neuron_types[neuron_types['primary_type'].isin(exclude_types)]['neuron'].values)]

#%% now keep in synapses 2 only the rows where pre appears the most (most frequent pre neuron) and keep only the rows where pre is in the top 10 most frequent pre neurons
pre_freq = synapses2['pre'].value_counts().head(10)
synapses3 = synapses2[synapses2['pre'].isin(pre_freq.index)]
#%%
#how much times each neuron appears in synapses3 as pre
pre_freq2 = synapses3['pre'].value_counts()
# %%
synapses3.to_feather(r"C:\Users\majd_\Desktop\mixed-polarity-paper\MSB-and-Post-on-MSB-pipeline\sample_synapses.ftr")
# %%
