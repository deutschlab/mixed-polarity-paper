import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import time
import datetime
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SYNAPSE_NON_PROCESSED_FTR, PROCESSED_SWC_DIR,
    PROCESSED_BIG_NEURONS_DIR, SYNAPSE_TABLE_SFC_FTR as SYNAPSE_TABLE_FTR,   # the SFC table, whatever SPLIT_METHOD says
    DERIVED_DATA_DIR,
)
import networkx as nx

from sklearn.cluster import AgglomerativeClustering
import random
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pickle
#%%
# The step 03 table holds every synapse of the CSV (about 77 million). It is read here piece by piece, and each piece
# gets the same treatment the whole table used to get, in the same order. First the SI of every cell, unchanged:
#%%add_si!
import os
import pickle
# to make sure u take the right iteraiton! Majd
#base_dir = r'C:\Users\user\organised_work\data\783\generated\783\trees\all_trees_princeton'
base_dir = PROCESSED_SWC_DIR

all_si_list = []

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):
        pkl_path = os.path.join(folder_path, 'all_SI.pkl')
        if os.path.exists(pkl_path):
            SI = pd.read_pickle(pkl_path)
            si_df = pd.DataFrame(SI)
            si_df.columns = ['nid', 'SI']
            all_si_list.append(si_df)
#%%
SI_df = pd.concat(all_si_list, ignore_index=True)
#%%big n afddition *******************
dfs = []
base_dir = PROCESSED_BIG_NEURONS_DIR

for fname in os.listdir(base_dir):
    if fname.startswith("all_SI_"):
        path = os.path.join(base_dir, fname)

        obj = pd.read_pickle(path)  # this is a list
        sid = fname.split("_", 1)[1].split(".", 1)[0]

        for v in obj:
            dfs.append({"source_id": sid, "value": v})

all_SI_df = pd.DataFrame(dfs)
#%%
out = pd.DataFrame({
    "nid": all_SI_df["value"].apply(lambda x: int(x[0])),
    "SI":  all_SI_df["value"].apply(lambda x: float(x[1]))
})
#%%


SI_df=pd.concat([SI_df,out])
#%%
SI_df['nid']=SI_df['nid'].astype(np.int64)
#%%
# Each cell has one SI, so a left merge on the cell ID is a lookup: every synapse keeps its row and order and gets the
# SI of its cell, or NaN when the cell has none.
assert SI_df['nid'].is_unique
SI_lookup=pd.Series(SI_df['SI'].to_numpy(), index=SI_df['nid'].to_numpy())

#%%
# The checks that were printed for the whole table, over the synapses with both labels. Each reads only the columns
# it needs, one at a time, so the whole table is never in memory. (pandas never counts a value as equal to None, so the
# None total was always 0 in pandas 2.2.2; and synapse_id, the CSV row number, is unique, so no row repeats as a whole.)
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

labels=feather.read_table(SYNAPSE_NON_PROCESSED_FTR, columns=['pre_compartment','post_compartment'])
labelled=pc.and_(pc.is_valid(labels['pre_compartment']), pc.is_valid(labels['post_compartment']))
del labels
with pa.ipc.open_file(SYNAPSE_NON_PROCESSED_FTR) as f:
    columns=f.schema.names
null_counts={}
for c in columns:
    col=feather.read_table(SYNAPSE_NON_PROCESSED_FTR, columns=[c])[c]
    null_counts[c]=pc.sum(pc.and_(pc.is_null(col), labelled)).as_py() or 0
    del col
null_counts=pd.Series(null_counts)
print(null_counts)
print("Total None values in DataFrame:", 0)
ids=feather.read_table(SYNAPSE_NON_PROCESSED_FTR, columns=['synapse_id'])['synapse_id']
assert pc.count_distinct(ids).as_py() == len(ids)
del ids
print("Number of duplicate rows:", 0)
for subset in (['pre','post','pre_x','pre_y','pre_z'], ['pre','post','post_x','post_y','post_z']):
    part=feather.read_table(SYNAPSE_NON_PROCESSED_FTR, columns=subset).filter(labelled)
    dup_count = part.num_rows - part.group_by(subset).aggregate([]).num_rows
    print("Number of duplicates based on these columns:", dup_count)
    del part
del labelled

#%%
# The synapse table, piece by piece: drop the synapses missing either label, add each end's SI, drop every row with
# an empty value, make comp, keep and rename the columns, and add the piece to SYNAPSE_TABLE_FTR, a feather (Arrow IPC)
# file written under a temporary name and renamed at the end. (syn_df and syn_df_SI2 are one piece, not the table.)
DERIVED_DATA_DIR.mkdir(parents=True, exist_ok=True)
writer=None
try:
    with pa.ipc.open_file(SYNAPSE_NON_PROCESSED_FTR) as reader:
        for i in range(reader.num_record_batches):
            syn_df=reader.get_batch(i).to_pandas()
            syn_df.dropna(subset=['post_compartment','pre_compartment'],inplace=True)
            syn_df['SI_x']=syn_df['pre'].map(SI_lookup)
            syn_df['SI_y']=syn_df['post'].map(SI_lookup)
            syn_df_SI2=syn_df.dropna()
            if len(syn_df_SI2) == 0:
                continue
            syn_df_SI2=syn_df_SI2.assign(comp=syn_df_SI2['pre_compartment']+syn_df_SI2['post_compartment'])
            syn_df_SI2=syn_df_SI2[['synapse_id','pre', 'post',
                   'post_x', 'post_y', 'post_z', 'pre_x', 'pre_y', 'pre_z',
                   'neuropil', 'comp','SI_x','SI_y']]
            syn_df_SI2.columns=['synapse_id', 'pre', 'post',
                   'post_x', 'post_y', 'post_z', 'pre_x', 'pre_y', 'pre_z',
                   'npil', 'comp','SI_pre','SI_post']
            syn_df_SI2=syn_df_SI2.reset_index(drop=True)
            if writer is None:
                table=pa.Table.from_pandas(syn_df_SI2, preserve_index=False)
                schema=table.schema
                writer=pa.ipc.new_file(str(SYNAPSE_TABLE_FTR) + '.tmp', schema,
                                       options=pa.ipc.IpcWriteOptions(compression='lz4'))
            else:
                table=pa.Table.from_pandas(syn_df_SI2, schema=schema, preserve_index=False)
            writer.write_table(table, max_chunksize=65536)
finally:
    if writer is not None:
        writer.close()
# in the same cell as the loop, so a run that stopped above never renames a file that lacks some pieces
assert writer is not None, "no synapse has both labels, an SI at both ends and a neuropil"
os.replace(str(SYNAPSE_TABLE_FTR) + '.tmp', SYNAPSE_TABLE_FTR)
#%%
