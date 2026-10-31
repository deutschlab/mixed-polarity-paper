

x=5
y=10
z=x+y

list_=[x,y,z]
#%%
#%%

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
    PROCESSED_SWC_DIR, CONNECTORS_DIR,
    PRE_CONNECTORS_FTR, POST_CONNECTORS_FTR,
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
#allsynapses=pd.read_feather(r'C:\Users\user\organised_work\data\connections\synapses')

#%%

#%%
# Define the base directories
input_base_dir = PROCESSED_SWC_DIR
#output_base_dir = r"C:\Users\user\organised_work\data\783\generated\783\trees\all_trees"


# Get the list of subdirectories in the input directory, in a fixed order (a set's order changes from run to run)
input_subfolders = sorted(f.name for f in os.scandir(input_base_dir) if f.is_dir())
assert input_subfolders, f"no folders in {input_base_dir}"

#%%
# Each folder's connectors are split into their pre and post rows and appended to the two output files one folder at
# a time, so step 01's connectors are never all in memory together. The files are feather (Arrow IPC) files with lz4
# compression, like to_feather writes; each takes the column types of its first folder, and a folder whose columns do
# not convert to them stops the run. Both are written under a temporary name (.tmp) and renamed only once every folder
# is in, so a run that stops part-way leaves the files of an earlier run as they were.
import pyarrow as pa

pre_cols=['pre_x','pre_y','pre_z','post','type','pre_compartment','pre','synapse_id','size']
post_cols=['post_x','post_y','post_z','pre','type','post_compartment','post','synapse_id','size']
CONNECTORS_DIR.mkdir(parents=True, exist_ok=True)
writers = {}
schemas = {}

try:
    for folder_name in input_subfolders:
        folder_path = os.path.join(input_base_dir, folder_name)

        connectors_data_file_path = os.path.join(folder_path, 'connectors.pkl')

        with open(connectors_data_file_path, 'rb') as f:
            connectors_data = pickle.load(f)
        print(len(connectors_data))
        connectors_data=connectors_data.drop(columns=['connector_id','node_id'])

        for side, cols in [('pre', pre_cols), ('post', post_cols)]:
            part=connectors_data[connectors_data['type']==side]
            part.columns=cols
            part=part.drop(columns=['type']).reset_index(drop=True)
            if side not in writers:
                table=pa.Table.from_pandas(part, preserve_index=False)
                schemas[side]=table.schema
                writers[side]=pa.ipc.new_file(str(PRE_CONNECTORS_FTR if side == 'pre' else POST_CONNECTORS_FTR) + '.tmp',
                                              table.schema, options=pa.ipc.IpcWriteOptions(compression='lz4'))
            else:
                table=pa.Table.from_pandas(part, schema=schemas[side], preserve_index=False)
            writers[side].write_table(table, max_chunksize=65536)
finally:
    for writer in writers.values():
        writer.close()
# in the same cell as the loop, so a run that stopped above never renames a half-written file
os.replace(str(PRE_CONNECTORS_FTR) + '.tmp', PRE_CONNECTORS_FTR)
os.replace(str(POST_CONNECTORS_FTR) + '.tmp', POST_CONNECTORS_FTR)
#%%
