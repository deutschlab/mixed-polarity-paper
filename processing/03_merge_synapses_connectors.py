

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
    PRINCETON_SYNAPSE_CSV,
    PRE_CONNECTORS_FTR, PRE_CONNECTORS_BIGN_FTR,
    POST_CONNECTORS_FTR, POST_CONNECTORS_BIGN_FTR,
    SYNAPSES_PRE_MERGE_FTR, SYNAPSE_NON_PROCESSED_FTR,
    CONNECTORS_DIR, DERIVED_DATA_DIR,
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
# The pre labels: step 02b's table for the standard cells and step 02a's for the large ones. Only the two columns the
# join uses are read from step 02b's table.
pre_connectors=pd.read_feather(PRE_CONNECTORS_FTR, columns=['pre_compartment','synapse_id'])
#%%
pre_connectors2=pd.read_feather(PRE_CONNECTORS_BIGN_FTR)
pre_connectors2.columns=['pre_x', 'pre_y', 'pre_z', 'post',
                         'synapse_id','size','pre_compartment','pre']
pre_connectors=pd.concat([pre_connectors,pre_connectors2[['pre_compartment','synapse_id']]])
del pre_connectors2
#%%
# Each synapse_id is in the label tables once, so a left join on synapse_id is a lookup: every synapse keeps its row
# and order, and gets the label of its synapse_id, or NaN when there is none.
assert pre_connectors['synapse_id'].is_unique
pre_label=pd.Series(pre_connectors['pre_compartment'].to_numpy(), index=pre_connectors['synapse_id'].to_numpy())
del pre_connectors

#%%
# Read the synapse table in chunks, as step 02a does, so the whole CSV is never in memory at once. Each chunk gets
# what the full table used to get: synapse_id is the row number in the CSV, counting its first data row as 0, the
# root IDs are rebuilt from their shortened form, the columns are named, and synapses from a neuron to itself are
# dropped. Then it gets its pre label and is added to SYNAPSES_PRE_MERGE_FTR, a feather (Arrow IPC) file written
# under a temporary name and renamed at the end. A blank or decimal value in a column read as int64 (all but neuropil)
# stops the read.
import pyarrow as pa

syn_cols=['pre_x', 'pre_y', 'pre_z', 'post_x', 'post_y', 'post_z', 'size',
          'pre_root_id_720575940', 'post_root_id_720575940']
CONNECTORS_DIR.mkdir(parents=True, exist_ok=True)
writer=None
try:
    for chunk in pd.read_csv(PRINCETON_SYNAPSE_CSV, usecols=syn_cols+['neuropil'],
                             dtype={**{c: np.int64 for c in syn_cols}, 'neuropil': object}, chunksize=5_000_000):
        chunk.insert(0, 'synapse_id', chunk.index.to_numpy(dtype=np.int64))
        for c in ['pre_root_id_720575940', 'post_root_id_720575940']:
            chunk[c] = (720575940 * 10**chunk[c].astype(str).str.len() + chunk[c].astype(int)).astype(np.int64)
        chunk=chunk[['synapse_id']+syn_cols+['neuropil']].rename(
            columns={'pre_root_id_720575940': 'pre', 'post_root_id_720575940': 'post'})
        chunk=chunk[chunk['pre']!=chunk['post']].reset_index(drop=True)
        chunk['pre_compartment']=chunk['synapse_id'].map(pre_label)
        if writer is None:
            table=pa.Table.from_pandas(chunk, preserve_index=False)
            schema=table.schema
            writer=pa.ipc.new_file(str(SYNAPSES_PRE_MERGE_FTR) + '.tmp', schema,
                                   options=pa.ipc.IpcWriteOptions(compression='lz4'))
        else:
            table=pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
        writer.write_table(table, max_chunksize=65536)
finally:
    if writer is not None:
        writer.close()
# in the same cell as the loop, so a run that stopped above never renames a file that lacks some chunks
os.replace(str(SYNAPSES_PRE_MERGE_FTR) + '.tmp', SYNAPSES_PRE_MERGE_FTR)
#%%
del pre_label
#%%  del all and restart

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
    PRINCETON_SYNAPSE_CSV,
    PRE_CONNECTORS_FTR, PRE_CONNECTORS_BIGN_FTR,
    POST_CONNECTORS_FTR, POST_CONNECTORS_BIGN_FTR,
    SYNAPSES_PRE_MERGE_FTR, SYNAPSE_NON_PROCESSED_FTR,
    CONNECTORS_DIR, DERIVED_DATA_DIR,
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
#%%
# The post labels, the same way
import pyarrow as pa
post_connectors=pd.read_feather(POST_CONNECTORS_FTR, columns=['post_compartment','synapse_id'])
#%%
post_connectors2=pd.read_feather(POST_CONNECTORS_BIGN_FTR)
post_connectors2.columns=['post_x', 'post_y', 'post_z', 'pre','synapse_id','size','type','post_compartment','post']
post_connectors=pd.concat([post_connectors,post_connectors2[['post_compartment','synapse_id']]])
del post_connectors2
assert post_connectors['synapse_id'].is_unique
post_label=pd.Series(post_connectors['post_compartment'].to_numpy(), index=post_connectors['synapse_id'].to_numpy())
del post_connectors
#%%
# SYNAPSES_PRE_MERGE_FTR is read back piece by piece; each piece gets its post label (the same lookup) and is added
# to SYNAPSE_NON_PROCESSED_FTR, written under a temporary name and renamed at the end.
DERIVED_DATA_DIR.mkdir(parents=True, exist_ok=True)
# (batch is one piece of the table, not all of it)
writer=None
try:
    with pa.ipc.open_file(SYNAPSES_PRE_MERGE_FTR) as reader:
        for i in range(reader.num_record_batches):
            batch=reader.get_batch(i).to_pandas()
            batch['post_compartment']=batch['synapse_id'].map(post_label)
            if writer is None:
                table=pa.Table.from_pandas(batch, preserve_index=False)
                schema=table.schema
                writer=pa.ipc.new_file(str(SYNAPSE_NON_PROCESSED_FTR) + '.tmp', schema,
                                       options=pa.ipc.IpcWriteOptions(compression='lz4'))
            else:
                table=pa.Table.from_pandas(batch, schema=schema, preserve_index=False)
            writer.write_table(table, max_chunksize=65536)
finally:
    if writer is not None:
        writer.close()
os.replace(str(SYNAPSE_NON_PROCESSED_FTR) + '.tmp', SYNAPSE_NON_PROCESSED_FTR)
#%%
# (A dropna on the two label columns stood here, after the table was written, so it changed nothing; step 04 drops
# the synapses missing either label.)
#%%
