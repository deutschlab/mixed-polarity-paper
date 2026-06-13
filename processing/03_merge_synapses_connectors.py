

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
allsynapses=pd.read_csv(PRINCETON_SYNAPSE_CSV)
#%%
allsynapses.columns


#%%
allsynapses=allsynapses.reset_index(drop=False)
#%%
allsynapses=allsynapses.drop(columns=['ctr_x', 'ctr_y', 'ctr_z'])
#%%
allsynapses["pre_root_id_720575940"] = (
    720575940 * 10**allsynapses["pre_root_id_720575940"].astype(str).str.len()
    + allsynapses["pre_root_id_720575940"].astype(int)
).astype(np.int64)

#%%
allsynapses["post_root_id_720575940"] = (
    720575940 * 10**allsynapses["post_root_id_720575940"].astype(str).str.len()
    + allsynapses["post_root_id_720575940"].astype(int)
).astype(np.int64)
#%%
allsynapses.columns

#%%
#%%
allsynapses.columns=['synapse_id','pre_x', 'pre_y', 'pre_z', 'post_x',
       'post_y', 'post_z', 'size', 'pre',
       'post', 'neuropil']
#%%

#%%

allsynapses=allsynapses[allsynapses['pre']!=allsynapses['post']]

#%%
pre_connectors=pd.read_feather(PRE_CONNECTORS_FTR)
#%%

pre_connectors2=pd.read_feather(PRE_CONNECTORS_BIGN_FTR)
#%%

#%%
pre_connectors2.columns=['pre_x', 'pre_y', 'pre_z', 'post',
                         
                         'synapse_id','size','pre_compartment','pre']
#%%
pre_connectors2=pre_connectors2[['pre_x', 'pre_y', 'pre_z', 'post', 'pre_compartment', 'pre',
       'synapse_id', 'size']]
#%%
pre_connectors.dtypes
#%%
pre_connectors2.dtypes

#%%
pre_connectors2.columns
#%%
pre_connectors=pd.concat([pre_connectors,pre_connectors2])
#%%
#%%
bb=pre_connectors.head()
#%%
pre_connectors.columns

#%%
# Step 2: Merge based on the specified columns
# Assuming you want to merge pre_connectors with itself
merged_connectors = allsynapses.merge(
    pre_connectors[['pre_compartment','synapse_id']],
    on='synapse_id',
    how='left'  # Adjust 'how' as necessary (e.g., 'inner', 'left', etc.)
)
#%%
CONNECTORS_DIR.mkdir(parents=True, exist_ok=True)
merged_connectors.reset_index(drop=True).to_feather(SYNAPSES_PRE_MERGE_FTR)
#%%
del allsynapses
del merged_connectors
del pre_connectors
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
merged_connectors=pd.read_feather(SYNAPSES_PRE_MERGE_FTR)
#%%


#%%
post_connectors=pd.read_feather(POST_CONNECTORS_FTR)
#%%
post_connectors2=pd.read_feather(POST_CONNECTORS_BIGN_FTR)
#%%

post_connectors2.columns=['post_x', 'post_y', 'post_z', 'pre','synapse_id','size','type','post_compartment','post']
post_connectors2=post_connectors2.drop(columns=['type'])
#%%
post_connectors2=post_connectors2[['post_x', 'post_y', 'post_z', 'pre', 'post_compartment', 'post',
       'synapse_id', 'size']]



#%%
post_connectors=pd.concat([post_connectors,post_connectors2])
#%%
# Assuming you want to merge pre_connectors with itself
merged_connectors_f = merged_connectors.merge(
    post_connectors[['post_compartment','synapse_id']],
    on='synapse_id',
    how='left'  # Adjust 'how' as necessary (e.g., 'inner', 'left', etc.)
)
#%%
del merged_connectors
#%%

#%%
aaa=merged_connectors_f.head(1000)



aaa=merged_connectors_f.head()
#%%
DERIVED_DATA_DIR.mkdir(parents=True, exist_ok=True)
merged_connectors_f.reset_index(drop=True).to_feather(SYNAPSE_NON_PROCESSED_FTR)
#%%
#syn_df=pd.read_feather(r'C:\Users\user\organised_work\data\783\generated\post_processing_data\article\synapses_783_non_processed.ftr')
#%%
merged_connectors_f.dropna(subset=['post_compartment','pre_compartment'],inplace=True)
#%%
