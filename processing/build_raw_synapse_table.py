# Build the raw synapse table: every Princeton synapse before any filtering, with its size and neuropil.
# Step 08, figures/fig1/create_split_axon_dendrite_princeton.py and figures/split_methods/ read it as
# SYNAPSE_TABLE_RAW_FTR; until now no script wrote it.
#
# Same recipe as Amit's ALT_SI_COMP/1_NEW_SI_METHOD_v2.py (lines 29-52): read the Codex CSV, keep each
# row's number as synapse_id (the same ids every other table uses), drop the three cleft-coordinate
# columns, rebuild the full root ids, optionally drop self-synapses, rename the 11 columns by position.
#
# Two versions: without self-synapses (as v2 builds it) and with them. RAW_TABLE_KEEP_SELF_SYNAPSES in
# config.py chooses which one the other scripts read; BUILD below chooses which ones this script writes.
# Took 2 minutes here (1 Oct 2026), with 22 GB resident and about 40 GB peak memory footprint (macOS, 36 GB RAM);
# each version is about 3.5 GB on disk (both: about 7 GB).
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (PRINCETON_SYNAPSE_CSV, SYNAPSE_TABLE_RAW_NO_SELF_FTR,
                    SYNAPSE_TABLE_RAW_WITH_SELF_FTR, DERIVED_DATA_DIR)

import numpy as np
import pandas as pd

BUILD = "both"  # "no_self", "with_self" or "both"
assert BUILD in ("no_self", "with_self", "both"), BUILD

#%% Read the Codex CSV and repair the root ids (as in v2)
allsynapses = pd.read_csv(PRINCETON_SYNAPSE_CSV)
allsynapses = allsynapses.reset_index(drop=False)
allsynapses = allsynapses.drop(columns=['ctr_x', 'ctr_y', 'ctr_z'])
# The rename below is by position, so check the Codex column order first.
assert list(allsynapses.columns) == [
    'index', 'pre_x', 'pre_y', 'pre_z', 'post_x', 'post_y', 'post_z', 'size',
    'pre_root_id_720575940', 'post_root_id_720575940', 'neuropil'], list(allsynapses.columns)

allsynapses["pre_root_id_720575940"] = (
    720575940 * 10**allsynapses["pre_root_id_720575940"].astype(str).str.len()
    + allsynapses["pre_root_id_720575940"].astype(int)
).astype(np.int64)

allsynapses["post_root_id_720575940"] = (
    720575940 * 10**allsynapses["post_root_id_720575940"].astype(str).str.len()
    + allsynapses["post_root_id_720575940"].astype(int)
).astype(np.int64)

allsynapses.columns = [
    "synapse_id",
    "pre_x", "pre_y", "pre_z",
    "post_x", "post_y", "post_z",
    "size", "pre", "post", "neuropil"
]

#%% Write the requested versions
DERIVED_DATA_DIR.mkdir(parents=True, exist_ok=True)
self_synapse = allsynapses["pre"] == allsynapses["post"]
print(f"{len(allsynapses):,} synapses, of which {int(self_synapse.sum()):,} are self-synapses")

if BUILD in ("with_self", "both"):
    allsynapses.reset_index(drop=True).to_feather(SYNAPSE_TABLE_RAW_WITH_SELF_FTR)
    print("wrote", SYNAPSE_TABLE_RAW_WITH_SELF_FTR, f"({len(allsynapses):,} rows)")

if BUILD in ("no_self", "both"):
    no_self = allsynapses[~self_synapse].reset_index(drop=True)
    no_self.to_feather(SYNAPSE_TABLE_RAW_NO_SELF_FTR)
    print("wrote", SYNAPSE_TABLE_RAW_NO_SELF_FTR, f"({len(no_self):,} rows)")
