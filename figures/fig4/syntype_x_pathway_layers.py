# Fig 4E (revision): connection types along two sensory pathways. For stages of the visual pathway
# (LA, ME, LO, LOP and the visual projection neurons) and of the olfactory pathway (AL, LH + MB_CA,
# the mushroom body lobes, and CRE + SMP + SIP + LAL), the share of AD, AA, DD and DA synapses,
# over all synapses and over synapses between cells of the same type (and, for olfaction, of
# different types), for rows that pass an SI >= 0.1 filter (see the input below). Tables and six plots; the
# revised Fig 4E is plots 3-6 (the DD and AA lines).
# Ported from the corresponding author's MATLAB script FlyWire_VisualOlfactoryPathway_LayerStats.m
# (not included in this repository).
# Changes:
#   - paths via config.py; the tables are written next to the figures (the MATLAB script wrote them
#     into its data folder), and the MATLAB .fig copies of the figures are not made;
#   - its input ex_table_npil_comp_pre_post_hiSI_PerTypeSummary.csv (one row per pre type, post type and
#     connection type, with one npil label; kept where the mean SI over the row's records, and over its
#     two types' records, is >= 0.1) is not in the repository and no code here writes it: see DATA_SOURCE below;
#   - neuron_data.csv, which the MATLAB script reads for the olfactory cell classes, is not in the
#     repository either; NEURON_DATA_FTR stands in for it (same columns: class, primary_type, cell_type);
#   - Olfactory_ClassNeuropilStats.csv is only printed by the MATLAB script; it is printed here if
#     present (OLFACTORY_CLASS_NEUROPIL_CSV) and skipped with a note if not (the MATLAB script stopped);
#   - missing text in the table is read as empty text and text is trimmed of surrounding spaces, as
#     MATLAB's readtable does, so two empty types count as the same type; a missing synapse count
#     makes the shares it falls in NaN, as MATLAB's sum does;
#   - with a stand-in (DATA_SOURCE 'princeton' or 'buhmann') the olfactory types are the chosen neuron
#     table's primary_type of the cells NEURON_DATA_FTR puts in the four olfactory classes (joined by
#     root_id), so they match the types of the stand-in's rows (81 olfactory cells have another
#     primary_type in NEURON_DATA_FTR than in NEURON_TABLE_FTR); with 'csv' they follow the MATLAB rule;
#   - run_info.txt in the output folder records which input the run used;
#   - figure size, fonts and legend placement follow matplotlib, not MATLAB; the plotted values,
#     colours, order and axis limits are the MATLAB script's.
# svg.fonttype='none' and Arial (repository convention).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (NEURON_DATA_FTR, NEURON_TABLE_FTR, NEURON_TABLE_NONP_FTR, OLFACTORY_CLASS_NEUROPIL_CSV,
                    OUTPUT_DIR, PER_TYPE_SUMMARY_CSV, SI_UPDATED_FTR, SYNAPSE_TABLE_FTR, SYNAPSE_TABLE_NONP_FTR)

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

_out_dir = OUTPUT_DIR / "fig4" / "syntype_x_pathway_layers"
_out_dir.mkdir(parents=True, exist_ok=True)

#%% parameters
# Where the per-type table comes from: 'csv' reads the authors' table (PER_TYPE_SUMMARY_CSV; put it
# in data/derived/), which the revised panel was drawn from. 'princeton' or 'buhmann' rebuild a
# stand-in from this repository's synapse and neuron tables (SYNAPSE_TABLE_FTR and NEURON_TABLE_FTR,
# or SYNAPSE_TABLE_NONP_FTR and NEURON_TABLE_NONP_FTR): synapses counted per pre type, post type,
# comp and npil, over cells with a type and SI >= 0.1 on both sides. A stand-in is not the panel:
# on 16 Oct 2026 none of the three gave the revised panel's values.
DATA_SOURCE = 'csv'
# With DATA_SOURCE = 'princeton' only: 'neuron_table' (NEURON_TABLE_FTR's SI) or 'SI_updated'.
SI_SOURCE = 'neuron_table'
assert DATA_SOURCE in ('csv', 'princeton', 'buhmann'), DATA_SOURCE
assert SI_SOURCE in ('neuron_table', 'SI_updated'), SI_SOURCE
SI_threshold = 0.1   # used only to rebuild a stand-in; the authors' table is already filtered

layerOrder = ['LA', 'ME', 'LO', 'LOP', 'VPN']   # row order for both tables and both plots

# colours of the four connection types, as in the Figure 6 script
colorAA = np.array([120, 80, 161]) / 255
colorDD = np.array([243, 185, 91]) / 255
colorAD = np.array([159, 75, 35]) / 255
colorDA = np.array([180, 180, 180]) / 255

TEXT_COLUMNS = ['preType', 'postType', 'comp', 'npil', 'pre_SuperClass', 'post_SuperClass']

#%% read data
if DATA_SOURCE == 'csv':
    if not PER_TYPE_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"{PER_TYPE_SUMMARY_CSV} is missing. It is the authors' per-type table, which this "
                                "repository does not build (docs/data_availability.md); set DATA_SOURCE = 'princeton' "
                                "or 'buhmann' to rebuild a stand-in from the synapse tables.")
    T = pd.read_csv(PER_TYPE_SUMMARY_CSV, dtype={c: str for c in TEXT_COLUMNS + ['pre_nt', 'post_nt']},
                    keep_default_na=False, na_values={'preSI': [''], 'postSI': [''], 'synapses': ['']})
    source_name = str(PER_TYPE_SUMMARY_CSV)
else:
    neuron_table = NEURON_TABLE_FTR if DATA_SOURCE == 'princeton' else NEURON_TABLE_NONP_FTR
    synapse_table = SYNAPSE_TABLE_FTR if DATA_SOURCE == 'princeton' else SYNAPSE_TABLE_NONP_FTR
    for table in (neuron_table, synapse_table):
        if not table.exists():
            raise FileNotFoundError(f"{table} is missing (DATA_SOURCE = '{DATA_SOURCE}').")
    nodes = pd.read_feather(neuron_table, columns=['root_id', 'primary_type', 'super_class', 'SI'])
    nodes['root_id'] = nodes['root_id'].astype(np.int64)
    assert nodes['root_id'].is_unique
    if DATA_SOURCE == 'princeton' and SI_SOURCE == 'SI_updated':
        si_upd = pd.read_feather(SI_UPDATED_FTR, columns=['root_id', 'SI'])
        si_upd['root_id'] = si_upd['root_id'].astype(np.int64)
        nodes = nodes.drop(columns='SI').merge(si_upd, on='root_id', how='left')
    nodes = nodes.set_index('root_id')
    syn = feather.read_table(synapse_table, columns=['pre', 'post', 'npil', 'comp'])
    conn = syn.group_by(['pre', 'post', 'npil', 'comp']).aggregate([([], 'count_all')]).to_pandas()
    del syn
    for side in ('pre', 'post'):
        conn[side] = conn[side].astype(np.int64)
        conn[side + 'Type'] = conn[side].map(nodes['primary_type'])
        conn[side + 'SI'] = conn[side].map(nodes['SI'])
        conn[side + '_SuperClass'] = conn[side].map(nodes['super_class'])
    conn = conn[(conn['preSI'] >= SI_threshold) & (conn['postSI'] >= SI_threshold)
                & conn['preType'].notna() & conn['postType'].notna()]
    conn = conn[(conn['preType'].astype(str).str.strip() != '') & (conn['postType'].astype(str).str.strip() != '')]
    conn['npil'] = conn['npil'].fillna('')
    T = (conn.groupby(['preType', 'postType', 'comp', 'npil', 'pre_SuperClass', 'post_SuperClass'], dropna=False)
         .agg(synapses=('count_all', 'sum'), preSI=('preSI', 'min'), postSI=('postSI', 'min')).reset_index())
    del conn
    source_name = f"a stand-in rebuilt from {synapse_table.name} and {neuron_table.name}"
for c in TEXT_COLUMNS:
    T[c] = T[c].fillna('').astype(str).str.strip()
(_out_dir / 'run_info.txt').write_text(f"DATA_SOURCE = {DATA_SOURCE!r}, SI_SOURCE = {SI_SOURCE!r}\ninput: {source_name}\n",
                                      encoding='utf-8')

print(f"read {len(T)} rows from {source_name}")
print(f"preSI range:  {T['preSI'].min():g} - {T['preSI'].max():g}")
print(f"postSI range: {T['postSI'].min():g} - {T['postSI'].max():g}")
print("(both should be >=0.1 - this table is already SI-filtered on both the pre and post side)")

#%% strip the left/right suffix off the neuropil ('ME_L' and 'ME_R' both become 'ME')
T['region'] = T['npil'].str.replace(r'_[LR]$', '', regex=True)
T.loc[T['region'] == '', 'region'] = 'None'   # rows with no neuropil recorded

#%% within-type rows (pre and post cell of the same type) and VPN rows (either cell visual_projection)
isWithinType = (T['preType'] == T['postType']).to_numpy()
isVPN = ((T['pre_SuperClass'] == 'visual_projection') | (T['post_SuperClass'] == 'visual_projection')).to_numpy()


def local_compCompositionStats(T, mask):
    """% of AD, AA, DD and DA synapses among the rows in mask (NaN when there are none), their total,
    and the number of distinct cell types appearing as pre or post among those rows."""
    Tsub = T[mask]
    tot = {c: Tsub.loc[Tsub['comp'] == c, 'synapses'].sum(skipna=False) for c in ('AD', 'AA', 'DD', 'DA')}
    totSyn = tot['AD'] + tot['AA'] + tot['DD'] + tot['DA']
    if totSyn == 0 or np.isnan(totSyn):
        pct = {c: np.nan for c in tot}
    else:
        pct = {c: 100 * tot[c] / totSyn for c in tot}
    nTypes = len(set(Tsub['preType']) | set(Tsub['postType']))
    return pct['AD'], pct['AA'], pct['DD'], pct['DA'], totSyn, nTypes


def composition_table(names, masks, key='layer'):
    rows = []
    for name, mask in zip(names, masks):
        pctAD, pctAA, pctDD, pctDA, totSyn, nTypes = local_compCompositionStats(T, mask)
        rows.append({key: name, 'synapses': totSyn, 'nTypes': nTypes,
                     'AD_pct': pctAD, 'AA_pct': pctAA, 'DD_pct': pctDD, 'DA_pct': pctDA})
    return pd.DataFrame(rows)


def stacked_bars(Tab, fig_num, title, name):
    fig = plt.figure(fig_num)
    fig.clf()
    ax = fig.add_subplot(111)
    x = np.arange(1, len(Tab) + 1)
    bottom = np.zeros(len(Tab))
    for col, color, label in (('AD_pct', colorAD, 'AD'), ('AA_pct', colorAA, 'AA'),
                              ('DD_pct', colorDD, 'DD'), ('DA_pct', colorDA, 'DA')):
        values = Tab[col].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, color=color, label=label, width=0.8)
        bottom = bottom + np.nan_to_num(values)
    ax.set_xticks(x)
    ax.set_xticklabels(Tab['layer'])
    ax.set_ylabel('% of synapses')
    ax.set_ylim(0, 100)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title(title, fontsize=12)
    fig.savefig(_out_dir / f"{name}.svg", format='svg', bbox_inches='tight')


def dd_aa_lines(Tab, fig_num, title, name, label_col, rotation=0):
    fig = plt.figure(fig_num)
    fig.clf()
    ax = fig.add_subplot(111)
    x = np.arange(1, len(Tab) + 1)
    ax.plot(x, Tab['DD_pct'], '-o', color=colorDD, linewidth=2, markerfacecolor=colorDD, label='DD')
    ax.plot(x, Tab['AA_pct'], '-o', color=colorAA, linewidth=2, markerfacecolor=colorAA, label='AA')
    ax.set_xticks(x)
    ax.set_xticklabels(Tab[label_col], rotation=rotation)
    ax.set_xlim(0.5, len(x) + 0.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel('% of synapses')
    ax.legend(loc='best', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title(title, fontsize=12)
    fig.savefig(_out_dir / f"{name}.svg", format='svg', bbox_inches='tight')


mpl.rcParams['font.size'] = 12

#%% Table 1: ALL synapses; Table 2: WITHIN-TYPE synapses only
layer_masks = [isVPN if layer == 'VPN' else (T['region'] == layer).to_numpy() for layer in layerOrder]
T_all = composition_table(layerOrder, layer_masks)
T_within = composition_table(layerOrder, [m & isWithinType for m in layer_masks])
# NOTE for the VPN row nTypes counts every type (VPN or not) that appears as pre or post in a
# synapse touching a VPN, not just VPN types
print("\n=== Table 1: ALL synapses, by layer ===")
print(T_all.to_string(index=False))
print("\n=== Table 2: WITHIN-TYPE synapses only, by layer ===")
print(T_within.to_string(index=False))
T_all.to_csv(_out_dir / 'LayerComposition_AllSynapses.csv', index=False)
T_within.to_csv(_out_dir / 'LayerComposition_WithinType.csv', index=False)

#%% plots 1 and 2: 100%-stacked bars of AD/AA/DD/DA across layers
stacked_bars(T_all, 1, 'ALL synapses: AD/AA/DD/DA composition by layer (SI>=0.1 both sides)',
             'LayerComposition_AllSynapses')
stacked_bars(T_within, 2, 'WITHIN-TYPE synapses only: AD/AA/DD/DA composition by layer (SI>=0.1 both sides)',
             'LayerComposition_WithinType')

#%% collapsed groups: LA+ME, LO+LOP and VPN (Tables 3 and 4)
collapsedOrder = ['LA+ME', 'LO+LOP', 'VPN']
collapsedRegions = [['LA', 'ME'], ['LO', 'LOP'], []]   # the VPN group uses isVPN instead
collapsed_masks = [isVPN if g == 'VPN' else T['region'].isin(r).to_numpy()
                   for g, r in zip(collapsedOrder, collapsedRegions)]
T_all_collapsed = composition_table(collapsedOrder, collapsed_masks, key='group')
T_within_collapsed = composition_table(collapsedOrder, [m & isWithinType for m in collapsed_masks], key='group')
print("\n=== Table 3: ALL synapses, LA+ME / LO+LOP collapsed ===")
print(T_all_collapsed.to_string(index=False))
print("\n=== Table 4: WITHIN-TYPE synapses only, LA+ME / LO+LOP collapsed ===")
print(T_within_collapsed.to_string(index=False))
T_all_collapsed.to_csv(_out_dir / 'LayerComposition_AllSynapses_collapsed.csv', index=False)
T_within_collapsed.to_csv(_out_dir / 'LayerComposition_WithinType_collapsed.csv', index=False)

#%% plots 3 and 4: DD and AA lines across the collapsed groups (Fig 4E, the two visual panels)
dd_aa_lines(T_all_collapsed, 3, 'ALL synapses: DD vs AA across collapsed groups (SI>=0.1 both sides)',
            'LayerComposition_AllSynapses_collapsed_lines', 'group')
dd_aa_lines(T_within_collapsed, 4, 'WITHIN-TYPE synapses only: DD vs AA across collapsed groups (SI>=0.1 both sides)',
            'LayerComposition_WithinType_collapsed_lines', 'group')

#%% olfactory layers: which neuropils make up each, and why
olfactoryLayerOrder = ['AL', 'LH+MB-CA', 'MB lobes (ML/PED/VL)', 'CRE+SMP+SIP+LAL']
olfactoryLayerRegions = [['AL'], ['LH', 'MB_CA'], ['MB_ML', 'MB_PED', 'MB_VL'], ['CRE', 'SMP', 'SIP', 'LAL']]
T_layerDefinitions = pd.DataFrame({
    'layer': olfactoryLayerOrder,
    'neuropils': ['AL', 'LH, MB_CA', 'MB_ML, MB_PED, MB_VL', 'CRE, SMP, SIP, LAL'],
    'rationale': [
        'Input stage. Olfactory receptor neurons (ORN) have ~99.6-99.8% of BOTH their input and output synapses '
        'in AL (they barely leave it), and olfactory projection neurons (ALPN/PN) receive the large majority of '
        'their input here (~82%).',
        'PN output stage, split into two parallel targets: the lateral horn (LH - the "innate" pathway, ~48% of '
        'PN output) and the mushroom body calyx (MB_CA - the "learned" pathway, ~21% of PN output), where Kenyon '
        'cell (KC) dendrites receive their input.',
        'KC output stage - Kenyon cell axons run through and synapse within the mushroom body lobes '
        '(medial/gamma+beta-prime, peduncle, vertical/alpha+alpha-prime). Mushroom body output neurons (MBON) '
        'receive most of their input here (~58% of MBON input, summed across the 3 lobes).',
        'MBON output stage - MBONs send the large majority of their own output here, exiting the mushroom body to '
        'downstream central-brain regions. The olfactory analogue of the VPN row in the visual pathway part of '
        'this script: the final, "output" stage of the circuit.'],
})
print("\n=== Olfactory layer definitions (why these 4 groups were formed) ===")
print(T_layerDefinitions.to_string(index=False))
T_layerDefinitions.to_csv(_out_dir / 'OlfactoryLayerDefinitions.csv', index=False)

# the numbers behind those definitions, computed outside this script (printed only)
if OLFACTORY_CLASS_NEUROPIL_CSV.exists():
    T_classNeuropilStats = pd.read_csv(OLFACTORY_CLASS_NEUROPIL_CSV, dtype={'cellClass': str, 'region': str})
    print("\n=== Concrete numbers behind the olfactory layer definitions ===")
    print("(pct_input/pct_output = % of that CLASS's total input/output synapses found in that region)")
    print(T_classNeuropilStats.to_string(index=False))
else:
    print(f"\n{OLFACTORY_CLASS_NEUROPIL_CSV.name} is not here; the numbers behind the layer definitions are not printed.")

#%% the cell types of the four olfactory classes (ORN = 'olfactory', PN = 'ALPN', KC = 'Kenyon_Cell', MBON)
isBetweenType = ~isWithinType
T_neuronData = pd.read_feather(NEURON_DATA_FTR, columns=['root_id', 'class', 'primary_type', 'cell_type'])
for c in ('class', 'primary_type', 'cell_type'):
    T_neuronData[c] = T_neuronData[c].fillna('').astype(str)
olfactoryClasses = ['olfactory', 'ALPN', 'Kenyon_Cell', 'MBON']
isOlfClass = T_neuronData['class'].isin(olfactoryClasses)
if DATA_SOURCE == 'csv':
    # primary_type where there is one (it matches preType/postType), otherwise cell_type
    typeName = T_neuronData['primary_type'].where(T_neuronData['primary_type'] != '', T_neuronData['cell_type'])
    olfactoryTypeNames = sorted(set(typeName[isOlfClass]) - {''})
else:
    # the stand-in's types come from the neuron table, so take those cells' types from it too
    olf_ids = T_neuronData.loc[isOlfClass, 'root_id'].astype(np.int64)
    olf_types = nodes['primary_type'].reindex(olf_ids).dropna().astype(str).str.strip()
    olfactoryTypeNames = sorted(set(olf_types) - {''})
print(f"\n{len(olfactoryTypeNames)} distinct cell types belong to ORN/ALPN/Kenyon_Cell/MBON")

# only synapses whose pre and post cell types are both olfactory-class
isOlfactoryPair = (T['preType'].isin(olfactoryTypeNames) & T['postType'].isin(olfactoryTypeNames)).to_numpy()
print(f"{isOlfactoryPair.sum()} / {len(T)} rows are olfactory-to-olfactory type pairs")

#%% olfactory Tables 1-3: all, within-type and between-type synapses, olfactory-to-olfactory only
olf_masks = [T['region'].isin(r).to_numpy() & isOlfactoryPair for r in olfactoryLayerRegions]
T_olf_all = composition_table(olfactoryLayerOrder, olf_masks)
T_olf_within = composition_table(olfactoryLayerOrder, [m & isWithinType for m in olf_masks])
T_olf_between = composition_table(olfactoryLayerOrder, [m & isBetweenType for m in olf_masks])
print("\n=== Olfactory Table 1: ALL synapses (olfactory-to-olfactory only), by layer ===")
print(T_olf_all.to_string(index=False))
print("\n=== Olfactory Table 2: WITHIN-TYPE synapses only (olfactory-to-olfactory only), by layer ===")
print(T_olf_within.to_string(index=False))
print("\n=== Olfactory Table 3: BETWEEN-TYPE synapses only (olfactory-to-olfactory only), by layer ===")
print(T_olf_between.to_string(index=False))
# the file names are the MATLAB script's; all three are olfactory-to-olfactory type pairs only
T_olf_all.to_csv(_out_dir / 'LayerComposition_Olfactory_AllSynapses.csv', index=False)
T_olf_within.to_csv(_out_dir / 'LayerComposition_Olfactory_WithinType.csv', index=False)
T_olf_between.to_csv(_out_dir / 'LayerComposition_Olfactory_BetweenType.csv', index=False)

#%% plots 5 and 6: DD and AA lines across the olfactory layers (Fig 4E, the two olfactory panels).
# 'LH+MB-CA' is left out of these two plots only; it stays in the tables.
dd_aa_lines(T_olf_all[T_olf_all['layer'] != 'LH+MB-CA'].reset_index(drop=True), 5,
            'OLFACTORY (ORN+ALPN+KC+MBON only) - ALL synapses: DD vs AA across layers',
            'LayerComposition_Olfactory_AllSynapses_lines', 'layer', rotation=20)
dd_aa_lines(T_olf_within[T_olf_within['layer'] != 'LH+MB-CA'].reset_index(drop=True), 6,
            'OLFACTORY (ORN+ALPN+KC+MBON only) - WITHIN-TYPE synapses only: DD vs AA across layers',
            'LayerComposition_Olfactory_WithinType_lines', 'layer', rotation=20)
plt.show()
