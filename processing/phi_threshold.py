# -*- coding: utf-8 -*-
"""
Find the Phi value that matches the SI = 0.1 cutoff.

SI and Phi both score how well a neuron keeps its inputs and outputs apart on its
axon and dendrite, but on different scales. A neuron with SI below 0.1 is called
mixed. So Phi needs its own cutoff that draws the same line.

Phi does not follow SI exactly. At the same SI, a neuron with very unequal numbers
of inputs and outputs gets a slightly lower Phi. So the match holds on average, not
for every single neuron.

The data: SI_comparisons.ftr, written by 08_alternative_split_methods.py, gives
each neuron both scores for the published SFC split (SFC_SI and SFC_Phi). Only
intrinsic neurons (central, optic, visual_projection, visual_centrifugal) are used,
because the figures that use the cutoff keep only those. The script stops if SFC_SI
is not the corrected SI from SI_updated.ftr.

The Phi cutoff is found in three ways:
  1. local median: the typical Phi of the neurons with SI within 0.005 of 0.1;
  2. same share: the Phi value that calls the same share of neurons mixed as
     SI < 0.1 does;
  3. best agreement: the Phi cutoff that agrees with SI < 0.1 for the most neurons.

The recommended cutoff is the median of the three, rounded to two decimals. Report
it with the range of the three: they differ by more than their confidence
intervals, so the range is the honest uncertainty. Use the same cutoff for every
split method, together with SI = 0.1, instead of picking a cutoff per method. The
confidence intervals resample whole cell types (primary_type), because neurons of
the same type are not independent.

Reads SI_COMPARISONS_FTR, SI_UPDATED_FTR and, from NEURON_TABLE_FTR, super_class,
primary_type and the axon and dendrite input and output counts (only to group
neurons). Prints the results and saves one figure,
OUTPUT_DIR / "phi_threshold" / "phi_threshold.svg".
"""
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SI_COMPARISONS_FTR, SI_UPDATED_FTR, NEURON_TABLE_FTR, OUTPUT_DIR

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SI_CUTOFF = 0.1
WINDOW = 0.005          # SI window around the cutoff for the local median
N_BOOTSTRAP = 1000
INTRINSIC = ['central', 'optic', 'visual_projection', 'visual_centrifugal']

#%% load
scores = pd.read_feather(SI_COMPARISONS_FTR)
scores['neuron_id'] = scores['neuron_id'].astype(np.int64)

nodes = pd.read_feather(NEURON_TABLE_FTR, columns=['neuron', 'super_class', 'primary_type',
                                                  'pre_A', 'pre_D', 'post_A', 'post_D'])
nodes['neuron'] = nodes['neuron'].astype(np.int64)
scores = scores.merge(nodes, left_on='neuron_id', right_on='neuron', how='left')

n_duplicated = scores['neuron_id'].duplicated().sum()
n_missing = scores['neuron'].isna().sum()
print(f"neurons: {len(scores)}, duplicated IDs: {n_duplicated}, not in the neuron table: {n_missing}")
if n_duplicated or n_missing:
    raise ValueError("SI_comparisons.ftr has duplicated IDs or neurons missing from the neuron table")
n_all = len(scores)
scores = scores[scores['super_class'].isin(INTRINSIC)].reset_index(drop=True)
print(f"intrinsic neurons kept: {len(scores)} (left out: {n_all - len(scores)} of other super-classes)")
print("missing values:", scores[['SFC_SI', 'SFC_Phi', 'MaxSI_SI', 'MinFisherP_Phi']].isna().sum().to_dict())
scores = scores.dropna(subset=['SFC_SI', 'SFC_Phi']).reset_index(drop=True)
print(f"neurons with both SFC_SI and SFC_Phi (used below): {len(scores)}")

# SFC_SI should be the corrected SI
si_updated = pd.read_feather(SI_UPDATED_FTR)
si_updated['root_id'] = si_updated['root_id'].astype(np.int64)
if si_updated['root_id'].duplicated().any():
    raise ValueError("SI_updated.ftr has duplicated root_id values")
check = scores[['neuron_id', 'SFC_SI']].merge(si_updated[['root_id', 'SI']],
                                            left_on='neuron_id', right_on='root_id', how='left')
largest = np.nanmax(np.abs(check['SFC_SI'] - check['SI']))
print(f"SFC_SI vs SI_updated: {check['SI'].notna().sum()} matched, largest difference {largest:.2e}")
if check['SI'].isna().any() or largest > 1e-12:
    raise ValueError("SFC_SI is not the corrected SI; the cutoff would be calibrated on a different SI")

#%% the three estimates
def same_share_cutoff(phi, n_mixed):
    """Return the cutoff that puts n_mixed values below it (fewer if the values there are tied)."""
    if not 0 < n_mixed < len(phi):
        raise ValueError("no neuron, or every neuron, is below the SI cutoff")
    s = np.sort(phi)
    return (s[n_mixed - 1] + s[n_mixed]) / 2


def best_agreement_cutoff(phi, mixed):
    """Return the cutoff where (phi < cutoff) matches `mixed` for the most neurons.
    If several cutoffs do equally well, return the lowest."""
    order = np.argsort(phi, kind='stable')
    s = phi[order]
    m = mixed[order].astype(float)
    n = len(s)
    k = np.arange(1, n + 1)                     # the first k are called mixed
    true_mixed = np.cumsum(m)
    # share called right: mixed neurons called mixed plus segregated neurons called segregated
    agree = (true_mixed + (n - k) - (m.sum() - true_mixed)) / n
    valid = np.r_[s[1:] > s[:-1], False]        # only between two different values
    i = np.where(valid)[0]
    if len(i) == 0:
        raise ValueError("all Phi values are equal")
    best = i[np.argmax(agree[i])]
    return (s[best] + s[best + 1]) / 2, agree[best]


def estimates(si, phi):
    mixed = si < SI_CUTOFF
    near = np.abs(si - SI_CUTOFF) <= WINDOW
    if not near.any():
        raise ValueError(f"no neuron has SI within {WINDOW} of {SI_CUTOFF}")
    local = np.median(phi[near])
    share = same_share_cutoff(phi, int(mixed.sum()))
    best, best_agree = best_agreement_cutoff(phi, mixed)
    return local, share, best, best_agree, near.sum()


si = scores['SFC_SI'].to_numpy()
phi = scores['SFC_Phi'].to_numpy()
mixed = si < SI_CUTOFF
near = np.abs(si - SI_CUTOFF) <= WINDOW
local, share, best, best_agree, n_near = estimates(si, phi)
three = [local, share, best]
phi_cutoff = round(float(np.median(three)), 2)

print(f"\nSFC_SI < {SI_CUTOFF}: {mixed.sum()} of {len(si)} neurons ({mixed.mean():.2%})")
print(f"1. local median:   {local:.3f}  (n = {n_near}; the middle half of these neurons have Phi "
      f"{np.percentile(phi[near], 25):.3f}-{np.percentile(phi[near], 75):.3f}, "
      f"5-95%: {np.percentile(phi[near], 5):.3f}-{np.percentile(phi[near], 95):.3f})")
print(f"2. same share:     {share:.3f}")
print(f"3. best agreement: {best:.3f}  ({best_agree:.2%} of neurons agree)")
print(f"Spearman SFC_SI vs SFC_Phi, all neurons: {spearmanr(si, phi).correlation:.3f}")
print(f"\nRecommended Phi cutoff: {phi_cutoff} (median of the three is {np.median(three):.3f}, "
      f"rounded to 2 decimals; the three range from {min(three):.3f} to {max(three):.3f})")
print(f"Neurons on different sides of SI {SI_CUTOFF} and Phi {phi_cutoff}: "
      f"{((phi < phi_cutoff) != mixed).sum()} ({((phi < phi_cutoff) != mixed).mean():.2%})")

#%% share of neurons called mixed, same cutoffs for every method
print(f"\nShare mixed at SI < {SI_CUTOFF} and Phi < {phi_cutoff} "
      f"(of the {len(scores)} neurons with both SFC scores):")
for column, cutoff in [('SFC_SI', SI_CUTOFF), ('SFC_Phi', phi_cutoff),
                       ('MaxSI_SI', SI_CUTOFF), ('MinFisherP_Phi', phi_cutoff)]:
    values = scores[column].dropna()
    print(f"  {column:15s} {(values < cutoff).mean():.1%}  (n = {len(values)})")
low, high = min(three), max(three)
for column in ['SFC_Phi', 'MinFisherP_Phi']:
    values = scores[column].dropna()
    print(f"  {column} for a Phi cutoff from {low:.3f} to {high:.3f}: "
          f"{(values < low).mean():.1%} to {(values < high).mean():.1%}")

#%% does the match hold across groups of neurons? (local median, neurons near SI = 0.1)
near_df = scores[near].copy()
total = near_df[['pre_A', 'pre_D', 'post_A', 'post_D']].sum(axis=1)
near_df['output share'] = (near_df['pre_A'] + near_df['pre_D']) / total
near_df['synapses'] = total
print("\nLocal median by super-class:")
print(near_df.groupby('super_class')['SFC_Phi'].agg(['count', 'median']).round(3))
for column in ['output share', 'synapses']:
    groups = pd.qcut(near_df[column], 4, duplicates='drop')
    print(f"\nLocal median in {groups.nunique()} equal-sized groups of {column} "
          "(counts from the neuron table):")
    print(near_df.groupby(groups, observed=True)['SFC_Phi'].agg(['count', 'median']).round(3))

#%% intervals, resampling whole primary types
rng = np.random.default_rng(0)
cluster = scores['primary_type'].fillna(scores['neuron_id'].astype(str)).to_numpy()
codes, _ = pd.factorize(cluster)
members = list(pd.Series(np.arange(len(codes))).groupby(codes).apply(np.array))
boot = []
for _ in range(N_BOOTSTRAP):
    picked = rng.integers(0, len(members), len(members))
    rows = np.concatenate([members[c] for c in picked])
    e = estimates(si[rows], phi[rows])[:3]
    boot.append([*e, np.median(e)])
boot = np.array(boot)
print(f"\n95% intervals over {N_BOOTSTRAP} resamples of {len(members)} primary types "
      "(neurons without a type count as their own type):")
for name, column in zip(['local median', 'same share', 'best agreement', 'median of the three'], boot.T):
    print(f"  {name:20s} {np.percentile(column, 2.5):.3f} to {np.percentile(column, 97.5):.3f}")

#%% figure: the typical Phi at each SI, and the Phi of the neurons at SI = 0.1
_out_dir = OUTPUT_DIR / "phi_threshold"
_out_dir.mkdir(parents=True, exist_ok=True)

step = 0.01
edges = np.arange(0, 0.3 + step / 2, step)
binned = pd.DataFrame({'si': si, 'phi': phi})
binned['bin'] = pd.cut(binned['si'], edges, right=False)
curve = binned.groupby('bin', observed=True)['phi'].agg(
    median='median', q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75))
curve['si'] = [interval.mid for interval in curve.index]

estimate_colors = {'local median': '#0072B2', 'same share': '#E69F00', 'best agreement': '#CC79A7'}
estimate_values = dict(zip(estimate_colors, three))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))

ax1.fill_between(curve['si'], curve['q1'], curve['q3'], color='#c9b3e6', lw=0,
                 label='middle half of neurons (IQR per bin)')
ax1.plot(curve['si'], curve['median'], color='#6d21bd', lw=1.0, label='typical Phi (median per bin)')
ax1.plot([SI_CUTOFF, SI_CUTOFF], [0, phi_cutoff], color='k', lw=0.8, ls='--')
ax1.plot([0, SI_CUTOFF], [phi_cutoff, phi_cutoff], color='k', lw=0.8, ls='--')
ax1.scatter([SI_CUTOFF], [phi_cutoff], color='k', zorder=3, s=18)
ax1.annotate(f"SI {SI_CUTOFF} \u2192 Phi {phi_cutoff}", (SI_CUTOFF, phi_cutoff),
             xytext=(8, -14), textcoords='offset points', fontsize=8)
ax1.set_xlim(0, 0.3)
ax1.set_ylim(0, 0.7)
ax1.set_xlabel('SI (SFC cut)')
ax1.set_ylabel('Phi (same cut)')
ax1.set_title(f'Typical Phi at each SI (steps of {step})', fontsize=9)
ax1.legend(frameon=False, fontsize=7, loc='lower right')

zoom_from = 0.30                          # for the plot only: lower values are counted, not drawn
ax2.hist(phi[near], bins=np.arange(zoom_from, phi[near].max() + 0.002, 0.002), color='#bdbdbd')
ax2.set_xlim(zoom_from, None)
ax2.text(0.02, 0.55, f'{(phi[near] < zoom_from).sum()} neurons with Phi < {zoom_from} not shown',
         transform=ax2.transAxes, fontsize=7, color='0.4')
for name, value in estimate_values.items():
    ax2.axvline(value, color=estimate_colors[name], lw=1.2,
                label=f'{name} {value:.3f}' + ('' if name == 'local median' else ' (all neurons)'))
ax2.axvline(phi_cutoff, color='k', lw=0.8, ls='--', label=f'recommended {phi_cutoff}')
ax2.set_xlabel(f'Phi of the {n_near} neurons with SI within {WINDOW} of {SI_CUTOFF}')
ax2.set_ylabel('Neurons')
ax2.set_title(f'Recommended Phi cutoff {phi_cutoff}', fontsize=9)
ax2.legend(frameon=False, fontsize=7, loc='upper left')

for ax in (ax1, ax2):
    ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(_out_dir / "phi_threshold.svg")
print(f"\nFigure saved to {_out_dir / 'phi_threshold.svg'}")
plt.show()
