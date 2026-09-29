# -*- coding: utf-8 -*-
"""
Finds the Phi value that corresponds to the SI = 0.1 cutoff.

SI and Phi both measure how separated a neuron's inputs and outputs are between
axon and dendrite, on different scales. Phi is not a fixed function of SI: at a
given SI it also depends on the neuron's balance of inputs and outputs. So the
match below holds on average, not neuron by neuron.

SI_comparisons.ftr (written by 08_alternative_split_methods.py) holds both scores
for the published SFC cut of every neuron in it (the intrinsic neurons of the
neuron table that step 08 split successfully): SFC_SI and SFC_Phi. SFC_SI is the
corrected SI (SI_updated.ftr); the script stops if it is not. On that cut the Phi
cutoff is matched to SI = 0.1 in three ways:

  1. local median: the median SFC_Phi of neurons whose SFC_SI is within 0.005 of
     0.1 (the typical Phi of a neuron right at the SI cutoff);
  2. same share: the SFC_Phi value that calls the same share of neurons mixed as
     SFC_SI < 0.1;
  3. best agreement: the SFC_Phi cutoff that agrees with SFC_SI < 0.1 on the most
     neurons.

The recommended cutoff is the median of the three, rounded to two decimals, and
should be reported with the range of the three: the three estimators differ by more
than their sampling intervals, so that range is the real uncertainty. The cutoff is
meant to be applied unchanged to every method, together with SI = 0.1, rather than
tuning a cutoff per method. Intervals come from resampling whole primary types, because
neurons of one type are not independent.

Reads SI_COMPARISONS_FTR, SI_UPDATED_FTR, and from NEURON_TABLE_FTR the
super_class, primary_type and the input and output counts of axon and dendrite
(used only to group neurons). Prints its results; writes nothing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SI_COMPARISONS_FTR, SI_UPDATED_FTR, NEURON_TABLE_FTR

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SI_CUTOFF = 0.1
WINDOW = 0.005          # SI window around the cutoff for the local median
N_BOOTSTRAP = 1000

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
    """Cutoff with n_mixed values below it (exactly, unless the two values at the boundary are tied)."""
    if not 0 < n_mixed < len(phi):
        raise ValueError("no neuron, or every neuron, is below the SI cutoff")
    s = np.sort(phi)
    return (s[n_mixed - 1] + s[n_mixed]) / 2


def best_agreement_cutoff(phi, mixed):
    """Cutoff at which (phi < cutoff) agrees with `mixed` on the most neurons.
    If several cutoffs tie, the lowest one is returned."""
    order = np.argsort(phi, kind='stable')
    s = phi[order]
    m = mixed[order].astype(float)
    n = len(s)
    k = np.arange(1, n + 1)                     # the first k are called mixed
    true_mixed = np.cumsum(m)
    # (mixed called mixed + segregated called segregated) / n
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
