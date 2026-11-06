
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from methods.figure_text import use_arial_8
use_arial_8()   # every text in Arial 8 pt when a figure is saved
from config import METHODS_DIR, NEURON_TABLE_FTR, OUTPUT_DIR
sys.path.insert(0, str(METHODS_DIR))
import copy
from methods_all import *
import os
import xgboost as xgb
from scipy.stats import iqr, pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
import mplcursors

#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
nodesG=nodesG.dropna(subset=['dend_correct','axon_correct','super_class','primary_type'])
allowed_values = ['optic', 'central',
                  'visual_projection', 'visual_centrifugal']
nodesG = nodesG[nodesG['super_class'].isin(allowed_values)]

#%%
df_f = nodesG[['super_class',
               'length_nm', 'area_nm', 'size_nm',
               'out_synapses_count', 'in_synapses_count',
               'out_partners', 'in_partners', 'leafs',
               'SI']]
df_tmp = df_f.replace([np.inf, -np.inf], pd.NA)

# 2. Rows that will be removed (have at least one NA after replacement)
mask_bad = df_tmp.isna().any(axis=1)

# 3. Save the original rows that will be removed
rows_removed = df_f[mask_bad].copy()        # full rows
removed_indices = rows_removed.index.tolist()  # just the indices if you want

# 4. Now actually clean df_f as you intended
df_f = df_tmp.dropna()

#%%
df_f = df_f.replace([float('inf'), -float('inf')], pd.NA).dropna()

df_f_f = copy.deepcopy(df_f)
# %%

df_all = df_f_f.drop(columns='super_class')
#%%
#%%


# Step 4: Plot the barplot
custom_palette = {
    'visual projection': '#D5A848',
    'central': '#F9574E',
    'optic': '#F4D826',
    'visual centrifugal': '#44733B',
    'all': '#7F7F7F'
}

#%%



# %% 1. Linear Regression on All Data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

results_lin_all = {}

X = df_all.drop(columns=['SI'])
y = df_all['SI']
X = X.select_dtypes(include='number')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r_squared = model.score(X_test, y_test)
results_lin_all['all'] = r_squared

print(f'R² score: {r_squared:.3f}')
#%% Figure 2 Panel: pearson per class

# Prepare data
X = df_all.drop(columns=['SI']).select_dtypes(include='number')
y = df_all['SI']

# Fit model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Compute correlation
r, p = pearsonr(y, y_pred)
print(f"Pearson r: {r:.3f}, p = {p:.3g}")


#%%pearson per class

# Drop missing SI
df_f_f = df_f_f.dropna(subset=['SI'])

# Custom colors
custom_palette = {
    'visual_projection': '#D5A848',
    'central': '#F9574E',
    'optic': '#F4D826',
    'visual_centrifugal': '#44733B'
}


#%% Figure 2 Panel: model fit comparison


X = df_all.drop(columns=['SI'])
y = df_all['SI']
X = X.select_dtypes(include='number')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("RF R²:", r2_score(y_test, rf_pred))
'''
# XGBoost
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, objective='reg:squarederror', random_state=42)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
print("XGB R²:", r2_score(y_test, xgb_pred))
'''
# %% 2. Linear Regression per Superclass

df_f_f = df_f_f.dropna(subset=['SI'])
results_lin_perclass = {}

# ---------- GLOBAL LINEAR REGRESSION ----------
X_global = df_f_f.drop(columns=['SI', 'super_class']).select_dtypes(include='number')
y_global = df_f_f['SI']

if len(X_global) >= 10 and X_global.shape[1] > 0:
    Xg_train, Xg_test, yg_train, yg_test = train_test_split(X_global, y_global, test_size=0.2, random_state=42)
    global_model = LinearRegression()
    global_model.fit(Xg_train, yg_train)
    global_r2 = global_model.score(Xg_test, yg_test)
    print(f"\nR² score on all neurons (Linear Regression): {global_r2:.3f}")
else:
    print("\nNot enough global data or numeric features for global model.")
    global_r2 = None

# ---------- PER super_class ----------
for super_class in df_f_f['super_class'].unique():
    print(f"\nRunning Linear Regression for super_class: {super_class}")
    df_sub = df_f_f[df_f_f['super_class'] == super_class].copy().drop(columns=['super_class'])
    X = df_sub.drop(columns=['SI']).select_dtypes(include='number')
    y = df_sub['SI']

    if len(X) < 10 or X.shape[1] == 0:
        print("  Skipped (not enough data or no numeric features).")
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    results_lin_perclass[super_class] = r2
    print(f"  R² score: {r2:.3f}")

# ---------- SUMMARY ----------
print("\nSummary of R² scores per super_class (Linear):")
for k, v in results_lin_perclass.items():
    print(f"  {k}: {v:.3f}")

if global_r2 is not None:
    print(f"\nR² score on all data (Linear Regression): {global_r2:.3f}")

# %% 3. Random Forest per Superclass (with model storage)

df_f_f = df_f_f.dropna(subset=['SI'])
results_rf_perclass = {}

# NEW: dictionaries to store models and their feature names
models_rf_perclass = {}
rf_columns_perclass = {}

for super_class in df_f_f['super_class'].unique():
    print(f"\nRunning Random Forest for super_class: {super_class}")
    df_sub = df_f_f[df_f_f['super_class'] == super_class].copy().drop(columns=['super_class'])
    X = df_sub.drop(columns=['SI']).select_dtypes(include='number')
    y = df_sub['SI']

    if len(X) < 10 or X.shape[1] == 0:
        print(f"  Skipped (not enough data or no numeric features).")
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Store results and model
    results_rf_perclass[super_class] = model.score(X_test, y_test)
    models_rf_perclass[super_class] = model
    rf_columns_perclass[super_class] = X.columns

    print(f"  R² score: {results_rf_perclass[super_class]:.3f}")


# %% 4. Random Forest on All Data

results_rf_all = {}

df_all_rf = df_all.dropna(subset=['SI'])
X = df_all_rf.drop(columns=['SI']).select_dtypes(include='number')
y = df_all_rf['SI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_all = RandomForestRegressor(n_estimators=100, random_state=42)
model_all.fit(X_train, y_train)
y_pred = model_all.predict(X_test)


r2_all = model_all.score(X_test, y_test)
results_rf_all['all'] = r2_all

print(f"\nR² score on all data (Random Forest): {r2_all:.3f}")

#%% Figure 2 Panel: regplot with tree variance

# -------------------- Collect per-tree predictions --------------------
# Each tree’s prediction on the test set
tree_preds = np.stack([tree.predict(X_test) for tree in model_all.estimators_], axis=1)

# Compute variance (std) across trees for each sample
tree_std_per_sample = tree_preds.std(axis=1)

# -------------------- Bin by actual SI --------------------
bins = np.linspace(y_test.min(), y_test.max(), 41)
bin_indices = np.digitize(y_test, bins)

df_bin = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred,
    'tree_std': tree_std_per_sample,
    'bin': bin_indices
})

# Compute bin-wise mean actual, mean predicted, and mean tree std
bin_stats = df_bin.groupby('bin').agg({
    'actual': 'mean',
    'predicted': 'mean',
    'tree_std': 'mean'
}).reset_index(drop=True)

# -------------------- Plot setup --------------------
fig, ax1 = plt.subplots(figsize=(1.2, 1.0))
ax2 = ax1.twinx()

# KDE on secondary y-axis
sns.kdeplot(
    data=nodesG['SI'],
    ax=ax2,
    color='blue',
    linewidth=0.25,
    fill=True,
    alpha=0.3
)

# Plot mean prediction with tree-based error bars
ax1.errorbar(
    x=bin_stats['actual'],
    y=bin_stats['predicted'],
    yerr=bin_stats['tree_std'],
    fmt='o',
    markersize=0.02,
    elinewidth=0.1,
    capsize=0.0,
    color='black'
)

# -------------------- Main axis formatting --------------------
ax1.set_xticks([0, 0.3, 0.6])
ax1.set_xticklabels([0, 0.3, 0.6], fontsize=4)
ax1.set_yticks([0, 0.25, 0.5])
ax1.set_yticklabels([0, 0.25, 0.5], fontsize=4)
ax1.set_xlabel('SI', fontsize=4)
ax1.set_ylabel('Predicted SI', fontsize=4)
ax1.set_xlim(0, 0.6)
ax1.grid(False)
sns.despine(ax=ax1, right=False, top=True)

# -------------------- KDE axis formatting --------------------
ax2.set_ylabel('Density', fontsize=4)
ax2.tick_params(axis='y', labelsize=4, width=0.3)
ax2.grid(False)

# -------------------- Linear regression line --------------------
model_lin = LinearRegression()
model_lin.fit(y_test.values.reshape(-1, 1), y_pred)
x_line = np.linspace(0, 0.6, 100).reshape(-1, 1)
y_line = model_lin.predict(x_line)
ax1.plot(x_line, y_line, color='red', linewidth=0.2, linestyle='--')

# -------------------- Styling --------------------
for spine in ax1.spines.values():
    spine.set_linewidth(0.3)
for spine in ax2.spines.values():
    spine.set_linewidth(0.3)

ax1.tick_params(axis='both', width=0.3)
ax2.tick_params(axis='both', width=0.3)

# -------------------- Save figure --------------------
_out_dir = OUTPUT_DIR / "fig2" / "feat_rf_and_lr_models"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(_out_dir / "regplot_with_tree_variance_princeton.svg")
plt.show()



#%%

r2_score(y_test, y_pred)
#%%
aa=pd.concat([pd.DataFrame(y_test).reset_index(drop=True),pd.DataFrame(y_pred).reset_index(drop=True)],axis=1).corr()
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Custom palette for 4 superclasses
custom_palette = {
    'visual_projection': '#D5A848',
    'central': '#F9574E',
    'optic': '#F4D826',
    'visual_centrifugal': '#44733B'
}
model_palette = {
    'Linear Regression': 'gray',
    'Random Forest': 'steelblue',
}

# Global colors for three models in one plot
global_colors = {
    'Linear Regression': 'gray',
    'Random Forest': 'steelblue',
}

# Prepare per-class data points
point_data = []
for sc in results_lin_perclass:
    if sc in results_rf_perclass:
        point_data.append({'Group': sc, 'Model': 'Linear Regression', 'R² Score': results_lin_perclass[sc]})
        point_data.append({'Group': sc, 'Model': 'Random Forest', 'R² Score': results_rf_perclass[sc]})

# Prepare global model data
point_data.append({'Group': 'Global', 'Model': 'Linear Regression', 'R² Score': results_lin_all.get('all', None)})
point_data.append({'Group': 'Global', 'Model': 'Random Forest', 'R² Score': results_rf_all.get('all', None)})

# Convert to DataFrame
point_df = pd.DataFrame(point_data)

# Plot
plt.figure(figsize=(8, 5))
# Plot per-class
sns.stripplot(
    data=point_df[point_df['Group'] != 'Global'],
    x='Model',
    y='R² Score',
    hue='Group',
    palette=custom_palette,
    dodge=True,
    size=8
)

# Plot global with manual color mapping
for model, color in global_colors.items():
    score = point_df[(point_df['Group'] == 'Global') & (point_df['Model'] == model)]['R² Score'].values
    if len(score) > 0:
        plt.scatter(
            x=['Linear Regression' if 'Linear' in model else 'Random Forest'] * len(score),
            y=score,
            label=f'Global: {model}',
            color=color,
            edgecolor='black',
            s=120,
            marker='D',
            zorder=3
        )

plt.title('R² Scores by Model')
plt.ylim(0, 1)
plt.ylabel('R² Score')
plt.xlabel('Model')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(title='Super Class / Global Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
#%%

import pandas as pd
import matplotlib.pyplot as plt

# Custom palette for 4 superclasses
custom_palette = {
    'visual_projection': '#D5A848',
    'central': '#F9574E',
    'optic': '#F4D826',
    'visual_centrifugal': '#44733B'
}

# Desired group order
group_order = ['Global', 'optic', 'central', 'visual_projection', 'visual_centrifugal']
model_order = ['Linear Regression', 'Random Forest']

# Collect point data
point_data = []

# Per-class data
for sc in ['optic', 'central', 'visual_projection', 'visual_centrifugal']:
    if sc in results_lin_perclass and sc in results_rf_perclass:
        point_data.append({'Group': sc, 'Model': 'Linear Regression', 'R² Score': results_lin_perclass[sc]})
        point_data.append({'Group': sc, 'Model': 'Random Forest', 'R² Score': results_rf_perclass[sc]})

# Global data (single black line)
point_data.append({'Group': 'Global', 'Model': 'Linear Regression', 'R² Score': results_lin_all.get('all', None)})
point_data.append({'Group': 'Global', 'Model': 'Random Forest', 'R² Score': results_rf_all.get('all', None)})

# Convert to DataFrame
point_df = pd.DataFrame(point_data)

# Ensure ordering
point_df['Group'] = pd.Categorical(point_df['Group'], categories=group_order, ordered=True)
point_df['Model'] = pd.Categorical(point_df['Model'], categories=model_order, ordered=True)
point_df = point_df.sort_values(['Group', 'Model'])

# Plot setup
plt.figure(figsize=(1, 2))
x_pos = [0, 1]  # 0 = LR, 1 = RF

# Draw lines per group
for group in group_order:
    sub = point_df[point_df['Group'] == group]
    if sub.shape[0] == 2:
        y_vals = sub['R² Score'].values
        color = 'black' if group == 'Global' else custom_palette.get(group, 'gray')
        plt.plot(x_pos, y_vals, marker='o', markersize=2.5, linewidth=1, label=group, color=color)
plt.xticks(size=8,rotation=90)
plt.yticks(size=8)

# Aesthetics
plt.xticks(x_pos, model_order)
plt.xlim(-0.5, 1.5)
plt.ylim(0, 1)
plt.ylabel('R² Score')
plt.xlabel('Model')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.grid(False)

#plt.title('R² Score per Group: Linear Regression vs. Random Forest')
plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
sns.despine(right=True,top=True)

plt.savefig(_out_dir / "rsquare_model_comp_princeton.svg")
plt.show()

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Custom palette for 4 superclasses
custom_palette = {
    'visual_projection': '#D5A848',
    'central': '#F9574E',
    'optic': '#F4D826',
    'visual_centrifugal': '#44733B'
}
model_palette = {
    'Linear Regression': 'orange',
    'Random Forest': 'steelblue',
}

# Global colors for three models in one plot
global_colors = {
    'Linear Regression': 'gray',
    'Random Forest': 'steelblue',
}

# Prepare per-class and global data
point_data = []

# Global scores
point_data.append({'Group': 'Global', 'Model': 'Linear Regression', 'R² Score': results_lin_all.get('all', None)})
point_data.append({'Group': 'Global', 'Model': 'Random Forest', 'R² Score': results_rf_all.get('all', None)})

# Per-class scores
for sc in ['optic', 'central', 'visual_projection', 'visual_centrifugal']:
    if sc in results_lin_perclass and sc in results_rf_perclass:
        point_data.append({'Group': sc, 'Model': 'Linear Regression', 'R² Score': results_lin_perclass[sc]})
        point_data.append({'Group': sc, 'Model': 'Random Forest', 'R² Score': results_rf_perclass[sc]})

# Convert to DataFrame
point_df = pd.DataFrame(point_data)

# Define desired group and model order
group_order = ['Global', 'optic', 'central', 'visual_projection', 'visual_centrifugal']
model_order = ['Linear Regression', 'Random Forest']

# Ensure categorical ordering
point_df['Group'] = pd.Categorical(point_df['Group'], categories=group_order, ordered=True)
point_df['Model'] = pd.Categorical(point_df['Model'], categories=model_order, ordered=True)

# Plot
plt.figure(figsize=(10, 5))
sns.stripplot(
    data=point_df,
    x='Group',
    y='R² Score',
    hue='Model',
    dodge=True,
    size=8,
    palette=model_palette,
    edgecolor='black'
)

plt.title('R² Scores per Group and Model')
plt.ylim(0, 1)
plt.ylabel('R² Score')
plt.xlabel('Group')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% Feature Importances with Dendrogram Ordering
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
ordered_features = [
    'in_partners',
    'out_partners',
    'out_synapses_count',
    'size_nm',
    'area_nm',
    'length_nm',
    'leafs',
    'in_synapses_count',
]
# Define palette (with distinct colors for globals)
custom_palette = {
    'visual_projection': '#D5A848',
    'central': '#F9574E',
    'optic': '#F4D826',
    'visual_centrifugal': '#44733B',
    'Global': 'black',
}

# Collect feature importances
feature_importance_data = []

# Per-superclass importances
for super_class, model in models_rf_perclass.items():
    features = rf_columns_perclass[super_class]
    importances = model.feature_importances_
    for feat, imp in zip(features, importances):
        feature_importance_data.append({
            'Feature': feat,
            'Importance': imp,
            'Source': super_class
        })

# Global (Unencoded)
for feat, imp in zip(X.columns, model_all.feature_importances_):
    feature_importance_data.append({
        'Feature': feat,
        'Importance': imp,
        'Source': 'Global'
    })

# Create DataFrame
feat_imp_df = pd.DataFrame(feature_importance_data)

# Remove dummy-encoded features for clarity
feat_imp_df = feat_imp_df[~feat_imp_df['Feature'].str.startswith('super_class_')]

# Apply reversed dendrogram order (to match heatmap + other plots)
feat_imp_df['Feature'] = pd.Categorical(
    feat_imp_df['Feature'],
    categories=ordered_features[::-1],
    ordered=True
)
feat_imp_df = feat_imp_df.sort_values('Feature')

# Barplot
plt.figure(figsize=(14, 6))
sns.barplot(
    data=feat_imp_df,
    x='Feature',
    y='Importance',
    hue='Source',
    order=ordered_features[::-1],
    palette=custom_palette
)
plt.xticks(rotation=90, ha='right')
plt.title('Feature Importances: Superclasses and Global Models (Dendrogram-Aligned)', fontsize=16)
plt.xlabel('Feature', fontsize=13)
plt.ylabel('Importance', fontsize=13)
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.savefig(_out_dir / "feat_import_bar_princeton.svg")
plt.show()
