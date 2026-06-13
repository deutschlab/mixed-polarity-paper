
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

# ✅ Apply reversed dendrogram order (to match heatmap + other plots)
feat_imp_df['Feature'] = pd.Categorical(
    feat_imp_df['Feature'],
    categories=ordered_features[::-1],
    ordered=True
)
feat_imp_df = feat_imp_df.sort_values('Feature')

# ✅ Barplot
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

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

# Apply dendrogram feature ordering
feat_imp_df['Feature'] = pd.Categorical(
    feat_imp_df['Feature'], categories=ordered_features[::-1], ordered=True
)
feat_imp_df = feat_imp_df.sort_values('Feature')

# Plot as lineplot
plt.figure(figsize=(16, 6))
sns.lineplot(
    data=feat_imp_df,
    x='Feature',
    y='Importance',
    hue='Source',
    palette=custom_palette,
    marker='o'
)
plt.xticks(rotation=45, ha='right')
plt.title('Feature Importances by Source (Ordered by Dendrogram)', fontsize=16)
plt.xlabel('Feature', fontsize=13)
plt.ylabel('Importance', fontsize=13)
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.show()
#%%



#%%
# df_all=pd.read_feather(r'C:\Users\user\organised_work\data\783\generated\post_processing_data\neuron_data_full.ftr')
#%%


'''

sns.clustermap(df_all[[  
            'length_nm', 'area_nm', 'size_nm', 
            'out_synapses_count', 'in_synapses_count', 
            'out_partners', 'in_partners', 'leafs', 'max_out_degree', 'max_in_degree',
            'mean_in_degree', 'mean_out_degree', 'partners_ratio', 'max_degree_ratio', 'mean_degree_ratio',
             'mean_out_synapses_per_partner', 'mean_in_synapses_per_partner',
             'mean_synapses_per_partner_ratio',
       ]].corr(), cmap='coolwarm',annot=True, figsize=(18, 14))


'''


sns.clustermap(df_all[[
    'length_nm', 'area_nm', 'size_nm',
    'out_synapses_count', 'in_synapses_count',
    'out_partners', 'in_partners', 'leafs', 'max_out_degree', 'max_in_degree', 'mean_in_degree', 'mean_out_degree'
]].corr(), cmap='coolwarm', annot=True, figsize=(18, 14))


# %%
# Create a cursor


for super_class in df_f_f['super_class'].unique():
    print(f"Processing super_class: {super_class}")

    # Filter data for the current super_class
    df_subset = df_f_f[df_f_f['super_class'] ==
                       super_class].drop(columns='super_class')
    df_subset = df_subset.drop(columns='SI')
    # Compute the correlation matrix
    corr_matrix = df_subset.corr()

    # Generate a clustermap for the current super_class
    sns.clustermap(corr_matrix, annot=True, cmap='coolwarm', figsize=(14, 10))
    plt.title(f"Correlation Clustermap for super_class: {super_class}")
    plt.show()

# %%

# %%
df_f_f = df_f_f.replace([float('inf'), -float('inf')], pd.NA).dropna()
# %%


# Assuming 'df_f_f' is your DataFrame and 'super_class' is one of the columns
features = ['length_nm', 'area_nm', 'size_nm', 'out_synapses_count',
            'in_synapses_count', 'out_partners', 'in_partners',
            'leafs', 'max_out_degree', 'max_in_degree', 'mean_in_degree',
            'mean_out_degree', 'SI']

# Loop over each feature
for feature in features:
    # Step 1: Calculate the 2.5% and 97.5% quantiles
    lower_quantile = df_f_f[feature].quantile(0.025)
    upper_quantile = df_f_f[feature].quantile(0.975)

    # Step 2: Filter the data to keep only the middle 95%
    df_filtered = df_f_f[(df_f_f[feature] >= lower_quantile)
                         & (df_f_f[feature] <= upper_quantile)]

    # Step 3: Bin the filtered feature into 100 bins
    df_filtered['binned_' + feature] = pd.cut(df_filtered[feature], bins=100)

    # Convert 'binned_' columns to numeric using the left bound of the bin
    df_filtered['binned_' + feature] = df_filtered['binned_' +
                                                   feature].apply(lambda x: x.left)

    # Step 4: Create a plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x='binned_' + feature, y='SI', hue='super_class',
                 estimator='mean', lw=2, palette=custom_palette)

    # Title and labels
    plt.title(f'Line plot for {feature} vs SI (Middle 95% Only)')
    plt.xlabel(f'{feature} Binned (Middle 95%)')
    plt.ylabel('SI')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show plot
    plt.tight_layout()
    plt.show()


# %%
df_all = df_all.drop('SI', axis=1)

# %%

# Remove the 'SI' feature from df_all

# Define a custom color palette for the super_class
custom_palette = {
    'visual_projection': '#D5A848',
    'central': '#F9574E',
    'optic': '#F4D826',
    'visual_centrifugal': '#44733B',
    'all': 'grey'  # Add a neutral color for the "all" row
}

# Normalize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(df_all)

# Compute the pairwise distances between features
# Transpose to cluster features
distance_matrix = pdist(data_normalized.T, metric='euclidean')

# Perform hierarchical clustering
linkage_result = linkage(distance_matrix, method='ward')

# Create a dendrogram to determine the feature order
dendro = dendrogram(linkage_result, labels=df_all.columns, no_plot=True)
feature_order = dendro['ivl']  # Get the feature order from the dendrogram

# Create correlation_long (ensure it doesn't include 'SI')
correlation_long = correlation_df.melt(
    id_vars='super_class', var_name='feature', value_name='correlation')
# Exclude 'SI'
correlation_long = correlation_long[correlation_long['feature'] != 'SI']

# Reorder the melted DataFrame according to the dendrogram feature order
correlation_long['feature'] = pd.Categorical(
    correlation_long['feature'], categories=feature_order, ordered=True)
correlation_long = correlation_long.sort_values(by='feature')

# Customization parameters
bar_width = 1            # Bar width
linewidth = 1.2          # Line width for bars
title_fontsize = 18      # Font size for the title
label_fontsize = 14      # Font size for axis labels
ticks_fontsize = 12      # Font size for tick labels
legend_fontsize = 12     # Font size for the legend

# Set up a grid for the bar plot and dendrogram
fig, ax = plt.subplots(2, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [
                       4, 2]})  # Increase the ratio for the dendrogram

# Plot the bar plot on the top subplot
sns.barplot(
    data=correlation_long,
    x='feature',
    y='correlation',
    hue='super_class',
    palette=custom_palette,
    dodge=True,
    ci=None,
    linewidth=linewidth,
    ax=ax[0]
)

# Customize the bar plot
ax[0].set_title('Feature Correlations with SI by Superclass',
                fontsize=title_fontsize)
ax[0].set_xlabel('')  # Leave blank as the dendrogram will use x-axis labels
ax[0].set_ylabel('Correlation', fontsize=label_fontsize)
ax[0].tick_params(axis='x', labelsize=10, rotation=90)
ax[0].tick_params(axis='y', labelsize=ticks_fontsize)
ax[0].legend(title='Super Class', fontsize=legend_fontsize)
for i in range(len(feature_order) - 1):  # Add one less line than the number of features
    ax[0].axvline(x=i + 0.5, color='grey', linestyle='--', linewidth=0.8)

# Plot the dendrogram on the bottom subplot (upside down)
dendrogram(
    linkage_result,
    labels=feature_order,
    ax=ax[1],
    color_threshold=0,
    orientation='top',  # Normal orientation; invert y-axis later for upside-down
    above_threshold_color='grey',
    link_color_func=lambda k: 'grey'
)

# Flip the dendrogram upside down
ax[1].invert_yaxis()
ax[1].set_xticks([])  # Remove x-tick labels
ax[1].set_yticks([])  # Remove y-tick labels
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['left'].set_visible(False)
ax[1].spines['bottom'].set_visible(False)

# Adjust dendrogram line widths
for line in ax[1].collections:
    line.set_linewidth(3.0)  # Adjust line width

# Adjust layout for clarity
plt.tight_layout()
plt.show()
# %%
allowed_values = ['optic', 'visual_projection', 'visual_centrifugal']

df_a2 = df[['SI', 'in_partners', 'super_class', 'primary_type']]
df_a2 = df_a2[df_a2['super_class'].isin(allowed_values)]

# %%
# Add mean SI per primary_type
df_a2['mean_SI_per_primary_type'] = df_a2.groupby(
    'primary_type')['SI'].transform('mean')

# Add count per primary_type
df_a2['count_per_primary_type'] = df_a2.groupby(
    'primary_type')['SI'].transform('count')

# Add mean in_partners per primary_type
df_a2['mean_in_partner_per_primary_type'] = df_a2.groupby(
    'primary_type')['in_partners'].transform('mean')
# %%
summary_df = df_a2.groupby(['primary_type', 'super_class']).agg(
    mean_SI=('SI', 'mean'),
    mean_in_partners=('in_partners', 'mean'),
    count=('SI', 'count')
).reset_index()


# %%

# Define the custom color palette
custom_palette = {
    'visual_projection': '#D5A848',
    'central': '#F9574E',
    'optic': '#F4D826',
    'visual_centrifugal': '#44733B',
}

# Scatterplot of SI vs. in_partners with super_class as hue
plt.figure(figsize=(14, 8))
ax = sns.scatterplot(
    data=df_a2,
    x='SI',
    y='in_partners',
    hue='super_class',
    palette=custom_palette,
    s=20  # Adjust point size for better visibility
)

# ax.set_yscale('log')
# ax.set_xscale('log')


# Customize the plot
plt.title('Scatterplot of SI vs. In-Partners', fontsize=16)
plt.xlabel('SI', fontsize=14)
plt.ylabel('In-Partners', fontsize=14)
plt.legend(title='Super Class', fontsize=12, title_fontsize=14)
plt.tight_layout()

# Show the plot
plt.show()

# %%
# Scatterplot of SI vs. in_partners with super_class as hue
plt.figure(figsize=(14, 8))
ax = sns.scatterplot(
    data=df_a2,
    x='SI',
    y='in_partners',
    hue='super_class',
    palette=custom_palette,
    size='count_per_primary_type',
    sizes=(15, 400),
    alpha=1,
    edgecolor='black',
    linewidth=0.5,
    legend=False  # Disable automatic legend
)

# Add custom size legend
size_labels = ['1', '10', '100', '1000']
size_values = [15, 100, 250, 400]  # Map these to sizes in the plot
for size, label in zip(size_values, size_labels):
    plt.scatter([], [], s=size, color='grey', edgecolor='black', label=label)

# Add the custom legend
plt.legend(
    title="Count",
    fontsize=10,
    title_fontsize=12,
    loc="lower right",
    frameon=True
)
# Customize the plot
plt.title(
    'Scatterplot of SI vs. In-Partners with Count-Based Gaussian Aura', fontsize=16)
plt.xlabel('SI', fontsize=14)
plt.ylabel('In-Partners', fontsize=14)
plt.legend(title='Super Class', fontsize=12, title_fontsize=14,
           loc='upper left', bbox_to_anchor=(1, 1))  # Adjust legend placement
plt.tight_layout()

# Show the plot
plt.show()

# %%
# Scatterplot
plt.figure(figsize=(14, 8))
ax = sns.scatterplot(
    data=df_a2,
    x='SI',
    y='in_partners',
    hue='super_class',
    palette=custom_palette,
    size='count_per_primary_type',
    sizes=(15, 400),
    alpha=1,
    edgecolor='black',
    linewidth=0.5
)

# Extract hue legend (color labels)
hue_handles, hue_labels = ax.get_legend_handles_labels()
hue_legend = plt.legend(
    # Restrict to hue-related handles
    handles=hue_handles[:len(custom_palette)],
    labels=hue_labels[:len(custom_palette)],  # Restrict to hue-related labels
    title="Super Class",
    fontsize=10,
    title_fontsize=12,
    loc='upper left',
    bbox_to_anchor=(1, 1)  # Place the hue legend outside the plot
)

# Add the hue legend explicitly
ax.add_artist(hue_legend)

# Manually create the size legend (circle sizes)

# Add the size legend
plt.legend(
    title="Count",
    fontsize=10,
    title_fontsize=12,
    loc="lower right",  # Place the size legend
    frameon=True  # Add a box around the legend
)

# Customize the plot
plt.title('Scatterplot of SI vs. In-Partners with Hue and Size Legends', fontsize=16)
plt.xlabel('SI', fontsize=14)
plt.ylabel('In-Partners', fontsize=14)
plt.tight_layout()

# Show the plot
plt.show()
# %%
plt.figure(figsize=(14, 8))
ax = sns.scatterplot(
    data=df_a2,
    x='SI',
    y='in_partners',
    hue='super_class',
    palette=custom_palette,
    size='count_per_primary_type',
    sizes=(15, 400),
    alpha=1,
    edgecolor='black',
    linewidth=0.5
)

# Extract hue legend (color labels)
hue_handles, hue_labels = ax.get_legend_handles_labels()
hue_legend = plt.legend(
    # Restrict to hue-related handles
    handles=hue_handles[:len(custom_palette)],
    labels=hue_labels[:len(custom_palette)],  # Restrict to hue-related labels
    title="Super Class",
    fontsize=10,
    title_fontsize=12,
    loc='upper left',
    bbox_to_anchor=(1, 1)  # Place the hue legend outside the plot
)

# Add the hue legend explicitly
ax.add_artist(hue_legend)

# Manually specify circle sizes and labels for the size legend
size_labels = ['1', '10', '100', '1000']  # Desired labels
size_values = [15, 100, 250, 400]  # Corresponding circle sizes
for size, label in zip(size_values, size_labels):
    plt.scatter([], [], s=size, color='grey', edgecolor='black', label=label)

# Add the size legend
plt.legend(
    title="Count",
    fontsize=10,
    title_fontsize=12,
    loc="lower right",  # Place the size legend
    frameon=True  # Add a box around the legend
)

# Customize the plot
plt.title(
    'Scatterplot of SI vs. In-Partners with Hue and Custom Size Legends', fontsize=16)
plt.xlabel('SI', fontsize=14)
plt.ylabel('In-Partners', fontsize=14)
plt.tight_layout()

# Show the plot
plt.show()

# %%
# Specify desired sizes for the legend and dots

# %%
# Specify desired sizes for the legend and dots
specified_sizes = {
    1: 10,     # Count value 1 -> size 10
    10: 70,    # Count value 10 -> size 40
    50: 140,   # Count value 50 -> size 120
    100: 250,  # Count value 100 -> size 250
    800: 600  # Count value 1000 -> size 700
}

# Extract the minimum and maximum of `count_per_primary_type` in your data
data_min = df_a2['count_per_primary_type'].min()
data_max = df_a2['count_per_primary_type'].max()

# Dynamically compute the size mapping for all data values in `count_per_primary_type`
# Range of sizes from specified_sizes
size_range = list(specified_sizes.values())
# Corresponding count values from specified_sizes
data_range = list(specified_sizes.keys())
df_a2['size_mapped'] = np.interp(
    df_a2['count_per_primary_type'],  # Map data values
    [data_min, data_max],            # Data range (min, max)
    [size_range[0], size_range[-1]]  # Size range (min, max)
)

# Define the desired order of superclasses
desired_order = ['optic', 'visual_projection', 'visual_centrifugal']

# Adjust the order of categories in the data and palette
custom_palette_ordered = {k: custom_palette[k] for k in desired_order}
df_a2['super_class'] = pd.Categorical(
    df_a2['super_class'],
    categories=desired_order,
    ordered=True
)
# %%


# %%

# %%
cursor = mplcursors.cursor(hover=True)

# Specify desired sizes for the legend and dots
specified_sizes = {
    1: 10,     # Count value 1 -> size 10
    10: 70,    # Count value 10 -> size 70
    50: 140,   # Count value 50 -> size 140
    100: 250,  # Count value 100 -> size 250
    800: 600   # Count value 800 -> size 600
}

# Extract the minimum and maximum of `count_per_primary_type` in your data
data_min = df_a2['count_per_primary_type'].min()
data_max = df_a2['count_per_primary_type'].max()

# Dynamically compute the size mapping for all data values in `count_per_primary_type`
# Range of sizes from specified_sizes
size_range = list(specified_sizes.values())
# Corresponding count values from specified_sizes
data_range = list(specified_sizes.keys())
df_a2['size_mapped'] = np.interp(
    df_a2['count_per_primary_type'],  # Map data values
    [data_min, data_max],            # Data range (min, max)
    [size_range[0], size_range[-1]]  # Size range (min, max)
)

# Define the desired order of superclasses
desired_order = ['optic', 'visual_projection', 'visual_centrifugal']

# Adjust the order of categories in the data and palette
custom_palette_ordered = {k: custom_palette[k] for k in desired_order}
df_a2['super_class'] = pd.Categorical(
    df_a2['super_class'],
    categories=desired_order,
    ordered=True
)

# Scatterplot with mapped sizes
plt.figure(figsize=(14, 8))
ax = sns.scatterplot(
    # Ensure `super_class` order is applied
    data=df_a2.sort_values('super_class'),
    x='SI',
    y='in_partners',
    hue='super_class',
    palette=custom_palette_ordered,
    size='size_mapped',
    sizes=(size_range[0], size_range[-1]),  # Apply the same size range
    alpha=1,
    edgecolor='black',
    linewidth=0.5,
    legend=False  # Disable automatic legend
)

# Explicitly create the hue legend for super_class
handles = [plt.Line2D([0], [0], marker='o', color=color, markersize=10, linestyle='')
           for color in custom_palette_ordered.values()]
labels = list(custom_palette_ordered.keys())

hue_legend = plt.legend(
    handles=handles,
    labels=labels,
    title="Super Class",
    fontsize=12,          # Make legend font size larger
    title_fontsize=14,    # Increase title font size
    loc='upper left',
    bbox_to_anchor=(1.05, 1.1)  # Adjust position to the right of the plot
)
ax.add_artist(hue_legend)  # Add the hue legend to the plot

# Add custom size legend based on specified_sizes
for count, size in specified_sizes.items():
    # Add custom size markers
    plt.scatter([], [], s=size, color='grey', label=str(count))

size_legend = plt.legend(
    title="Count",
    fontsize=12,          # Make size legend font size larger
    title_fontsize=14,    # Increase title font size
    loc="lower right",
    frameon=True
)
ax.add_artist(size_legend)  # Add the size legend to the plot

# Add interactivity with mplcursors

# Customize the cursor tooltips
# Add interactivity with mplcursors
cursor = mplcursors.cursor(ax.collections, hover=True)

# Customize the cursor tooltips


@cursor.connect("add")
def on_add(sel):
    # Retrieve the data point index
    index = sel.index

    # Format the tooltip with data from the DataFrame
    sel.annotation.set_text(
        f"SI: {df_a2.iloc[index]['SI']:.2f}\n"
        f"In-Partners: {df_a2.iloc[index]['in_partners']}\n"
        f"Count: {df_a2.iloc[index]['count_per_primary_type']}\n"
        f"Class: {df_a2.iloc[index]['super_class']}"
    )

    # Optional: Adjust the position of the tooltip for better alignment
    sel.annotation.set_position((20, 20))  # Offset tooltip by 20px (x, y)


plt.title(
    'Scatterplot of SI vs. In-Partners with Specified Super Class Order', fontsize=16)
plt.xlabel('SI', fontsize=14)
plt.ylabel('In-Partners', fontsize=14)
plt.tight_layout()

# Show the plot
plt.show()
