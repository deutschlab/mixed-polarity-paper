import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

import os
import time
import pickle
import navis 
import sys
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (NEURON_TABLE_FTR, SYNAPSE_TABLE_FTR, PCA_TABLE_FTR,
                    PC1_TABLE_CSV, PC1_TABLE_PREDICTIONS_CSV,
                    OUTPUT_DIR, METHODS_DIR)
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *

#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
#%%
nodesG=nodesG.dropna(subset='primary_type')
a=nodesG[nodesG['super_class']=='central']
#%%
sns.kdeplot(a['SI'], cumulative=True)
plt.title('central, n=32k')
plt.axvline(0.1)
plt.axhline(0.52)
#%%
nodesG=nodesG[['neuron','super_class',
       'length_nm', 'area_nm', 'size_nm', 
       'out_synapses_count', 'in_synapses_count', 
        'out_partners',
       'in_partners', 'leafs', ]]
#%%
nodesG=nodesG[['neuron','super_class' ]]
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
#nodesG=nodesG[nodesG['super_class'].isin(['optic'])]

#%%

pcadf=pd.read_feather(PCA_TABLE_FTR)
#%%

nodesG=nodesG.merge(pcadf,on='neuron',how='left')
#%%
plt.figure(figsize=(6,2))
custom_palette = {
    'visual_projection': '#D5A848',
    'central': '#F9574E',
    'optic': '#F4D826',
    'visual_centrifugal': '#44733B'
}
sns.kdeplot(nodesG,x='PC1',hue='super_class',   palette=custom_palette)
plt.xlim(-2,5)
_out_dir = OUTPUT_DIR / "fig4" / "syntype_x_pc1"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "kde_sclass.svg")
plt.show()
#%%
syn_df=pd.read_feather(SYNAPSE_TABLE_FTR)
#%%
#syn_df=syn_df.sample(1000000)

#%%
syn_df=syn_df[[ 'pre', 'post',
       'comp', 'SI_pre', 'SI_post']]
#%%
syn_df=syn_df[(syn_df['SI_pre']>=0.1)&(syn_df['SI_post']>=0.1)]

#%%
syn_df=syn_df[syn_df['comp'].isin(['AD','AA','DD','DA'])]


#%%
syn_df_m=syn_df.merge(nodesG,left_on='pre',right_on='neuron',how='left')
#%%
del syn_df

#%%
syn_df_m=syn_df_m.merge(nodesG,left_on='post',right_on='neuron',how='left')
#%%
syn_df_m=syn_df_m.dropna(subset=['super_class_x','super_class_y'])
#%%
instrinsiclen=len(syn_df_m[(syn_df_m['super_class_x'].isin(['central','optic','visual_projection','visual_centrifugal']))&(syn_df_m['super_class_y'].isin(['central','optic','visual_projection','visual_centrifugal']))])
opticlen=len(syn_df_m[(syn_df_m['super_class_x'].isin(['optic']))&(syn_df_m['super_class_y'].isin(['optic']))])
centrallen=len(syn_df_m[(syn_df_m['super_class_x'].isin(['optic']))&(syn_df_m['super_class_y'].isin(['optic']))])
central_optic=len(syn_df_m[(syn_df_m['super_class_x'].isin(['central']))&(syn_df_m['super_class_y'].isin(['optic']))])
optic_central=len(syn_df_m[(syn_df_m['super_class_x'].isin(['optic']))&(syn_df_m['super_class_y'].isin(['central']))])
#%%

print('Intrinsic total:', instrinsiclen)
print('Optic -> Optic:', opticlen)
print('Central -> Central:', centrallen)
print('Central -> Optic:', central_optic)
print('Optic -> Central:', optic_central)
#%%
#all_df=syn_df_m[[ 'comp',       'PC1_x','PC1_y','PC2_x','PC2_y']]
all_df=syn_df_m[[ 'comp',       'PC1_x','PC1_y']]

#%%


AA_mean=all_df[all_df['comp']=='AA'].drop(columns=['comp'])
DD_mean=all_df[all_df['comp']=='DD'].drop(columns=['comp'])

all_mean=all_df.drop(columns=['comp']).mean()

all_mean.dtypes



#%%
# Calculate AA_mean and DD_mean
AA_mean = AA_mean.mean()
DD_mean = DD_mean.mean()

# Calculate differences
AA_dif = ( AA_mean -all_mean  ) / (all_mean + AA_mean)  
DD_dif = ( DD_mean-all_mean) / (all_mean + DD_mean)  

# Sort values for bar plot
AA_dif = AA_dif.sort_values()
DD_dif = DD_dif.sort_values()
#%%

plt.figure(figsize=(14,8))
sns.barplot(DD_dif)
plt.xticks(rotation=90,size=9)

#%%

plt.figure(figsize=(14,8))
sns.barplot(AA_dif)
plt.xticks(rotation=90,size=9)
#%%
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
#%%
# Function to assign colors based on suffix
def assign_color(label):
    if str(label).endswith('_y') or str(label).endswith('post'):
        return 'cyan'
    if str(label).endswith('_x') or str(label).endswith('pre'):
        return 'red'
    else:
        return 'gray'  # Default color for unexpected cases

# ---------------- AA_dif Plot ----------------
# Assign colors
AA_colors = [assign_color(str(label)) for label in AA_dif.index]

# Clean x-axis labels (remove _x and _y)
clean_labels_AA = [
    str(label).replace('_x', '').replace('_y', '').replace('_', ' ').title()
    for label in AA_dif.index
]
# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=AA_dif.index, y=AA_dif.values, palette=AA_colors)
plt.axhline(0, color='gray', linestyle='--')
plt.title('Normalized Differences for AA')
plt.xlabel('Index')
plt.ylabel('Normalized Difference [-1, 1]')
plt.xticks(ticks=range(len(clean_labels_AA)), labels=clean_labels_AA, rotation=90)

# Add legend
legend_elements = [
    Patch(facecolor='red', edgecolor='red', label='pre'),
    Patch(facecolor='cyan', edgecolor='cyan', label='post')
]
plt.legend(handles=legend_elements, title='Neuron Role', loc='upper right')

plt.tight_layout()
sns.despine(right=True,top=True)
plt.savefig(_out_dir / "synaptic_type_feature_diff_AA_filtered_features_princeton.svg")
plt.show()

# ---------------- DD_dif Plot ----------------
# Assign colors
DD_colors = [assign_color(str(label)) for label in DD_dif.index]

# Clean x-axis labels (remove _x and _y)
clean_labels_DD = [
    str(label).replace('_x', '').replace('_y', '').replace('_', ' ').title()
    for label in DD_dif.index
]
# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=DD_dif.index, y=DD_dif.values, palette=DD_colors)
plt.axhline(0, color='gray', linestyle='--')
plt.title('Normalized Differences for DD')
plt.xlabel('Index')
plt.ylabel('Normalized Difference [-1, 1]')
plt.xticks(ticks=range(len(clean_labels_DD)), labels=clean_labels_DD, rotation=90)

# Add legend
plt.legend(handles=legend_elements, title='Neuron Role', loc='upper right')
sns.despine(right=True,top=True)

plt.tight_layout()
plt.savefig(_out_dir / "synaptic_type_feature_diff_DD_filtered_features_princeton.svg")
plt.show()


#%%
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Function to assign bar colors based on suffix
def assign_color(label):
    if str(label).endswith('_y') or str(label).endswith('post'):
        return 'cyan'
    if str(label).endswith('_x') or str(label).endswith('pre'):
        return 'red'
    else:
        return 'gray'

# List of labels that should be colored blue after cleaning
blue_labels = [
    'Out Synapses Count',
    'Max Out Degree',
    'In Synapses Count',
    'Max In Degree',
    'Out Partners',
    'In Partners',
    'Mean Out Degree',
    'Mean In Degree'
]

# ---------------- AA_dif Plot ----------------
AA_colors = [assign_color(str(label)) for label in AA_dif.index]

# Clean x-axis labels
clean_labels_AA = [
    str(label).replace('_x', '').replace('_y', '').replace('_', ' ').title()
    for label in AA_dif.index
]

# Assign label colors based on cleaned text
label_colors_AA = [
    'darkblue' if label in blue_labels else 'black'
    for label in clean_labels_AA
]

plt.figure(figsize=(4, 1))
sns.barplot(x=AA_dif.index, y=AA_dif.values, palette=AA_colors)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.xlabel('Index')
plt.ylabel('Normalized Difference')
plt.xticks(ticks=range(len(clean_labels_AA)), labels=clean_labels_AA, rotation=90)
plt.xticks(size=4)

for tick_label, color in zip(plt.gca().get_xticklabels(), label_colors_AA):
    tick_label.set_color(color)

legend_elements = [
    Patch(facecolor='red', edgecolor='red', label='pre'),
    Patch(facecolor='cyan', edgecolor='cyan', label='post')
]
plt.legend(handles=legend_elements, title='Neuron Role', loc='upper right')
plt.yticks([-0.1,-0.05,0,0.05,0.1],size=8)

plt.tight_layout()
sns.despine(right=True, top=True)
plt.savefig(_out_dir / "synaptic_type_feature_diff_AA_filtered_features.svg")
plt.show()
#%%
# ---------------- DD_dif Plot ----------------
DD_colors = [assign_color(str(label)) for label in DD_dif.index]

clean_labels_DD = [
    str(label).replace('_x', '').replace('_y', '').replace('_', ' ').title()
    for label in DD_dif.index
]

label_colors_DD = [
    'darkblue' if label in blue_labels else 'black'
    for label in clean_labels_DD
]

plt.figure(figsize=(4, 1))
sns.barplot(x=DD_dif.index, y=DD_dif.values, palette=DD_colors)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.xlabel('Index')
plt.ylabel('Normalized Difference')
plt.xticks(ticks=range(len(clean_labels_DD)), labels=clean_labels_DD, rotation=90)

for tick_label, color in zip(plt.gca().get_xticklabels(), label_colors_DD):
    tick_label.set_color(color)
plt.xticks(size=4)

plt.legend(handles=legend_elements, title='Neuron Role', loc='upper right')
sns.despine(right=True, top=True)
plt.yticks([-0.15,-0.1,-0.05,0],size=8)

plt.tight_layout()
plt.savefig(_out_dir / "synaptic_type_feature_diff_DD_filtered_features.svg")
plt.show()
#%%
all_df=syn_df_m[['comp',
      'PC1_x','PC1_y']]



#%%
syn_df_m=syn_df_m[syn_df_m['comp'].isin(['AD','AA','DD','DA'])]
#%%
len_s=len(syn_df_m[syn_df_m['comp']=='DA'])
print(len_s)
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 1: Balance the Dataset
# Define the number of samples per 'comp' type
x = len_s  # Desired number of samples per class

# Get unique 'comp' types
comp_types = syn_df_m['comp'].unique()

# Resample 'x' samples for each 'comp' type
balanced_syn_df_m = pd.concat([
    syn_df_m[syn_df_m['comp'] == comp_type].sample(n=x, random_state=42, replace=True)
    for comp_type in comp_types
], ignore_index=True)

# Check the new distribution
print("Balanced 'comp' Distribution:")
print(balanced_syn_df_m['comp'].value_counts())

#%%
'''
# Randomly sample from the full DataFrame
balanced_syn_df_m = syn_df_m.sample(n=x*4, random_state=42)

# Check distribution
print("Unbalanced 'comp' Distribution:")
print(sampled_syn_df_m['comp'].value_counts())
'''
#%%


# Step 2: Split Data into Features and Target
X = balanced_syn_df_m[[ 
   
'PC1_x','PC1_y']]
     

y = balanced_syn_df_m['comp']  # Target
#y = balanced_syn_df_m['comp'].sample(frac=1).reset_index(drop=True)

#%%

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#%%
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#%%

# Step 3: Train Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Step 4: Evaluate the Classifier
y_pred = rf_clf.predict(X_test)
#%%

import joblib
import os
#%%
model_file = _out_dir / "rf_pc1_model.joblib"
joblib.dump(rf_clf, model_file)

print("Model saved to:", model_file)

#%%
model_file = _out_dir / "rf_pc1_model.joblib"

rf_clf = joblib.load(model_file)
y_pred = rf_clf.predict(X_test)

#%%
'''
t=all_df.head(10000)


#%%
X_dummy = np.array([[15.644506 , 10.027891]])  # shape (1, 2)

# Predict using dummy data
y_pred_dummy = rf_clf.predict(X_dummy)
print(y_pred_dummy)
#%%
sample = all_df.sample(n=100, random_state=10)  # pick 10 random rows
X_sample = sample[['PC1_x', 'PC1_y']]
y_true = sample['comp']

y_pred = rf_clf.predict(X_sample)

# ---------- Evaluate ----------
print("Sample accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification report:\n", classification_report(y_true, y_pred))

# Optional: merge predictions into the DataFrame
sample['predicted'] = y_pred
print("\nPredictions:")
print(sample)
'''
#%%

# ---------- Confusion matrix ----------
cm = confusion_matrix(y_pred, y_pred, labels=rf_clf.classes_)
cm_df = pd.DataFrame(cm, index=rf_clf.classes_, columns=rf_clf.classes_)

plt.figure(figsize=(10,10))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()
#%%simple model
'''
# Step 3: Train Random Forest Classifier
rf_clf = RandomForestClassifier(
    n_estimators=100,        # enough trees for stability
    max_depth=5,            # cap tree depth (prevents huge trees)
    min_samples_split=10,    # node must have at least 20 samples to split
    min_samples_leaf=10,     # each leaf must have at least 10 samples
    max_features='sqrt',     # good default
    bootstrap=True,
    oob_score=True,
    random_state=42
)
rf_clf.fit(X_train, y_train)

# Step 4: Evaluate the Classifier
y_pred = rf_clf.predict(X_test)
'''
#%%
'''
# --- tree sizes ---
depths = [t.get_depth() for t in rf_clf.estimators_]
leaves = [t.get_n_leaves() for t in rf_clf.estimators_]
print(f"Tree depth (min/median/max): {min(depths)} / {np.median(depths)} / {max(depths)}")
print(f"Leaves    (min/median/max): {min(leaves)} / {int(np.median(leaves))} / {max(leaves)}")
'''


#%%
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))
#%%
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Create DataFrame of feature importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_clf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Assign colors based on suffix
colors = feature_importances['Feature'].apply(
    lambda x: 'cyan' if str(x).endswith('_y') else ('red' if str(x).endswith('_x') else 'skyblue')
)

# Create the plot
plt.figure(figsize=(2,2.7))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color=colors)
plt.title('Feature Importances')
plt.gca().invert_yaxis()  # Most important on top
plt.xlabel('Importance')
plt.ylabel('Feature')

# Replace y-axis labels to remove _x and _y suffixes
clean_labels = feature_importances['Feature'].str.replace('_x', '', regex=False).str.replace('_y', '', regex=False)
plt.yticks(ticks=range(len(clean_labels)), labels=clean_labels)

# Add legend for color coding
legend_elements = [
    Patch(facecolor='red', edgecolor='red', label='pre'),
    Patch(facecolor='cyan', edgecolor='cyan', label='post')
]
plt.tick_params(axis='both', width=0.3)

plt.legend(handles=legend_elements, title='Neuron Role', loc='upper right')
#plt.legend(handles=legend_elements, title='Neuron Role', loc='upper right')
sns.despine(right=True,top=True)
plt.rc('axes', titlesize=8)    
plt.rc('axes', labelsize=8)      
plt.xticks(size=4)
plt.xticks(np.arange(0, 0.6, 0.1), size=4)

plt.yticks(size=4)    
plt.rc('legend', fontsize=8)      
plt.rc('legend', title_fontsize=8) 
# Save and show the plot
plt.tight_layout()
plt.savefig(_out_dir / "feature_importance_filtered_pc_usingpc1_not_simpler_non_balanced_princeton.svg", dpi=300, bbox_inches='tight')
plt.show()

#%%

#%%
# Step 6: Compute and Display the Confusion Matrix
# Compute and normalize confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=rf_clf.classes_)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Setup figure and axis
fig, ax = plt.subplots(figsize=(1.5,1.5))  # Set figure size
disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=rf_clf.classes_)

# Plot confusion matrix with formatting
disp.plot(cmap='Blues', values_format='.2f', ax=ax, colorbar=False)
for text in disp.text_.ravel():  # ✅ flatten the array
    text.set_fontsize(4)
# Format axis labels and ticks
ax.set_xlabel('Predicted Label', fontsize=8)
ax.set_ylabel('True Label', fontsize=8)
#ax.set_title('Confusion Matrix (Percent)', fontsize=6)
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
plt.setp(ax.get_xticklabels(), rotation=0, ha='right')

# Save and show
plt.tight_layout()
plt.savefig(_out_dir / "confusion_matrix_filtered_pc_usingpc1_not_simpler_non_balanced_princeton.svg", dpi=300, bbox_inches='tight')
plt.show()
#%%
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# --- Use the exact features you trained on ---
f1, f2 = 'PC1_x', 'PC1_y'   # x = PC1_x, y = PC1_y

# --- Class → color map ---
class_colors = {'AA':'#8b2be2', 'DD':'#F4B95A', 'AD':'#9F4800', 'DA':'#B3B3B3'}

classes = list(rf_clf.classes_)

cmap = ListedColormap([class_colors.get(c, '#cccccc') for c in classes])
class_to_idx = {c: i for i, c in enumerate(classes)}

# --- Fixed axis ranges ---
x_min, x_max = -2,5
y_min, y_max = -2,5

# --- Grid over fixed range ---
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)

# --- Predict on the grid ---
Z = rf_clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z_idx = np.vectorize(class_to_idx.get)(Z).reshape(xx.shape)

# =========================
# 1) Decision surface only
# =========================
plt.figure(figsize=(20, 20))
plt.contourf(xx, yy, Z_idx, alpha=1, cmap=cmap, levels=len(classes))
plt.xlabel("PC1_x")
plt.ylabel("PC1_y")
plt.title('Decision surface (no points)')
plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)
sns.despine(right=True, top=True)
plt.tight_layout()


#plt.savefig(r"C:\Users\user\organised_work\code\783\article\princeton\final\Fig3\Fig_3\syntype X PC1\surface_desicion_map_not_simpler_princeton.png", dpi=1200, bbox_inches='tight')
plt.show()

#%%
class_colors = {'AA':'#8b2be2', 'DD':'#F4B95A', 'AD':'#9F4800', 'DA':'#B3B3B3'}

#%%

test_plot_df = X_test.copy()
test_plot_df['comp'] = y_test.values

point_colors = test_plot_df['comp'].map(class_colors)

plt.figure(figsize=(20, 20))
plt.scatter(
    test_plot_df['PC1_x'],
    test_plot_df['PC1_y'],
    c=point_colors,
    s=6,          # small points
    alpha=0.5,
    linewidths=0
)

plt.xlabel("PC1_x")
plt.ylabel("PC1_y")
plt.title("Test-set neuron-pair points")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

legend_elements = [
    Patch(facecolor=class_colors[c], edgecolor=class_colors[c], label=c)
    for c in classes
]
plt.legend(handles=legend_elements, title='Comp', loc='upper right')

sns.despine(right=True, top=True)
plt.tight_layout()

plt.savefig(
    _out_dir / "test_points_only_princeton.png",
    dpi=1200,
    bbox_inches='tight'
)
plt.show()
#%%
#%%
from matplotlib.colors import ListedColormap

# =========================
# 4 decision maps: one per syn type
# =========================

test_plot_df = X_test.copy()
test_plot_df['comp'] = y_test.values
test_plot_df['predicted'] = y_pred

fig, axes = plt.subplots(2, 2, figsize=(30, 30))
axes = axes.flatten()

# optional fixed order
plot_order = ['AA', 'AD', 'DA', 'DD']
plot_order = [c for c in plot_order if c in classes]

for ax, comp_type in zip(axes, plot_order):

    # choose true class points
    sub = test_plot_df[test_plot_df['comp'] == comp_type].copy()

    ax.scatter(
        sub['PC1_x'],
        sub['PC1_y'],
        c=class_colors[comp_type],
        s=2,
        alpha=0.7,
        linewidths=0
    )

    ax.set_title(comp_type)
    ax.set_xlabel('PC1_x')
    ax.set_ylabel('PC1_y')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    sns.despine(ax=ax, right=True, top=True)

# in case fewer than 4 classes
for j in range(len(plot_order), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig(
    _out_dir / "scatter_4_syn_types_princeton.png",
    dpi=1200,
    bbox_inches='tight'
)
plt.show()
#%%

# =========================
# 4 decision maps: one per syn type
# 2D KDE
# choose at top: all points or sampled points
# =========================
# --- Class → color map ---
class_colors = {'AA':'#8b2be2', 'DD':'#F4B95A', 'AD':'#9F4800', 'DA':'#B3B3B3'}

classes = list(rf_clf.classes_)

cmap = ListedColormap([class_colors.get(c, '#cccccc') for c in classes])
class_to_idx = {c: i for i, c in enumerate(classes)}

# --- Fixed axis ranges ---
x_min, x_max = -2,5
y_min, y_max = -2,5
#%%

# ---- choose mode here ----
use_all_points = False   # True = use all points
n_points = 5000      # used only if use_all_points = False
random_state = 42

test_plot_df = X_test.copy()
test_plot_df['comp'] = y_test.values
test_plot_df['predicted'] = y_pred

fig, axes = plt.subplots(2, 2, figsize=(20, 20))
axes = axes.flatten()

# optional fixed order
plot_order = ['AA', 'AD', 'DA', 'DD']
plot_order = [c for c in plot_order if c in classes]

for ax, comp_type in zip(axes, plot_order):

    # choose true class points
    sub = test_plot_df[test_plot_df['comp'] == comp_type].copy()

    # use all points or sample
    if not use_all_points and len(sub) > n_points:
        sub_plot = sub.sample(n=n_points, random_state=random_state)
    else:
        sub_plot = sub.copy()

    if len(sub_plot) > 1:
        sns.kdeplot(
        data=sub_plot,
        x='PC1_x',
        y='PC1_y',
        fill=True,
        color=class_colors[comp_type],
        levels=10,
        thresh=0.6,
        bw_adjust=0.15,
        alpha=1,
        linewidths=2,
        warn_singular=False,
        ax=ax
    )

    ax.set_title(
        f"{comp_type} | "
        f"{'all points' if use_all_points else f'sampled {min(n_points, len(sub))}'}"
    )
    ax.set_xlabel('PC1_x')
    ax.set_ylabel('PC1_y')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    sns.despine(ax=ax, right=True, top=True)

# in case fewer than 4 classes
for j in range(len(plot_order), len(axes)):
    axes[j].axis('off')

save_name = (
    "kde2d_4_syn_types_princeton_all_points.png"
    if use_all_points
    else f"kde2d_4_syn_types_princeton_sample_{n_points}.png"
)

plt.tight_layout()

plt.show()
#%%
# ---- choose mode here ----
use_all_points = False   # True = use all points
n_points = 500       # used only if use_all_points = False
random_state = 42

test_plot_df = X_test.copy()
test_plot_df['comp'] = y_test.values
test_plot_df['predicted'] = y_pred

fig, ax = plt.subplots(figsize=(20, 20))

# optional fixed order
plot_order = ['AA', 'AD', 'DA', 'DD']
plot_order = [c for c in plot_order if c in classes]

for comp_type in plot_order:

    # choose true class points
    sub = test_plot_df[test_plot_df['comp'] == comp_type].copy()

    # use all points or sample
    if not use_all_points and len(sub) > n_points:
        sub_plot = sub.sample(n=n_points, random_state=random_state)
    else:
        sub_plot = sub.copy()

    if len(sub_plot) > 1:
        sns.kdeplot(
            data=sub_plot,
            x='PC1_x',
            y='PC1_y',
            fill=True,
            color=class_colors[comp_type],
            levels=10,
            thresh=0.6,
            bw_adjust=0.15,
            cut=0,
            alpha=0.6,
            linewidths=0,
            warn_singular=False,
            ax=ax,
            label=comp_type
        )

ax.set_title(
    f"All syn types | "
    f"{'all points' if use_all_points else f'sampled {n_points} per class'}"
)
ax.set_xlabel('PC1_x')
ax.set_ylabel('PC1_y')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.legend()
sns.despine(ax=ax, right=True, top=True)

save_name = (
    "kde2d_all_syn_types_princeton_all_points.png"
    if use_all_points
    else f"kde2d_all_syn_types_princeton_sample_{n_points}.png"
)

plt.tight_layout()
plt.savefig(
    _out_dir / save_name,
    dpi=1200,
    bbox_inches='tight'
)
plt.show()
#%%
# ---- choose mode here ----
use_all_points = True   # True = use all points
n_points = 500           # used only if use_all_points = False
random_state = 42

test_plot_df = X_test.copy()
test_plot_df['comp'] = y_test.values
test_plot_df['predicted'] = y_pred

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes = axes.flatten()

# groups: AA+DD together, AD+DA together
plot_groups = [
    ['AA', 'DD'],
    ['AD', 'DA']
]

group_titles = [
    'AA + DD',
    'AD + DA'
]

for ax, group, title in zip(axes, plot_groups, group_titles):

    for comp_type in group:
        if comp_type not in classes:
            continue

        sub = test_plot_df[test_plot_df['comp'] == comp_type].copy()

        if not use_all_points and len(sub) > n_points:
            sub_plot = sub.sample(n=n_points, random_state=random_state)
        else:
            sub_plot = sub.copy()

        if len(sub_plot) > 1:
            sns.kdeplot(
                data=sub_plot,
                x='PC1_x',
                y='PC1_y',
                fill=True,
                color=class_colors[comp_type],
                levels=10,
                thresh=0.5,
                bw_adjust=0.2,
                cut=0,
                alpha=0.75,
                linewidths=0,
                warn_singular=False,
                ax=ax,
                label=comp_type
            )

    ax.set_title(
        f"{title} | "
        f"{'all points' if use_all_points else f'sampled {n_points} per class'}"
    )
    ax.set_xlabel('PC1_x')
    ax.set_ylabel('PC1_y')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend()
    sns.despine(ax=ax, right=True, top=True)

save_name = (
    "kde2d_grouped_AADD_ADDA_all_points.png"
    if use_all_points
    else f"kde2d_grouped_AADD_ADDA_sample_{n_points}.png"
)

plt.tight_layout()
plt.savefig(
    _out_dir / save_name,
    dpi=1200,
    bbox_inches='tight'
)
plt.show()
#%%

test_plot_df = X_test.copy()
test_plot_df['comp'] = y_test.values
test_plot_df['predicted'] = y_pred

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

plot_order = ['AA', 'AD', 'DA', 'DD']
plot_order = [c for c in plot_order if c in classes]

for ax, comp_type in zip(axes, plot_order):

    sub = test_plot_df[test_plot_df['comp'] == comp_type].copy()

    # KDE
    if len(sub) > 1:
        sns.kdeplot(
            data=sub,
            x='PC1_x',
            y='PC1_y',
            fill=True,
            levels=6,
            thresh=0.05,
            alpha=0.5,
            color=class_colors[comp_type],
            ax=ax
        )

    # Scatter on top
    ax.scatter(
        sub['PC1_x'],
        sub['PC1_y'],
        c=class_colors[comp_type],
        s=6,
        alpha=0.5,
        linewidths=0
    )

    ax.set_title(comp_type)
    ax.set_xlabel('PC1_x')
    ax.set_ylabel('PC1_y')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    sns.despine(ax=ax, right=True, top=True)

for j in range(len(plot_order), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig(
    _out_dir / "kde_scatter_4_syn_types_princeton.png",
    dpi=1200,
    bbox_inches='tight'
)
plt.show()


 #%%
# =========================
# Recover original test pairs + keep only central-central
# =========================

test_pairs_df = balanced_syn_df_m.loc[
    X_test.index,
    ['pre', 'post', 'comp', 'super_class_x', 'super_class_y', 'PC1_x', 'PC1_y']
].copy()

test_pairs_df['predicted'] = y_pred
test_pairs_df['is_correct'] = test_pairs_df['comp'] == test_pairs_df['predicted']

central_central_test = test_pairs_df[
    (test_pairs_df['super_class_x'] == 'central') &
    (test_pairs_df['super_class_y'] == 'central')
].copy()

print('Number of central-central test pairs:', len(central_central_test))
print(central_central_test.head())
#%%
# =========================
# Plot only central-central test pairs
# colored by TRUE class
# =========================

point_colors = central_central_test['comp'].map(class_colors)

plt.figure(figsize=(20, 20))
plt.scatter(
    central_central_test['PC1_x'],
    central_central_test['PC1_y'],
    c=point_colors,
    s=6,
    alpha=0.8,
    linewidths=0
)

plt.xlabel("PC1_x")
plt.ylabel("PC1_y")
plt.title("Central-central test-set neuron-pair points")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

legend_elements = [
    Patch(facecolor=class_colors[c], edgecolor=class_colors[c], label=c)
    for c in classes
]
plt.legend(handles=legend_elements, title='Comp', loc='upper right')

sns.despine(right=True, top=True)
plt.tight_layout()

plt.savefig(
    _out_dir / "test_points_central_central_princeton.png",
    dpi=1200,
    bbox_inches='tight'
)
plt.show()
#%%

# =========================
# Recover original test pairs + keep only optic-optic
# =========================

test_pairs_df = balanced_syn_df_m.loc[
    X_test.index,
    ['pre', 'post', 'comp', 'super_class_x', 'super_class_y', 'PC1_x', 'PC1_y']
].copy()

test_pairs_df['predicted'] = y_pred
test_pairs_df['is_correct'] = test_pairs_df['comp'] == test_pairs_df['predicted']

optic_optic_test = test_pairs_df[
    (test_pairs_df['super_class_x'] == 'optic') &
    (test_pairs_df['super_class_y'] == 'optic')
].copy()

print('Number of optic-optic test pairs:', len(optic_optic_test))
print(optic_optic_test.head())
#%%
# =========================
# Plot only optic-optic test pairs
# colored by TRUE class
# =========================

point_colors = optic_optic_test['comp'].map(class_colors)

plt.figure(figsize=(20, 20))
plt.scatter(
    optic_optic_test['PC1_x'],
    optic_optic_test['PC1_y'],
    c=point_colors,
    s=8,
    alpha=0.8,
    linewidths=0
)

plt.xlabel("PC1_x")
plt.ylabel("PC1_y")
plt.title("optic-optic test-set neuron-pair points")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
classes = list(rf_clf.classes_)
legend_elements = [
    Patch(facecolor=class_colors[c], edgecolor=class_colors[c], label=c)
    for c in classes
]
plt.legend(handles=legend_elements, title='Comp', loc='upper right')

sns.despine(right=True, top=True)
plt.tight_layout()

plt.savefig(
    _out_dir / "test_points_optic_optic_princeton2.png",
    dpi=1200,
    bbox_inches='tight'
)

plt.show()
#%%

#%%
# =========================
# 4 subplots for central-central
# one panel per synaptic type
# =========================

fig, axes = plt.subplots(2, 2, figsize=(24, 24), sharex=True, sharey=True)
axes = axes.flatten()

plot_order = ['AA', 'AD', 'DA', 'DD']

for ax, comp_type in zip(axes, plot_order):
    sub = central_central_test[central_central_test['comp'] == comp_type].copy()

    ax.scatter(
        sub['PC1_x'],
        sub['PC1_y'],
        c=class_colors[comp_type],
        s=6,
        alpha=0.8,
        linewidths=0
    )

    ax.set_title(f'central-central\n{comp_type}', fontsize=18)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('PC1_x', fontsize=14)
    ax.set_ylabel('PC1_y', fontsize=14)
    sns.despine(ax=ax, right=True, top=True)

plt.tight_layout()
plt.savefig(
    _out_dir / "central_central_4_panels_princeton.png",
    dpi=1200,
    bbox_inches='tight'
)
plt.show()

#%%
x_min=-2
x_max=5
y_min=-2
y_max=5
# =========================
# 4 subplots for optic-optic
# one panel per synaptic type
# =========================

fig, axes = plt.subplots(2, 2, figsize=(24, 24), sharex=True, sharey=True)
axes = axes.flatten()

plot_order = ['AA', 'AD', 'DA', 'DD']

for ax, comp_type in zip(axes, plot_order):
    sub = optic_optic_test[optic_optic_test['comp'] == comp_type].copy()

    ax.scatter(
        sub['PC1_x'],
        sub['PC1_y'],
        c=class_colors[comp_type],
        s=2,
        alpha=0.8,
        linewidths=0
    )

    ax.set_title(f'optic-optic\n{comp_type}', fontsize=18)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('PC1_x', fontsize=14)
    ax.set_ylabel('PC1_y', fontsize=14)
    sns.despine(ax=ax, right=True, top=True)

plt.tight_layout()
plt.savefig(
    _out_dir / "optic_optic_4_panels_princeton.png",
    dpi=1200,
    bbox_inches='tight'
)
plt.show()
#%%
connections2=pd.read_csv(PC1_TABLE_CSV)
#%%

#%%
# =========================
# Full list of test samples
# pre id, post id, predicted label
# =========================

test_samples_full = balanced_syn_df_m.loc[
    X_test.index,
    ['pre', 'post']
].copy()

test_samples_full['predicted_label'] = y_pred

#%%

connections2=connections2.merge(test_samples_full,on=['pre','post'],how='left')
#%%
a = connections2.drop_duplicates(subset=['pre', 'post'])
#%%
aa=connections2.head(10000)
#%%
connections_tested=connections2.dropna()

#%%
#%%
syn_cols = ['AA', 'AD', 'DA', 'DD']

connections_tested['mode_syn_type'] = connections_tested[syn_cols].idxmax(axis=1)

connections_tested['mode_matches_predicted'] = (
    connections_tested['mode_syn_type'] == connections_tested['predicted_label']
).astype(int)

connections_tested['mode_matches_predicted'].mean()

#%%
connections_tested.to_csv(PC1_TABLE_PREDICTIONS_CSV)

#%%
df=pd.read_csv(PC1_TABLE_PREDICTIONS_CSV)
