

from nglui.statebuilder import ChainedStateBuilder
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'

import matplotlib.pyplot as plt
import numpy as np
from fafbseg import flywire
import navis
import pandas as pd

import time
import datetime
import os
import sys
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import NEURON_TABLE_FTR, SYNAPSE_TABLE_FTR, OUTPUT_DIR, METHODS_DIR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
client = CAVEclient('flywire_fafb_production')
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pickle
import itertools
import gc
#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)

#%%
       #%%

                                         

#%%
syn_df=pd.read_feather(SYNAPSE_TABLE_FTR)
       #%%#%%

nodesG=nodesG[['neuron','super_class',
       'length_nm', 'area_nm', 'size_nm', 
       'out_synapses_count', 'in_synapses_count', 'max_out_degree',
       'max_in_degree', 'mean_in_degree', 'mean_out_degree', 'out_partners',
       'in_partners', 'nodes', 'leafs', 'branches', 'cable_length']]
#%%

nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]

#%%
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
all_df=syn_df_m[[ 'comp',
       'length_nm_x', 'area_nm_x', 'size_nm_x', 
       'out_synapses_count_x', 'in_synapses_count_x',
               'out_partners_x', 'in_partners_x', 'leafs_x', 
        
   
       'length_nm_y', 'area_nm_y', 'size_nm_y', 
       'out_synapses_count_y', 'in_synapses_count_y', 'max_out_degree_y',
       
       'out_partners_y', 'in_partners_y',
        'leafs_y',
       ]]
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
_out_dir = OUTPUT_DIR / "fig3" / "syntype_x_features"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "synaptic_type_feature_diff_AA_filtered_features_princeton_filtered.svg")
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
plt.savefig(_out_dir / "synaptic_type_feature_diff_DD_filtered_features_princeton_filtered_bign.svg")
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
plt.yticks([-0.15,-0.1,-0.05,0,0.05,0.1,0.15],size=8)

plt.tight_layout()
sns.despine(right=True, top=True)
plt.savefig(_out_dir / "synaptic_type_feature_diff_AA_filtered_features_princeton_filtered_bign.svg")
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
plt.savefig(_out_dir / "synaptic_type_feature_diff_DD_filtered_features_princeton_filtered_bign.svg")
plt.show()
#%%
all_df=syn_df_m[['comp',
      'length_nm_x', 'area_nm_x', 'size_nm_x', 
       'out_synapses_count_x', 'in_synapses_count_x', 'max_out_degree_x',
      
       'out_partners_x', 'in_partners_x', 'leafs_x',
       'length_nm_y', 'area_nm_y', 'size_nm_y', 
       'out_synapses_count_y', 'in_synapses_count_y', 'max_out_degree_y',
      
       'out_partners_y', 'in_partners_y',  'leafs_y', 
]]


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


# Step 2: Split Data into Features and Target
X = balanced_syn_df_m[[ 
   
         'length_nm_x', 'area_nm_x', 'size_nm_x', 
          'out_synapses_count_x', 'in_synapses_count_x', 
          'out_partners_x', 'in_partners_x', 'leafs_x', 
          
          'length_nm_y', 'area_nm_y', 'size_nm_y', 
          'out_synapses_count_y', 'in_synapses_count_y',
          'out_partners_y', 'in_partners_y','leafs_y',
     ]]

y = balanced_syn_df_m['comp']  # Target
#y = balanced_syn_df_m['comp'].sample(frac=1).reset_index(drop=True)

#%%

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 3: Train Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Step 4: Evaluate the Classifier
y_pred = rf_clf.predict(X_test)

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
plt.xticks([0,0.02,0.04,0.06,0.08],size=4)

plt.yticks(size=4)    
plt.rc('legend', fontsize=8)      
plt.rc('legend', title_fontsize=8) 
# Save and show the plot
save_path = _out_dir / "feature_importance_filtered_princeton_filtered.svg"
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()



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
plt.savefig(_out_dir / "confusion_matrix_filtered_princeton_filtered.svg", dpi=300, bbox_inches='tight')
plt.show()


#%%



# Step 2: Split Data into Features and Target
X = balanced_syn_df_m[[ 
   
         'length_nm_x', 'area_nm_x', 'size_nm_x', 
          'out_synapses_count_x', 'in_synapses_count_x', 'max_out_degree_x',
          'max_in_degree_x', 'mean_in_degree_x', 'mean_out_degree_x',
          'out_partners_x', 'in_partners_x', 'nodes_x', 'leafs_x', 'branches_x',
          'cable_length_x', 
          
          'length_nm_y', 'area_nm_y', 'size_nm_y', 
          'out_synapses_count_y', 'in_synapses_count_y', 'max_out_degree_y',
          'max_in_degree_y', 'mean_in_degree_y', 'mean_out_degree_y',
          'out_partners_y', 'in_partners_y', 'nodes_y', 'leafs_y', 'branches_y',
          'cable_length_y'
]]
y = balanced_syn_df_m['comp']  # Target




# Step 3: Train Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Step 4: Evaluate the Classifier
y_pred = rf_clf.predict(X_test)
#%%
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))





#%%
from sklearn.model_selection import cross_val_score

# Step 2: Perform cross-validation
cv_scores = cross_val_score(rf_clf, X, y, cv=5, scoring='accuracy')  # cv=5 for 5-fold CV

# Step 3: Output results
print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", cv_scores.mean())