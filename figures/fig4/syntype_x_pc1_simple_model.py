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
from config import NEURON_TABLE_FTR, SYNAPSE_TABLE_FTR, PCA_TABLE_NONP_FTR, OUTPUT_DIR, METHODS_DIR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *

#%%
nodesG=pd.read_feather(NEURON_TABLE_FTR)
#%%

nodesG=nodesG[['neuron','super_class',
       'length_nm', 'area_nm', 'size_nm', 
       'out_synapses_count', 'in_synapses_count', 
        'out_partners',
       'in_partners', 'leafs', ]]
#%%

nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_projection','visual_centrifugal'])]
#%%

pcadf=pd.read_feather(PCA_TABLE_NONP_FTR)
#%%

nodesG=nodesG.merge(pcadf,on='neuron',how='left')
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
rf_clf = RandomForestClassifier(n_estimators=100,max_leaf_nodes=100,random_state=42)
rf_clf.fit(X_train, y_train)

# Step 4: Evaluate the Classifier
y_pred = rf_clf.predict(X_test)
#%%

import joblib
import os
r'''
save_path = r"C:\Users\user\organised_work\code\783\article\princeton\final\Fig3\Fig_3\syntype X PC1"

# Create directory if it does not exist
os.makedirs(save_path, exist_ok=True)

model_file = os.path.join(save_path, "rf_pc1_model.joblib")
joblib.dump(rf_clf, model_file)

print("Model saved to:", model_file)
'''
#%%
#rf_loaded = joblib.load(model_file)


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
_out_dir = OUTPUT_DIR / "fig4" / "syntype_x_pc1_simple_model"
_out_dir.mkdir(parents=True, exist_ok=True)
save_path = _out_dir / "feature_importance_filtered_pc_usingpc1_not_simpler_non_balanced_princeton.svg"
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
plt.figure(figsize=(4, 4))
plt.contourf(xx, yy, Z_idx, alpha=1, cmap=cmap, levels=len(classes))
plt.xlabel("PC1_x")
plt.ylabel("PC1_y")
plt.title('Decision surface (no points)')
plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)
sns.despine(right=True, top=True)
plt.tight_layout()


plt.savefig(_out_dir / "surface_desicion_map_not_simpler_princeton.png", dpi=300, bbox_inches='tight')
plt.show()
