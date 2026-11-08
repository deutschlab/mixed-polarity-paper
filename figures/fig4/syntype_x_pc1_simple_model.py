# Random forest predicting a synapse's type from PC1 of its two cells, on PCA_TABLE_FTR, with at most 100 leaves
# per tree: the Fig 4 model. Its decision surface is Fig 4D. Fig 4A, the same model's confusion matrix on neurons it
# never saw, is drawn by fig4_heldout_models.py; the confusion matrix saved here scores a random split of synapses,
# whose cells are also in training. (The published 4D was drawn by syntype_x_pc1.py's forest without a leaf limit.)
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
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from methods.figure_text import use_arial_8
use_arial_8()   # every text in Arial 8 pt when a figure is saved
from config import NEURON_TABLE_FTR, SYNAPSE_TABLE_FTR, PCA_TABLE_FTR, OUTPUT_DIR, METHODS_DIR
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import seaborn as sns

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

pcadf=pd.read_feather(PCA_TABLE_FTR)
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
for text in disp.text_.ravel():  # flatten the array
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
# 1) Decision surface only (Fig 4D)
# =========================
# Each grid point drawn as a cell in its predicted type's colour (a contour plot would interpolate between the type
# numbers and paint thin bands of types the model never predicted). The surface is a raster inside the SVG; the text
# stays editable. The four type names above the plot are the key, as in the published panel.
from matplotlib.colors import BoundaryNorm
norm = BoundaryNorm(np.arange(-0.5, len(classes)), cmap.N)
# The page is 2.4 x 2.6 inches with the axes 1.75 x 1.75 inches, 0.5 inches from its foot, as the side plot's page
# and axes (Fig 4D-2, syntype_x_pc1.py), so the two line up when the panel is put together.
fig = plt.figure(figsize=(2.4, 2.6))
ax = fig.add_axes([0.6 / 2.4, 0.5 / 2.6, 1.75 / 2.4, 1.75 / 2.6])
ax.pcolormesh(xx, yy, Z_idx, cmap=cmap, norm=norm, shading='auto', rasterized=True)
ax.set_xlabel("PC1 pre neuron")
ax.set_ylabel("PC1 post neuron")
ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_aspect('equal')
ax.set_xticks(range(x_min, x_max + 1)); ax.set_yticks(range(y_min, y_max + 1))
for k, t in enumerate(['AD', 'AA', 'DD', 'DA']):
    ax.text(0.2 + 0.2 * k, 1.03, t, color=class_colors[t], transform=ax.transAxes, ha='center', va='bottom')
sns.despine(ax=ax, right=True, top=True)

# The four numbered examples of the legend, drawn in Supp 4-S1: each a connection from a presynaptic to a
# postsynaptic cell, placed at the two cells' PC1 (PCA_TABLE_FTR) and filled with the colour of its synapse type.
# For the reciprocal pairs 1 and 2 both directions are drawn, joined by an arrow; for the reciprocal AA pair 3 one
# direction is drawn, as the other lies next to it.
#   1: T3 and LC11 (T3 to LC11 AD, LC11 to T3 DA)    2: Tm4 and C2 (Tm4 to C2 AD, C2 to Tm4 DA)
#   3: AVLP046 to PS096 (AA)                          4: Tm2 to L5 (DD)
EXAMPLE_CELLS = {'T3': 720575940618929477, 'LC11': 720575940604601022, 'Tm4': 720575940618081286,
                 'C2': 720575940617954260, 'AVLP046': 720575940621448357, 'PS096': 720575940623501129,
                 'L5': 720575940640137048, 'Tm2': 720575940603314656}
EXAMPLES = [(1, [('T3', 'LC11', 'AD'), ('LC11', 'T3', 'DA')]),
            (2, [('Tm4', 'C2', 'AD'), ('C2', 'Tm4', 'DA')]),
            (3, [('AVLP046', 'PS096', 'AA')]),
            (4, [('Tm2', 'L5', 'DD')])]
pc1_of = pcadf.set_index('neuron')['PC1']
assert pc1_of.index.is_unique and pc1_of.index.dtype == np.int64
cell_pc1 = {t: float(pc1_of.loc[np.int64(i)]) for t, i in EXAMPLE_CELLS.items()}
for num, directions in EXAMPLES:
    points = [(cell_pc1[pre], cell_pc1[post], syn) for pre, post, syn in directions]
    for x, y, syn in points:
        predicted = rf_clf.predict(pd.DataFrame({f1: [x], f2: [y]}))[0]
        print(f"example {num}: ({x:.3f}, {y:.3f}) {syn}, the map predicts {predicted}")
        ax.plot(x, y, 'o', markersize=8, color='white', zorder=4, clip_on=False)   # a white ring, so the circle shows on a region of its own colour
        ax.plot(x, y, 'o', markersize=6, markerfacecolor=class_colors[syn], markeredgecolor='black',
                markeredgewidth=0.6, zorder=4, clip_on=False)
    if len(points) == 2:
        (x0, y0, _), (x1, y1, _) = points
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), zorder=3,
                    arrowprops=dict(arrowstyle='<->', color='black', lw=0.6, shrinkA=4, shrinkB=4,
                                    connectionstyle='arc3,rad=0' if num == 1 else 'arc3,rad=-0.5'))
        lx, ly = (x0 + x1) / 2, (y0 + y1) / 2
        if num == 2:
            lx, ly = lx - 0.55, ly - 0.55   # beside the curved arrow, on its outer side
        else:
            lx, ly = lx + 0.25, ly + 0.25
    else:
        lx, ly = points[0][0] + 0.40, points[0][1] + 0.40
    # a white disc behind the number keeps it readable and keeps it text in the SVG (a path effect would not)
    ax.text(lx, ly, str(num), ha='center', va='center', color='black', zorder=5,
            bbox=dict(boxstyle='circle,pad=0.12', fc='white', ec='none', alpha=0.85))


plt.savefig(_out_dir / "surface_desicion_map_not_simpler_princeton.svg", dpi=600)   # not cropped: the page is 2.6 in tall with the axes 0.5 in from its foot, as in 4D-2, so the two line up edge to edge
plt.show()
