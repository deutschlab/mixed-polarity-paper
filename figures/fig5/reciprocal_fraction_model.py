
import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'



from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    NEURON_TABLE_FTR, RECI_PROP_FTR, RF_MODEL_PKL, OUTPUT_DIR, METHODS_DIR,
)
sys.path.insert(0, str(METHODS_DIR))
from methods_all import *
import os
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
#%%


nodesG=pd.read_feather(NEURON_TABLE_FTR)
nodesG=nodesG[nodesG['super_class'].isin(['central','optic','visual_centrifugal','visual_projection'])]
nodesG['root_id']=nodesG['root_id'].astype(np.int64)
#%%
final_neuron_stats_df=pd.read_feather(RECI_PROP_FTR)
#%%
final_neuron_stats_df['root_id']=final_neuron_stats_df['root_id'].astype(np.int64)
#%%

nodesG=nodesG.merge(final_neuron_stats_df[['root_id','ratio']],on='root_id',how='left')
#%%
nodesG=nodesG.rename(columns={'mean_in_degree':'in synapses/in partners','mean_out_degree':'out synapses/out partners'})
#%%
nodesG_nclass=nodesG[nodesG['super_class'].isin(['central','optic','visual_centrifugal','visual_projection'])]

t_sample=nodesG_nclass[['length_nm', 'area_nm', 'size_nm', 
       'out_synapses_count', 'in_synapses_count', 'max_out_degree',
       'max_in_degree', 'in synapses/in partners', 'out synapses/out partners', 'out_partners',
       'in_partners', 'nodes', 'leafs', 'branches', 'cable_length','ratio']].dropna() #%%.dropna() #.dropna()  #.sample(50000,random_state=42).dropna()
X=t_sample[['length_nm', 'area_nm', 'size_nm', 
       'out_synapses_count', 'in_synapses_count', 'max_out_degree',
       'max_in_degree', 'in synapses/in partners', 'out synapses/out partners', 'out_partners',
       'in_partners', 'nodes', 'leafs', 'branches', 'cable_length']]
y=t_sample[['ratio']]


#%%
import multiprocessing

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import numpy as np
# Define the parameter grid you want to search
param_grid = {
    'n_estimators': [100, 200, 300],            # 3 options
    'max_depth': [10, 20, None],                 # 3 options
    'min_samples_split': [2, 5],   
    'min_samples_leaf': [1, 2]               # 2 options
}

# Create base model
rf = RandomForestRegressor(random_state=42)

# Define inner cross-validation (for hyperparameter tuning)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define outer cross-validation (for model evaluation)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

n_jobs = multiprocessing.cpu_count() // 2
#%%
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=inner_cv,
    scoring='r2',
    n_jobs=3
)
#%%
# Perform Nested CV (outer loop)
nested_scores = cross_val_score(
    grid_search, X, y.values.ravel(),  # IMPORTANT: ravel y if needed
    cv=outer_cv,
    scoring='r2',  # or 'neg_mean_squared_error'
    n_jobs=3
)

print(f"Nested CV R2 scores: {nested_scores}")
print(f"Mean Nested CV R2: {np.mean(nested_scores):.3f}")

#%%
# Step 6: Refit GridSearchCV on full data
grid_search_full = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=inner_cv,
    scoring='r2',
    n_jobs=3
)

grid_search_full.fit(X, y.values.ravel())

best_model = grid_search_full.best_estimator_

print(f"Best Hyperparameters: {grid_search_full.best_params_}")



#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
#%%
# Train on train set
best_model.fit(X_train, y_train.values.ravel())
#%%
# Predict
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {mse:.3f}")
print(f"Test R2: {r2:.3f}")

# Get feature importances
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values(by='importance', ascending=False)

print(feature_importances)

plt.figure(figsize=(8,6))
plt.barh(feature_importances['feature'], feature_importances['importance'])
plt.gca().invert_yaxis()
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importances')
plt.tight_layout()
_out_dir = OUTPUT_DIR / "fig5" / "reciprocal_fraction_model"
_out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(_out_dir / "feature_importance.svg", format='svg', dpi=300, bbox_inches='tight')
plt.show()
#%%
import joblib


# Save the final model
joblib.dump(
    best_model,
    RF_MODEL_PKL
)

#%%
r'''
import joblib

# Load the saved model
best_model = joblib.load(
    r'C:\Users\user\organised_work\code\783\article\5\new_org\figs\reci partners\organized\New folder\final_random_forest_model.pkl'
)
'''


#%%
print(f"Random Forest MSE: {mse:.3f}")
print(f"Random Forest R²: {r2:.3f}")
from sklearn.metrics import mean_squared_error
y_baseline = np.full_like(y_test, y_test.mean())  # Predict mean for all
mse_baseline = mean_squared_error(y_test, y_baseline)

print(f"Baseline MSE: {mse_baseline:.4f}")
print(f"Model MSE: {mse:.4f}")

#%%

yr=pd.concat([pd.DataFrame(y_pred),pd.DataFrame(y_test).reset_index(drop=True)],axis=1)
yr.columns=['predicted','actual']

yr['predicted_dif']=abs(yr['predicted']-yr['actual'])
yr=yr*50
yr[['predicted','actual']]=yr[['predicted','actual']].astype(int)

yr_g = yr.groupby('actual')['predicted'].agg(['mean', 'std']).reset_index()
yr_g.columns = ['actual', 'predicted', 'std']

#yr_g[['actual', 'predicted', 'std']] = yr_g[['actual', 'predicted', 'std']]

yr_g_c = yr.groupby('actual')['predicted'].count().reset_index()
yr_g_c['actual'] = yr_g_c['actual'] / 50
yr_g=yr_g/50


#%%

# Plot
fig, ax1 = plt.subplots(figsize=(8, 6))

# Left y-axis: mean predicted line
sns.lineplot(data=yr_g, x='actual', y='predicted', ax=ax1, color='blue')

# Shaded area for ±1 SD
ax1.fill_between(
    yr_g['actual'],
    yr_g['predicted'] - yr_g['std'],
    yr_g['predicted'] + yr_g['std'],
    color='blue',
    alpha=0.2,
    label='±1 SD'
)

ax1.set_ylabel('Predicted Reciprocal Proportion', color='black')
ax1.tick_params(axis='y', labelcolor='black')

# Right y-axis: neuron count
ax2 = ax1.twinx()
sns.lineplot(data=yr_g_c, x='actual', y='predicted', ax=ax2, color='orange')
ax2.set_ylabel('Neurons', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# X-axis and title
ax1.set_xlabel('Actual Reciprocal Proportion')
plt.title('Predicted vs Actual reciprocal ratio binned by 0.02, n=22849')


# Diagonal x = y reference line

# Custom legend
legend_elements = [
    Line2D([0], [0], color='orange', lw=2, label='Number of neurons'),
    Line2D([0], [0], color='blue', lw=2, label='Result (Mean over 0.02 Binns)'),

    Patch(facecolor='blue', edgecolor='blue', alpha=0.2, label='±1 SD')
]

ax1.legend(handles=legend_elements, loc='upper right', frameon=False)

# Clean up spines
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.set_ylim(0,0.35)
plt.xlim(0, 0.35)
plt.tight_layout()

# Save as SVG
fig_path = _out_dir / "binned_prediction_and_count_with_sd_princeton.svg"
plt.savefig(fig_path, format='svg', dpi=300, bbox_inches='tight')
plt.show()
#%%
best_model.get_params()