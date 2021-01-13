import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
DATA_PATH = 'dataset/'

TRAIN_FEATURES = DATA_PATH + 'train_features.csv'
TEST_FEATURES = DATA_PATH + 'test_features.csv'
TRAIN_TARGETS_NON_SCORED = DATA_PATH + 'train_targets_nonscored.csv'
TRAIN_TARGETS_SCORED = DATA_PATH + 'train_targets_scored.csv'

train_features = pd.read_csv(TRAIN_FEATURES)
train_targets = pd.read_csv(TRAIN_TARGETS_SCORED)
test_features = pd.read_csv(TEST_FEATURES)

#train mean and std visual
def scatter_description(description,x,step=9):
    plt.figure(figsize=(10,20))
    ax = sns.scatterplot(x=x,y='index',data=description)
    N = len(description)
    ax.set_yticks(np.arange(0, N, step))
    ax.set_yticklabels(description['index'].values[::step], fontsize=12)
    ax.set_ylabel('index', fontsize=16)
    ax.set_xlabel(x, fontsize=16)
    ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    ax.set_title(x, fontsize=24)
    plt.show()
description = train_features.drop(columns=['cp_time', "cp_type", "cp_dose"]).describe().T.reset_index()
scatter_description(description, x='mean')
scatter_description(description, x='std')


#train var visual
description['mean/var'] = description['mean']/description['std']**2
description['group'] = [name[0] for name in description['index']]
description['var'] = description['std']**2
plt.figure(figsize=(15, 8))
ax = sns.scatterplot(x='mean', y='var', data=description, hue='group')
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.set_ylabel('Var', fontsize=18)
ax.set_xlabel('Mean', fontsize=18)
ax.legend(prop=dict(size=16))
ax.set_title("Features g- and c- var/mean dependency", fontsize=20)
plt.show()

#test mean and std visual
description1 = test_features.drop(columns=['cp_time', "cp_type", "cp_dose"]).describe().T.reset_index()
scatter_description(description1, x='mean')
scatter_description(description1, x='std')


#test var visual
description1['mean/var'] = description1['mean']/description1['std']**2
description1['group'] = [name[0] for name in description1['index']]
description1['var'] = description1['std']**2
plt.figure(figsize=(15, 8))
ax = sns.scatterplot(x='mean', y='var', data=description1, hue='group')
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.set_ylabel('Var', fontsize=18)
ax.set_xlabel('Mean', fontsize=18)
ax.legend(prop=dict(size=16))
ax.set_title("Features g- and c- var/mean dependency", fontsize=20)
plt.show()



#cell feature visual
cell_features = list([x for x in list(train_features.columns) if "c-" in x])
fig, ax = plt.subplots(7, 5, figsize=(35, 16))
rand_feats = np.random.choice(cell_features, 35, replace=False)
train_features[rand_feats].plot(
    kind='kde',
    subplots=True,
    ax=ax,
)
fig.tight_layout()
plt.show()

#train cell and gene features correlation visual
rand_cell_feats = np.random.choice(cell_features, 35, replace=False)
gene_features = list([x for x in list(train_features.columns) if "g-" in x])
rand_gene_feats = np.random.choice(gene_features, 35, replace=False)
random_selected_features = list(rand_gene_feats) + list(rand_cell_feats)

plt.figure(figsize=(20, 10))
ax = sns.heatmap(train_features[random_selected_features].corr())
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)

plt.title('Correlation matrix for cell and gene features', fontsize=20)
plt.show()



#train targets correlation visual
plt.figure(figsize=(20, 10))
ax = sns.heatmap(train_targets.corr())
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
plt.title('Correlation matrix for targets', fontsize=20)
plt.show()


targets_corr = train_targets.corr()
c = targets_corr.abs()
s = c.unstack()
so = s.sort_values(kind="quicksort")
print(pd.DataFrame(so[so>0.7], columns=["correlation"]).head(8))


#show the number of labels distribution
sns.set_palette(sns.color_palette("colorblind"))
target_cols = list(train_targets.columns)
print(target_cols)
target_cols.remove('sig_id')
multiple_labels = train_targets[target_cols].sum(axis=1)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.countplot(multiple_labels, ax=ax)
ax.set_xlabel('Number of labels', fontsize=18)
ax.set_ylabel('Frequency', fontsize=18)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.set_title('Distribution of number of labels', fontsize=20)
plt.show()


#show the labels frequency
multiple_labels = train_targets[target_cols].sum(axis=0).sort_values(ascending=False)[:10]
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
chart = sns.barplot(multiple_labels.values, multiple_labels.index)
ax.set_title('Most frequent triggered mechanisms', fontsize=20)
ax.set_xlabel('Frequency', fontsize=14)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
chart.set_yticklabels(chart.get_yticklabels(), fontsize=16)
plt.show()
print(train_features.head(8))
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
train_features['cp_time'] = train_features['cp_time'].map({24: 0, 48: 1, 72: 2})
train_features['cp_dose'] = train_features['cp_dose'].map({'D1': 3, 'D2': 4})
le = LabelEncoder()
train_features['cp_type'] = le.fit_transform(train_features['cp_type'])
import random
cat_features = ['cp_type', 'cp_time', 'cp_dose']
print(train_features.head(8))
features = cell_features +  gene_features + cat_features
X = train_features[features].values
y_1 = train_targets[target_cols].sum(axis=1).values
y_2 = train_targets['nfkb_inhibitor'].values
y_3 = train_targets['proteasome_inhibitor'].values
y_4 = train_targets['cyclooxygenase_inhibitor'].values
y_5 = train_targets['pdgfr_inhibitor'].values
y_6 = train_targets['flt3_inhibitor'].values
y_7 = train_targets['kit_inhibitor'].values
indices = random.choices(range(len(X)), k=5000)

X = X[indices,]
y_1 = y_1[indices,]
y_2 = y_2[indices,]
y_3 = y_3[indices,]
y_4 = y_4[indices,]
y_5 = y_5[indices,]
y_6 = y_6[indices,]
y_7 = y_7[indices,]

print('X shape:', X.shape)
print('y shape:', y_1.shape)
# First we reduce the number of features using PCA
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)

# Then we can apply TSNE for low-dimensional visualization of the data points
t_sne_results_2d = TSNE(n_components=2).fit_transform(X_reduced)
fig, ax = plt.subplots(1, 1, figsize=(16,10))
sns.scatterplot(t_sne_results_2d[:, 0], t_sne_results_2d[:, 1], hue=y_1,
                palette=sns.color_palette("colorblind", len(np.unique(y_1))), legend="full",
                alpha=0.3, ax=ax)
ax.legend(prop=dict(size=18))
ax.set_title('2D visualization of T-SNE components on PCA  - Number of triggered mechanisms', fontsize=20);
plt.show()



fig, ax = plt.subplots(1, 1, figsize=(16,10))
sns.scatterplot(t_sne_results_2d[:, 0], t_sne_results_2d[:, 1], hue=y_2,
                palette=sns.color_palette("colorblind", 2), legend="full",
                alpha=0.3, ax=ax)
ax.legend(prop=dict(size=16))
ax.set_title('2D visualization of T-SNE components on PCA - nfkb_inhibitor triggered ', fontsize=20)
plt.show()
