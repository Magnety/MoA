import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from joblib import dump, load
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, brier_score_loss, precision_score, recall_score, f1_score
from datetime import date
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#data preparation
DATA_PATH = 'dataset/'
TRAIN_FEATURES = DATA_PATH + 'train_features.csv'
TEST_FEATURES = DATA_PATH + 'test_features.csv'
TRAIN_TARGETS_NON_SCORED = DATA_PATH + 'train_targets_nonscored.csv'
TRAIN_TARGETS_SCORED = DATA_PATH + 'train_targets_scored.csv'
SAMPLE_SUBMISSION = DATA_PATH + 'sample_submission.csv'
train_features_dtypes = {"cp_type": "category","cp_dose": "category"}
train_features = pd.read_csv(TRAIN_FEATURES,dtype = train_features_dtypes)
train_features['train'] = 1
test_features = pd.read_csv(TEST_FEATURES,dtype = train_features_dtypes)
test_features['train'] = 0
temp = pd.concat([train_features,test_features],axis = 0)
del train_features,test_features
print(temp.shape)
for col, col_dtype in train_features_dtypes.items():
    if col_dtype == "category":
        temp[col] = temp[col].cat.codes.astype("int16")
        temp[col] -= temp[col].min()
train_features = temp[temp['train'] == 1].copy()
test_features = temp[temp['train'] == 0].copy()
del temp
train_features = train_features.drop(columns='train')
test_features = test_features.drop(columns='train')
print(test_features.shape)
print(train_features.shape)
print(train_features.head())
train_targets_scored = pd.read_csv(TRAIN_TARGETS_SCORED)
list(train_targets_scored.columns)
print(train_targets_scored.shape)
print(train_targets_scored.head())
df = pd.merge(train_features,train_targets_scored,how='inner',on='sig_id')
print(df.shape)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION)
print(sample_submission.shape)
print(sample_submission.head())
submission = sample_submission.copy()
for col in submission.columns[1:]:
    submission[col].values[:] = 0
X_cols = list(train_features.drop('sig_id',axis=1).columns)
print(X_cols)
y_cols = list(train_targets_scored.drop('sig_id',axis=1).columns)


#model
nfolds = 5
kf = KFold(n_splits=nfolds)
pca = PCA(n_components = 300)
svm0 = SVC(C = 0.1,probability =True)
base_model = Pipeline(steps=[('pca', pca), ('svm', svm0)])
mo_base = MultiOutputClassifier(base_model, n_jobs=-1)
xtrain = df[X_cols]
print(xtrain.shape)
ytrain = df[y_cols]
xtest = test_features[X_cols]
prval = np.zeros(ytrain.shape)
prval.shape
for (ff, (id0, id1)) in enumerate(kf.split(xtrain)):
    x0, x1 = xtrain.loc[id0], xtrain.loc[id1]
    y0, y1 = np.array(ytrain.loc[id0]), np.array(ytrain.loc[id1])
    # fix for empty
    check_for_empty_cols = np.where(y0.sum(axis=0) == 0)[0]
    if len(check_for_empty_cols):
        y0[0, check_for_empty_cols] = 1

    # fit model
    mo_base.fit(x0, y0)

    # predicitons
    prv = mo_base.predict_proba(x1)  # [:, 1] see note below, this does not appear to work on a multioutput scenario
    prf = mo_base.predict_proba(xtest)  # [:, 1]

    # some tactical workarounds to get SVC and MultiOutputClassifier outputs into a workable format,
    # as predict_proba generates probability of both pos and neg class, we need to cycle through each
    # target prediction and take the one we want.
    prv_n = []
    for i in range(0, 206):
        #     print(i)
        prv_n.append(prv[i][:, 1])
    prf_n = []
    for i in range(0, 206):
        #     print(i)
        prf_n.append(prf[i][:, 1])
    # generate the prediction
    prval[id1, :] = pd.DataFrame(prv_n).T  # formatting into dataframe and transpose to line up data
    prf_n_df = pd.DataFrame(prf_n).T  # formatting into dataframe and transpose to line up data
    prf_n_df.columns = y_cols
    for i in y_cols:
        submission[i] += prf_n_df[i] / nfolds




from sklearn.linear_model import LogisticRegression
# from cuml import LogisticRegression

N_STARTS = 3
N_SPLITS = 5

res_lr = train_targets.copy()
ss_lr.loc[:, train_targets.columns] = 0
res_lr.loc[:, train_targets.columns] = 0
for tar in tqdm(range(train_targets.shape[1])):
    start_time = time()
    targets = train_targets.values[:, tar]
    if targets.sum() >= N_SPLITS:
        for seed in range(N_STARTS):
            skf = StratifiedKFold(n_splits = N_SPLITS, random_state = seed, shuffle = True)
            for n, (tr, te) in enumerate(skf.split(targets, targets)):
                x_tr, x_val = X_new[tr, tar].reshape(-1, 1), X_new[te, tar].reshape(-1, 1)
                y_tr, y_val = targets[tr], targets[te]
                model = LogisticRegression(C = 35, max_iter = 1000)
                model.fit(x_tr, y_tr)
                ss_lr.loc[:, train_targets.columns[tar]] += model.predict_proba(x_tt_new[:, tar].reshape(-1, 1))[:, 1] / (N_SPLITS * N_STARTS)
                res_lr.loc[te, train_targets.columns[tar]] += model.predict_proba(x_val)[:, 1] / N_STARTS
    score = log_loss(train_targets.loc[:, train_targets.columns[tar]], res_lr.loc[:, train_targets.columns[tar]])
    print(f'[{str(datetime.timedelta(seconds = time() - start_time))[2:7]}] LR Target {tar}:', score)
def log_loss_metric(y_true, y_pred):
    y_pred_clip = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = - np.mean(np.mean(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip), axis = 1))
    return loss
print(f'Model OOF Metric: {log_loss_metric(ytrain, prval)}')
prval_df = pd.DataFrame(prval)
prval_df.columns = y_cols
prval_df.head()
def log_loss_metric_ind(y_true, y_pred):
    y_pred_clip = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = - np.mean(np.mean(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip)))
    return loss
perf_check = []
for i in y_cols:
    perf_check.append((i,log_loss_metric_ind(ytrain[i], prval_df[i])))
results = pd.DataFrame(perf_check)
results.columns = ['target','log_loss']
results.sort_values('log_loss',ascending=False).head(20)
#best performing models
results.sort_values('log_loss',ascending=True).head(20)
submission.to_csv('submission.csv', index=False)


