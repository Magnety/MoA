import os
import numpy as np
import pandas as pd
import random
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import DataLoader, Dataset
from torch.utils import data
from torchvision import models

from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import log_loss
sys.path.append('iterative-stratification-master')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=42)
DATA_PATH = 'dataset/'

TRAIN_FEATURES = DATA_PATH + 'train_features.csv'
TEST_FEATURES = DATA_PATH + 'test_features.csv'
TRAIN_TARGETS_NON_SCORED = DATA_PATH + 'train_targets_nonscored.csv'
TRAIN_TARGETS_SCORED = DATA_PATH + 'train_targets_scored.csv'
SAMPLE_SUBMISSION = DATA_PATH + 'sample_submission.csv'

train_features0 = pd.read_csv(TRAIN_FEATURES)
train_targets_scored = pd.read_csv(TRAIN_TARGETS_SCORED)
train_targets_nonscored = pd.read_csv(TRAIN_TARGETS_NON_SCORED)
test_features0 = pd.read_csv(TEST_FEATURES)
submission = pd.read_csv(SAMPLE_SUBMISSION)

GENES = [col for col in train_features0.columns if col.startswith('g-')]
CELLS = [col for col in train_features0.columns if col.startswith('c-')]
for col in (GENES + CELLS):
    transformer = QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal")
    vec_len = len(train_features0[col].values)
    vec_len_test = len(test_features0[col].values)
    raw_vec = train_features0[col].values.reshape(vec_len, 1)
    transformer.fit(raw_vec)
    train_features0[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]
    test_features0[col] = transformer.transform(test_features0[col].values.reshape(vec_len_test, 1)).reshape(1, vec_len_test)[0]


def fe_stats(train, test, extra=True):
    features_g = [col for col in train.columns if 'g-' in col]
    features_c = [col for col in train.columns if 'c-' in col]

    for df in [train, test]:
        df['g_sum'] = df[features_g].sum(axis=1)
        df['g_mean'] = df[features_g].mean(axis=1)
        df['g_std'] = df[features_g].std(axis=1)
        df['g_kurt'] = df[features_g].kurtosis(axis=1)
        df['g_skew'] = df[features_g].skew(axis=1)
        df['g_std'] = df[features_g].std(axis=1)
        df['c_sum'] = df[features_c].sum(axis=1)
        df['c_mean'] = df[features_c].mean(axis=1)
        df['c_std'] = df[features_c].std(axis=1)
        df['c_kurt'] = df[features_c].kurtosis(axis=1)
        df['c_skew'] = df[features_c].skew(axis=1)
        df['c_std'] = df[features_c].std(axis=1)
        df['gc_sum'] = df[features_g + features_c].sum(axis=1)
        df['gc_mean'] = df[features_g + features_c].mean(axis=1)
        df['gc_std'] = df[features_g + features_c].std(axis=1)
        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis=1)
        df['gc_skew'] = df[features_g + features_c].skew(axis=1)
        df['gc_std'] = df[features_g + features_c].std(axis=1)

        #
        if extra:
            df['g_sum-c_sum'] = df['g_sum'] - df['c_sum']
            df['g_mean-c_mean'] = df['g_mean'] - df['c_mean']
            df['g_std-c_std'] = df['g_std'] - df['c_std']
            df['g_kurt-c_kurt'] = df['g_kurt'] - df['c_kurt']
            df['g_skew-c_skew'] = df['g_skew'] - df['c_skew']

            df['g_sum*c_sum'] = df['g_sum'] * df['c_sum']
            df['g_mean*c_mean'] = df['g_mean'] * df['c_mean']
            df['g_std*c_std'] = df['g_std'] * df['c_std']
            df['g_kurt*c_kurt'] = df['g_kurt'] * df['c_kurt']
            df['g_skew*c_skew'] = df['g_skew'] * df['c_skew']

            df['g_sum/c_sum'] = df['g_sum'] / df['c_sum']
            df['g_mean/c_mean'] = df['g_mean'] / df['c_mean']
            df['g_std/c_std'] = df['g_std'] / df['c_std']
            df['g_kurt/c_kurt'] = df['g_kurt'] / df['c_kurt']
            df['g_skew/c_skew'] = df['g_skew'] / df['c_skew']

    return train, test


def fe_pca(train, test, n_components_g=50, n_components_c=10, SEED=123):
    features_g = [col for col in train.columns if 'g-' in col]
    features_c = [col for col in train.columns if 'c-' in col]

    def create_pca(train, test, features, kind='g', n_components=n_components_g):
        train_ = train[features].copy()
        test_ = test[features].copy()

        # data = pd.concat([train_, test_], axis=0)
        pca = PCA(n_components=n_components, random_state=SEED)
        # data = pca.fit_transform(data)
        data1 = pca.fit_transform(train_)
        data2 = pca.transform(test_)
        data = np.concatenate((data1, data2), axis=0)

        columns = [f'pca_{kind}{i + 1}' for i in range(n_components)]
        data = pd.DataFrame(data, columns=columns)
        train_ = data.iloc[:train.shape[0]]
        test_ = data.iloc[train.shape[0]:].reset_index(drop=True)
        train = pd.concat([train.reset_index(drop=True),
                           train_.reset_index(drop=True)], axis=1)
        test = pd.concat([test.reset_index(drop=True),
                          test_.reset_index(drop=True)], axis=1)
        return train, test

    train, test = create_pca(train, test, features_g, kind='g', n_components=n_components_g)
    train, test = create_pca(train, test, features_c, kind='c', n_components=n_components_c)

    return train, test


from sklearn.cluster import KMeans


def fe_cluster(train, test, n_clusters_g=15, n_clusters_c=5, SEED=123):
    features_g = [col for col in train.columns if 'g-' in col]
    features_c = [col for col in train.columns if 'c-' in col]

    def create_cluster(train, test, features, kind='g', n_clusters=n_clusters_g):
        train_ = train[features].copy()
        test_ = test[features].copy()

        # StandardScaler
        scaler = preprocessing.StandardScaler()
        train_.iloc[:, :] = scaler.fit_transform(train_)
        test_.iloc[:, :] = scaler.transform(test_)

        # data = pd.concat([train_, test_], axis = 0)
        # kmeans = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
        # train[f'clusters_{kind}'] = kmeans.labels_[:train.shape[0]]
        # test[f'clusters_{kind}'] = kmeans.labels_[train.shape[0]:]

        kmeans = KMeans(n_clusters=n_clusters, random_state=SEED).fit(train_)
        train[f'clusters_{kind}'] = kmeans.predict(train_.values)
        test[f'clusters_{kind}'] = kmeans.predict(test_.values)
        train = pd.get_dummies(train, columns=[f'clusters_{kind}'])
        test = pd.get_dummies(test, columns=[f'clusters_{kind}'])
        return train, test

    train, test = create_cluster(train, test, features_g, kind='g', n_clusters=n_clusters_g)
    train, test = create_cluster(train, test, features_c, kind='c', n_clusters=n_clusters_c)
    return train, test


from sklearn.feature_selection import VarianceThreshold


def fe_feature_selection(train_features0, test_features0, threshold=0.8):
    var_thresh = VarianceThreshold(threshold)  # <-- Update
    data = train_features0.append(test_features0)
    data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])
    cols = test_features0.iloc[:, 4:].columns[var_thresh.get_support()]  # 获取保留下来的列名

    train_features0_transformed = pd.DataFrame(data_transformed[: train_features0.shape[0]], columns=cols)
    test_features0_transformed = pd.DataFrame(data_transformed[-test_features0.shape[0]:], columns=cols)

    train_features0 = pd.DataFrame(train_features0[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(-1, 4), \
                                   columns=['sig_id', 'cp_type', 'cp_time', 'cp_dose'])
    train_features0 = pd.concat([train_features0, train_features0_transformed], axis=1)

    test_features0 = pd.DataFrame(test_features0[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(-1, 4), \
                                  columns=['sig_id', 'cp_type', 'cp_time', 'cp_dose'])
    test_features0 = pd.concat([test_features0, test_features0_transformed], axis=1)

    return train_features0, test_features0


class MoATrainDataset():

    def __init__(self, train_data, train_labels, feature_cols):

        super(MoATrainDataset).__init__()
        pad_size_l = int((32 * 32 - len(feature_cols)) // 2)
        if (32 * 32 - len(feature_cols)) % 2 == 0:
            pad_size_r = pad_size_l
        else:
            pad_size_r = pad_size_l + 1
        self.X = np.pad(train_data, ((0, 0), (pad_size_l, pad_size_r)), 'constant',
                        constant_values=(0)).reshape(-1, 1, 32, 32)

        self.X = torch.from_numpy(self.X).float()
        self.Y = torch.from_numpy(train_labels).float()

    def __getitem__(self, index):

        image = self.X[index]
        label = self.Y[index]

        dct = {'x': image,
               'y': label}

        return dct

    def __len__(self):
        return len(self.X)


class MoATestDataset():

    def __init__(self, test_data, feature_cols):

        super(MoATestDataset).__init__()
        pad_size_l = int((32 * 32 - len(feature_cols)) / 2)
        if (32 * 32 - len(feature_cols)) % 2 == 0:
            pad_size_r = pad_size_l
        else:
            pad_size_r = pad_size_l + 1
        self.X = np.pad(test_data, ((0, 0), (pad_size_l, pad_size_r)), 'constant',
                        constant_values=(0)).reshape(-1, 1, 32, 32)

        self.X = torch.from_numpy(self.X).float()

    def __getitem__(self, index):

        image = self.X[index]
        dct = {'x': image}
        return dct

    def __len__(self):
        return len(self.X)


def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()  # set the model to training mode
    final_loss = 0  # Initialise final loss to zero

    for data in dataloader:
        optimizer.zero_grad()  # every time we use the gradients to update the parameters, we need to zero the gradients afterwards
        inputs, targets = data['x'].to(device), data['y'].to(
            device)  # Sending data to GPU(cuda) if gpu is available otherwise CPU
        outputs = model(inputs)  # output
        loss = loss_fn(outputs, targets)  # loss function
        loss.backward()  # compute gradients(work its way BACKWARDS from the specified loss)
        optimizer.step()  # gradient optimisation
        scheduler.step()

        final_loss += loss.item()  # Final loss

    final_loss /= len(dataloader)  # average loss

    return final_loss


def valid_fn(model, loss_fn, dataloader, device):
    model.eval()  # set the model to evaluation/validation mode
    final_loss = 0  # Initialise validation final loss to zero
    valid_preds = []  # Empty list for appending prediction

    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(
            device)  # Sending data to GPU(cuda) if gpu is available otherwise CPU
        outputs = model(inputs)  # output
        loss = loss_fn(outputs, targets)  # loss calculation

        final_loss += loss.item()  # final validation loss
        valid_preds.append(
            outputs.sigmoid().detach().cpu().numpy())  # get CPU tensor as numpy array # cannot get GPU tensor as numpy array directly

    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)  # concatenating predictions under valid_preds

    return final_loss, valid_preds


def inference_fn(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():  # need to use NO_GRAD to keep the update out of the gradient computation
            outputs = model(inputs)

        preds.append(outputs.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16 #16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer3 = self._make_layer(block, 32, num_blocks[2], stride=2)
        self.linear = nn.Linear(32, num_classes)
        self.apply(_weights_init)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes):
    return ResNet(BasicBlock, [3, 3, 3], num_classes)


def resnet32(num_classes):
    return ResNet(BasicBlock, [5, 5, 5], num_classes)


def resnet44(num_classes):
    return ResNet(BasicBlock, [7, 7, 7], num_classes)


def resnet56(num_classes):
    return ResNet(BasicBlock, [9, 9, 9], num_classes)


def resnet110(num_classes):
    return ResNet(BasicBlock, [18, 18, 18], num_classes)


def resnet1202(num_classes):
    return ResNet(BasicBlock, [200, 200, 200], num_classes)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))
class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


def run_training(fold, seed,writer):
    seed_everything(seed)  # seeding

    # train = process_data(folds) #converting train set categorical columns
    # test_ = process_data(test) #converting test set categorical columns

    trn_idx = folds[folds['kfold'] != fold].index
    val_idx = folds[folds['kfold'] == fold].index

    # cross validation (splitting randomly train and validation set )
    train_df = folds[folds['kfold'] != fold].reset_index(drop=True)
    valid_df = folds[folds['kfold'] == fold].reset_index(drop=True)

    x_train, y_train = train_df[feature_cols].values, train_df[target_cols].values
    x_valid, y_valid = valid_df[feature_cols].values, valid_df[target_cols].values

    train_dataset = MoATrainDataset(x_train, y_train, feature_cols)
    valid_dataset = MoATrainDataset(x_valid, y_valid, feature_cols)
    # train batch optimization with specified
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # specified batch size, mini-batch (of size BATCH_SIZE), one epoch has N/BATCH_SIZE updates
    # optimization with random batches of size BATCH_SIZE for validation set
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = resnet32(num_classes=206)
    model.to(device)

    # using adam optimizer for optimization
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    # learning rate scheduler, fit one cycle
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                              pct_start=0.1,
                                              div_factor=1e3,
                                              max_lr=1e-2,
                                              epochs=EPOCHS,
                                              steps_per_epoch=len(trainloader))
    # Binary Cross entropy loss
    loss_fn = nn.BCEWithLogitsLoss()
    loss_tr = SmoothBCEwLogits(smoothing=0.0015)

    # early stopping to prevent overfitting and computational time
    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0

    oof = np.zeros((len(folds), len(target_cols)))
    best_loss = np.inf

    for epoch in range(EPOCHS):

        train_loss = train_fn(model, optimizer, scheduler, loss_tr, trainloader, device)  # training loss
        print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}")
        valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, device)  # validation loss
        print(f"FOLD: {fold}, EPOCH: {epoch}, valid_loss: {valid_loss}")
        writer.add_scalar('train_loss', train_loss, global_step=epoch)
        writer.add_scalar('val_loss', valid_loss, global_step=epoch)
        if valid_loss < best_loss:

            best_loss = valid_loss
            oof[val_idx] = valid_preds
            torch.save(model.state_dict(), f"FOLD{fold}_.pth")

        elif (EARLY_STOP == True):

            early_step += 1
            if (early_step >= early_stopping_steps):
                break

    x_test = test_features[feature_cols].values
    testdataset = MoATestDataset(x_test, feature_cols)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)

    model = resnet32(num_classes=206)
    model.to(device)
    model.load_state_dict(torch.load(f"FOLD{fold}_.pth"))
    model.to(device)
    predictions = np.zeros((len(test_features), len(target_cols)))
    predictions = inference_fn(model, testloader, device)

    return oof, predictions


def run_k_fold(NFOLDS, seed):
    oof = np.zeros((len(folds), len(target_cols)))
    predictions = np.zeros((len(test_features), len(target_cols)))

    for fold in range(NFOLDS):
        writer = SummaryWriter('CNN/scalar_example%d'%fold)
        oof_, pred_ = run_training(fold, seed,writer)

        predictions += pred_ / NFOLDS
        oof += oof_

    return oof, predictions
EPOCHS =50
BATCH_SIZE = 256 #128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
NFOLDS = 7
EARLY_STOPPING_STEPS = 10
EARLY_STOP = False
ADD_PCA = True
ADD_STATS = False
ADD_CLUSTER = False
IS_FE_SELECT = True
PCA_G_FEATS = 50
PCA_C_FEATS = 10
if ADD_STATS:
    train_features0, test_features0 = fe_stats(train_features0, test_features0, extra=True)
if ADD_PCA:
    train_features0, test_features0 = fe_pca(train_features0, test_features0,
                                             n_components_g=PCA_G_FEATS,
                                             n_components_c=PCA_C_FEATS,
                                             SEED=123)
if ADD_CLUSTER:
    train_features0, test_features0 = fe_cluster(train_features0, test_features0)
if IS_FE_SELECT:
    train_features0, test_features0 = fe_feature_selection(train_features0, test_features0, threshold=0.8)

print(train_features0.shape)

# 裁剪
train_features = train_features0.iloc[:, 4:]
test_features = test_features0.iloc[:, 4:]

target_cols = train_targets_scored.drop('sig_id', axis=1).columns.values.tolist()
print(train_features.shape)
print(len(target_cols))
train_data = train_features.copy()
train_labels = train_targets_scored.iloc[:, 1:]

folds = train_data.copy()
mlsk = MultilabelStratifiedKFold(n_splits=7)
for f, (t_idx, v_idx) in enumerate(mlsk.split(X=train_data, y=train_labels)):
    folds.loc[v_idx, 'kfold'] = int(f)
folds['kfold'] = folds['kfold'].astype(int)
folds.head()

# # StandardScaler
# scaler = preprocessing.StandardScaler()
# folds.iloc[:, :-1] = scaler.fit_transform(train_features)
# test_features.iloc[:, :] = scaler.transform(test_features)


# Max-Min Scaler
scaler = preprocessing.MinMaxScaler()
folds.iloc[:, :-1] = scaler.fit_transform(train_features)
test_features.iloc[:, :] = scaler.transform(test_features)

folds = pd.concat([folds, train_labels], axis=1)

feature_cols = [c for c in test_features.columns]
print(len(feature_cols))
folds.head()
SEED = [0]  # , 1, 2]
oof = np.zeros((len(folds), len(target_cols)))
predictions = np.zeros((len(test_features), len(target_cols)))

for seed in SEED:
    oof_, predictions_ = run_k_fold(NFOLDS, seed)
    oof += oof_ / len(SEED)
    predictions += predictions_ / len(SEED)