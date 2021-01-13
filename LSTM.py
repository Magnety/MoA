import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('iterative-stratification-master')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
import random
import time
from tensorboardX import SummaryWriter

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss

import category_encoders as ce

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F

import warnings


def get_logger(filename='models'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger




def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class TrainDataset(Dataset):
    def __init__(self, df, num_features, cat_features, labels):
        self.cont_values = df[num_features].values
        self.cate_values = df[cat_features].values
        self.labels = labels

    def __len__(self):
        return len(self.cont_values)

    def __getitem__(self, idx):
        cont_x = torch.FloatTensor(self.cont_values[idx])
        cate_x = torch.LongTensor(self.cate_values[idx])
        label = torch.tensor(self.labels[idx]).float()

        return cont_x, cate_x, label


class TestDataset(Dataset):
    def __init__(self, df, num_features, cat_features):
        self.cont_values = df[num_features].values
        self.cate_values = df[cat_features].values

    def __len__(self):
        return len(self.cont_values)

    def __getitem__(self, idx):
        cont_x = torch.FloatTensor(self.cont_values[idx])
        cate_x = torch.LongTensor(self.cate_values[idx])

        return cont_x, cate_x




def cate2num(df):
    """hours converts to days"""
    df['cp_time'] = df['cp_time'].map({24: 0, 48: 1, 72: 2})
    df['cp_dose'] = df['cp_dose'].map({'D1': 3, 'D2': 4})
    return df






class LSTMClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gru_hidden_size = 64
        self.lstm_hidden_size = 873
        self.embedding_dropout = nn.Dropout2d(0.2)
        self.lstm = nn.LSTM(self.lstm_hidden_size, cfg.hidden_dim, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(cfg.hidden_dim * 2, self.gru_hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(384, self.gru_hidden_size * 6)
        self.cls = nn.Linear(cfg.hidden_dim, len(cfg.target_cols))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(self.gru_hidden_size * 6, len(cfg.target_cols))
        # self.softmax = nn.LogSoftmax()
    def forward(self, cont_x, cate_x):
        cont_x = torch.unsqueeze(cont_x, 1)
        h_lstm, lstm_out = self.lstm(cont_x)
        h_gru, hh_gru = self.gru(h_lstm)
        hh_gru = hh_gru.view(-1, self.gru_hidden_size * 2)
        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)
        conc = torch.cat((hh_gru, avg_pool, max_pool), 1)
        dropped = self.dropout(conc)
        out = self.out(dropped)
        return out
def train_fn(train_loader, model, optimizer, epoch, scheduler, device):
    losses = AverageMeter()
    model.train()
    for step, (cont_x, cate_x, y) in enumerate(train_loader):
        cont_x, cate_x, y = cont_x.to(device), cate_x.to(device), y.to(device)
        batch_size = cont_x.size(0)
        pred = model(cont_x, cate_x)
        loss = nn.BCEWithLogitsLoss()(pred, y)
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

    return losses.avg
def validate_fn(valid_loader, model, device):
    losses = AverageMeter()

    model.eval()
    val_preds = []

    for step, (cont_x, cate_x, y) in enumerate(valid_loader):

        cont_x, cate_x, y = cont_x.to(device), cate_x.to(device), y.to(device)
        batch_size = cont_x.size(0)

        with torch.no_grad():
            pred = model(cont_x, cate_x)

        loss = nn.BCEWithLogitsLoss()(pred, y)
        losses.update(loss.item(), batch_size)

        val_preds.append(pred.sigmoid().detach().cpu().numpy())

        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

    val_preds = np.concatenate(val_preds)

    return losses.avg, val_preds


def inference_fn(test_loader, model, device):
    model.eval()
    preds = []

    for step, (cont_x, cate_x) in enumerate(test_loader):
        cont_x, cate_x = cont_x.to(device), cate_x.to(device)

        with torch.no_grad():
            pred = model(cont_x, cate_x)

        preds.append(pred.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def run_single_nn(writer,cfg, train, test, folds, num_features, cat_features, target, device, fold_num=0, seed=42):
    # Set seed
    # if not DEBUG:
    logger.info(f'Set seed {seed}')
    seed_everything(seed=seed)
    # loader
    trn_idx = folds[folds['fold'] != fold_num].index
    val_idx = folds[folds['fold'] == fold_num].index
    train_folds = train.loc[trn_idx].reset_index(drop=True)
    valid_folds = train.loc[val_idx].reset_index(drop=True)
    train_target = target[trn_idx]
    valid_target = target[val_idx]
    train_dataset = TrainDataset(train_folds, num_features, cat_features, train_target)
    valid_dataset = TrainDataset(valid_folds, num_features, cat_features, valid_target)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, drop_last=False)

    # model
    # model = TabularNN(cfg)
    model = LSTMClassifier(cfg)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                              max_lr=1e-2, epochs=cfg.epochs, steps_per_epoch=len(train_loader))

    # log
    log_df = pd.DataFrame(columns=(['EPOCH'] + ['TRAIN_LOSS'] + ['VALID_LOSS']))

    # train & validate
    best_loss = np.inf
    for epoch in range(cfg.epochs):
        train_loss = train_fn(train_loader, model, optimizer, epoch, scheduler, device)
        valid_loss, val_preds = validate_fn(valid_loader, model, device)
        writer.add_scalar('train_loss', train_loss, global_step=epoch)
        writer.add_scalar('val_loss', valid_loss, global_step=epoch)
        log_row = {
            'EPOCH': epoch,
            'TRAIN_LOSS': train_loss,
            'VALID_LOSS': valid_loss,
        }
        log_df = log_df.append(pd.DataFrame(log_row, index=[0]), sort=False)
        # logger.info(log_df.tail(1))
        if valid_loss < best_loss:
            # if not DEBUG:
            logger.info(f'epoch{epoch} save best model... {valid_loss}')
            best_loss = valid_loss
            oof = np.zeros((len(train), len(cfg.target_cols)))
            oof[val_idx] = val_preds
            # if not DEBUG:
            torch.save(model.state_dict(), f"LSTM/models/fold{fold_num}_seed{seed}.pth")

    # predictions
    test_dataset = TestDataset(test, num_features, cat_features)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    # model = TabularNN(cfg)
    model = LSTMClassifier(cfg)
    model.load_state_dict(torch.load(f"LSTM/models/fold{fold_num}_seed{seed}.pth"))
    model.to(device)
    predictions = inference_fn(test_loader, model, device)
    # del
    torch.cuda.empty_cache()
    return oof, predictions


def run_kfold_nn(cfg, train, test, folds, num_features, cat_features, target, device, n_fold=5, seed=42):
    oof = np.zeros((len(train), len(cfg.target_cols)))
    predictions = np.zeros((len(test), len(cfg.target_cols)))

    for _fold in range(n_fold):
        # if not DEBUG:
        logger.info("Fold {}".format(_fold))
        writer = SummaryWriter('LSTM/scalar_example%d'%_fold)
        _oof, _predictions = run_single_nn(writer,cfg,train,test,folds,num_features,cat_features,target,device, fold_num=_fold,seed=seed)
        oof += _oof
        predictions += _predictions / n_fold

    score = 0
    for i in range(target.shape[1]):
        _score = log_loss(target[:, i], oof[:, i])
        score += _score / target.shape[1]
    # if not DEBUG:
    logger.info(f"CV score: {score}")
    return oof, predictions

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DEBUG = True
    logger = get_logger()
    seed_everything(seed=42)
    DATA_PATH = 'dataset/'

    TRAIN_FEATURES = DATA_PATH + 'train_features.csv'
    TEST_FEATURES = DATA_PATH + 'test_features.csv'
    TRAIN_TARGETS_NON_SCORED = DATA_PATH + 'train_targets_nonscored.csv'
    TRAIN_TARGETS_SCORED = DATA_PATH + 'train_targets_scored.csv'
    SAMPLE_SUBMISSION = DATA_PATH + 'sample_submission.csv'
    train_features = pd.read_csv(TRAIN_FEATURES)
    train_targets_scored = pd.read_csv(TRAIN_TARGETS_SCORED)
    train_targets_nonscored = pd.read_csv(TRAIN_TARGETS_NON_SCORED)
    test_features = pd.read_csv(TEST_FEATURES)
    submission = pd.read_csv(SAMPLE_SUBMISSION)

    train = train_features.merge(train_targets_scored, on='sig_id')
    target_cols = [c for c in train_targets_scored.columns if c not in ['sig_id']]
    cols = target_cols + ['cp_type']
    train = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
    test = test_features[test_features['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
    print(train.shape, test.shape)
    print(train_features.shape, test_features.shape)

    folds = train.copy()
    Fold = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[target_cols])):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)
    print(folds.shape)
    cat_features = ['cp_dose']
    num_features = [c for c in train.columns if train.dtypes[c] != 'object']
    num_features = [c for c in num_features if c not in cat_features]
    num_features = [c for c in num_features if c not in target_cols]
    target = train[target_cols].values
    train = cate2num(train)
    test = cate2num(test)


    class CFG:
        max_grad_norm = 1000
        gradient_accumulation_steps = 1
        hidden_size = 1024
        hidden_dim = 256
        dropout = 0.5
        lr = 1e-2
        weight_decay = 1e-6
        batch_size = 256
        epochs = 20
        # total_cate_size=5
        # emb_size=4
        num_features = num_features
        cat_features = cat_features
        target_cols = target_cols
    oof = np.zeros((len(train), len(CFG.target_cols)))
    predictions = np.zeros((len(test), len(CFG.target_cols)))

    SEED = [0]
    for seed in SEED:
        _oof, _predictions = run_kfold_nn(CFG, train, test, folds, num_features, cat_features, target, device, n_fold=5,
                                          seed=seed)
        oof += _oof / len(SEED)
        predictions += _predictions / len(SEED)
    score = 0
    for i in range(target.shape[1]):
        _score = log_loss(target[:, i], oof[:, i])
        score += _score / target.shape[1]
    if not DEBUG:
        logger.info(f"Seed Averaged CV score: {score}")

