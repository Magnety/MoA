import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import random
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau


def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

DATA_PATH = 'dataset/'

TRAIN_FEATURES = DATA_PATH + 'train_features.csv'
TEST_FEATURES = DATA_PATH + 'test_features.csv'
TRAIN_TARGETS_NON_SCORED = DATA_PATH + 'train_targets_nonscored.csv'
TRAIN_TARGETS_SCORED = DATA_PATH + 'train_targets_scored.csv'
SAMPLE_SUBMISSION = DATA_PATH + 'sample_submission.csv'

seed_everything(42)
train_features = pd.read_csv(TRAIN_FEATURES)
train_targets = pd.read_csv(TRAIN_TARGETS_SCORED)
COLS = ['cp_type','cp_dose']
FE = []
for col in COLS:
    for mod in train_features[col].unique():
        FE.append(mod)
        train_features[mod] = (train_features[col] == mod).astype(int)
del train_features['sig_id']
del train_features['cp_type']
del train_features['cp_dose']
FE+=list(train_features.columns)
del train_targets['sig_id']
def model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(877),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(4096, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(4096, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(206, activation="sigmoid")
        ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=2.75e-5), loss='binary_crossentropy', metrics=["accuracy", "AUC"])
    return model
from sklearn.model_selection import KFold

NFOLD = 5
kf = KFold(n_splits=NFOLD)
BATCH_SIZE=128
EPOCHS=35

test_features = pd.read_csv(TEST_FEATURES)
for col in COLS:
    for mod in test_features[col].unique():
        test_features[mod] = (test_features[col] == mod).astype(int)
sig_id = pd.DataFrame()
sig_id = test_features.pop('sig_id')
del test_features['cp_type']
del test_features['cp_dose']
pe = np.zeros((test_features.shape[0], 206))
train_features = train_features.values
train_targets = train_targets.values
pred = np.zeros((train_features.shape[0], 206))

cnt=0
import datetime
log_dir = os.path.join('logs',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


for tr_idx, val_idx in kf.split(train_features):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')
    cnt += 1
    print(f"FOLD {cnt}")
    net = model()
    net.fit(train_features[tr_idx], train_targets[tr_idx], batch_size=BATCH_SIZE, epochs=EPOCHS,
            validation_data=(train_features[val_idx], train_targets[val_idx]), verbose=0, callbacks=[tensorboard_callback])



    print("train", net.evaluate(train_features[tr_idx], train_targets[tr_idx], verbose=0, batch_size=BATCH_SIZE))
    print("val", net.evaluate(train_features[val_idx], train_targets[val_idx], verbose=0, batch_size=BATCH_SIZE))


    print("predict val...")
    pred[val_idx] = net.predict(train_features[val_idx], batch_size=BATCH_SIZE, verbose=0)
    total = len(val_idx)
    """acc = 0
    sensitivity =0
    specificity =0
    for i in range(206):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        sum = 0
        for j in val_idx:
            print('j',j)
            print('pred:',pred[j][i])
            
            if pred[j][i]==1:
                if pred[j][i] == train_targets[j][i]:
                    tp +=1
                else:
                    fp+=1
            if pred[j][i]==0:
                if pred[j][i] == train_targets[j][i]:
                    tn +=1
                else:
                    fn+=1
        print('sum',sum)
        print(tp+tn+fp+fn)
        acc += (tp+tn)/(tp+tn+fp+fn)
        sensitivity += tp/(tp+fn)
        specificity += tn/(fn+tn)
    acc = acc/206
    sensitivity = sensitivity/206
    specificity = specificity/206
    print('acc:',acc)
    print('sensitivity:',sensitivity)
    print('specificity:',specificity)
"""
    print("predict test...")
    pe += net.predict(test_features, batch_size=BATCH_SIZE, verbose=0) / NFOLD
pe.shape

columns = pd.read_csv(TRAIN_TARGETS_SCORED)
del columns['sig_id']
sub = pd.DataFrame(data=pe, columns=columns.columns)
sample = pd.read_csv(SAMPLE_SUBMISSION)
sub.insert(0, column = 'sig_id', value=sample['sig_id'])
sub.to_csv('submission.csv', index=False)
def Diff(list1, list2):
    return (list(list(set(list1)-set(list2)) + list(set(list2)-set(list1))))
Diff (sub.columns, pd.read_csv(SAMPLE_SUBMISSION).columns)