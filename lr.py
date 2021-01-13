import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

DATA_PATH = 'dataset/'

TRAIN_FEATURES = DATA_PATH + 'train_features.csv'
TEST_FEATURES = DATA_PATH + 'test_features.csv'
TRAIN_TARGETS_NON_SCORED = DATA_PATH + 'train_targets_nonscored.csv'
TRAIN_TARGETS_SCORED = DATA_PATH + 'train_targets_scored.csv'
SAMPLE_SUBMISSION = DATA_PATH + 'sample_submission.csv'

train_features = pd.read_csv(TRAIN_FEATURES)
train_targets = pd.read_csv(TRAIN_TARGETS_SCORED)
test_features = pd.read_csv(TEST_FEATURES)
test_predictions = pd.read_csv(SAMPLE_SUBMISSION)
