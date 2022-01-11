import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

from utils import read_data
from config import TRAIN_DATA_PATH, TEST_DATA_PATH

train_df = read_data(TRAIN_DATA_PATH)
test_df = read_data(TEST_DATA_PATH)

train_df[train_df=='?'] = np.nan
for col in ['workclass', 'occupation', 'native.country']:
	train_df[col].fillna(train_df[col].mode()[0], inplace=True)

categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country', 'income']
for feature in categorical:
	le = preprocessing.LabelEncoder()
	train_df[feature] = le.fit_transform(train_df[feature])

train_correlation = train_df.corr()

# Plotting correlation map
fig, ax = plt.subplots(figsize=(16,12))
dataplot = sns.heatmap(train_correlation, cmap="YlGnBu", annot=True)

plt.savefig("./plots/correlation.png")