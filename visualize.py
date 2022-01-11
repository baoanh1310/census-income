import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import plot_confusion_matrix
from joblib import load

from utils import read_data, prepare_data
from config import TRAIN_DATA_PATH, TEST_DATA_PATH, BASIC_MODEL_PATH

def visualize_corr():
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

def visualize_confusion_matrix():
	clf = load(BASIC_MODEL_PATH)
	train_df = read_data(TRAIN_DATA_PATH)
	test_df = read_data(TEST_DATA_PATH)
	X_train, y_train, X_test, y_test = prepare_data(train_df, test_df)
	plot_confusion_matrix(clf, X_test, y_test, normalize='true')
	plt.savefig("./plots/confusion_matrix.png")

if __name__ == "__main__":
	# visualize_corr()
	visualize_confusion_matrix()