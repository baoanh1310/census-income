import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import plot_confusion_matrix
from joblib import load

from utils import *
from config import *

def visualize_corr(method='pearson'):
	train_df = read_data(TRAIN_DATA_PATH)
	test_df = read_data(TEST_DATA_PATH)

	train_df[train_df=='?'] = np.nan
	for col in ['workclass', 'occupation', 'native.country']:
		train_df[col].fillna(train_df[col].mode()[0], inplace=True)

	categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country', 'income']
	for feature in categorical:
		le = preprocessing.LabelEncoder()
		train_df[feature] = le.fit_transform(train_df[feature])

	train_correlation = train_df.corr(method=method)

	# Plotting correlation map
	fig, ax = plt.subplots(figsize=(16,12))
	dataplot = sns.heatmap(train_correlation, cmap="YlGnBu", annot=True)

	img_name = "./plots/" + method + "_correlation.png"
	plt.savefig(img_name)

def visualize_confusion_matrix(model_path):
	clf = load(model_path)
	train_df = read_data(TRAIN_DATA_PATH)
	test_df = read_data(TEST_DATA_PATH)
	img_name = "./plots/confusion_matrix" + "_"

	if model_path == BASIC_MODEL_PATH:
		img_name = img_name + "basic.png"
		X_train, y_train, X_test, y_test = prepare_data(train_df, test_df)
	elif model_path == DROP_MODEL_PATH:
		img_name = img_name + "drop.png"
		X_train, y_train, X_test, y_test = prepare_drop_data(train_df, test_df)
	elif model_path == OVERSAMPLING_MODEL_PATH:
		img_name = img_name + "oversampling.png"
		X_train, y_train, X_test, y_test = prepare_oversampling_data(train_df, test_df)

	plot_confusion_matrix(clf, X_test, y_test, normalize='true')
	plt.savefig(img_name)

if __name__ == "__main__":
	# visualize_corr('spearman')
	visualize_confusion_matrix(DROP_MODEL_PATH)
	visualize_confusion_matrix(OVERSAMPLING_MODEL_PATH)
