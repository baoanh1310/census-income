import os

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

def read_data(path):
	df = pd.read_csv(path, encoding='latin-1')
	return df

def prepare_data(train_df, test_df):
	# Encode ? as NaN
	train_df[train_df=='?'] = np.nan
	test_df[test_df=='?'] = np.nan

	# Impute missing values with mode
	for col in ['workclass', 'occupation', 'native.country']:
		train_df[col].fillna(train_df[col].mode()[0], inplace=True)
		test_df[col].fillna(test_df[col].mode()[0], inplace=True)

	# Check again for missing values
	# print(df.isnull().sum())

	# Prepare train, test data
	X_train = train_df.drop(['income'], axis=1)
	y_train = train_df['income']
	X_test = test_df.drop(['income'], axis=1)
	y_test = test_df['income']

	categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
	for feature in categorical:
		le = preprocessing.LabelEncoder()
		X_train[feature] = le.fit_transform(X_train[feature])
		X_test[feature] = le.transform(X_test[feature])

	# Feature scaling
	scaler = StandardScaler()
	X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
	X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

	return X_train, y_train, X_test, y_test

def prepare_oversampling_data(train_df, test_df):
	# Encode ? as NaN
	train_df[train_df=='?'] = np.nan
	test_df[test_df=='?'] = np.nan

	# Oversampling data
	train_greater_df = train_df[train_df['income'] == ">50K"]
	train_df = pd.concat([train_df, train_greater_df])

	# Impute missing values with mode
	for col in ['workclass', 'occupation', 'native.country']:
		train_df[col].fillna(train_df[col].mode()[0], inplace=True)
		test_df[col].fillna(test_df[col].mode()[0], inplace=True)

	# Check again for missing values
	# print(df.isnull().sum())

	# Prepare train, test data
	X_train = train_df.drop(['income'], axis=1)
	y_train = train_df['income']
	X_test = test_df.drop(['income'], axis=1)
	y_test = test_df['income']

	categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
	for feature in categorical:
		le = preprocessing.LabelEncoder()
		X_train[feature] = le.fit_transform(X_train[feature])
		X_test[feature] = le.transform(X_test[feature])

	# Feature scaling
	scaler = StandardScaler()
	X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
	X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

	return X_train, y_train, X_test, y_test

def prepare_drop_data(train_df, test_df):
	# Encode ? as NaN
	train_df[train_df=='?'] = np.nan
	test_df[test_df=='?'] = np.nan

	# Impute missing values with mode
	for col in ['workclass', 'occupation', 'native.country']:
		train_df[col].fillna(train_df[col].mode()[0], inplace=True)
		test_df[col].fillna(test_df[col].mode()[0], inplace=True)

	# Prepare train, test data
	X_train = train_df.drop(['income'], axis=1)
	y_train = train_df['income']
	X_test = test_df.drop(['income'], axis=1)
	y_test = test_df['income']

	categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
	for feature in categorical:
		le = preprocessing.LabelEncoder()
		X_train[feature] = le.fit_transform(X_train[feature])
		X_test[feature] = le.transform(X_test[feature])

	# Feature scaling
	scaler = StandardScaler()
	X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
	X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

	X_train = X_train.drop(['workclass', 'fnlwgt', 'marital.status', 'relationship'], axis=1)
	X_test = X_test.drop(['workclass', 'fnlwgt', 'marital.status', 'relationship'], axis=1)

	return X_train, y_train, X_test, y_test

def train_logreg(X_train, y_train, X_test, y_test, model_path):
	logreg = LogisticRegression()
	logreg.fit(X_train, y_train)
	y_pred = logreg.predict(X_test)

	# Save model
	if not os.path.exists(model_path):
		dump(logreg, model_path)

def train_decision_tree(X_train, y_train, X_test, y_test, model_path):
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)

	# Save model
	if not os.path.exists(model_path):
		dump(clf, model_path)

def train_knn(X_train, y_train, X_test, y_test, model_path):
	best_score = 0
	for k in range(1, 26):
		knn = KNeighborsClassifier(n_neighbors=k)
		knn.fit(X_train, y_train)
		y_pred = knn.predict(X_test)
		score = accuracy_score(y_test, y_pred)
		if score > best_score:
			best_score = score
			dump(knn, model_path)

def evaluate(model_path, X_test, y_test):
	clf = load(model_path)
	y_pred = clf.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print("\nAccuracy: {:2}%".format(accuracy * 100))
	print("Confusion matrix: \n", confusion_matrix(y_test, y_pred))
	print("Classification report: \n", classification_report(y_test, y_pred))

def predict(model_path, X_test, y_test, index):
	clf = load(model_path)
	print("Input data: \n", X_test.iloc[[index]])
	print("Raw label: ", np.array(y_test.iloc[[index]])[0])
	print("Result: ", clf.predict(X_test.iloc[[index]])[0])

def user_predict(model_path, user_input):
	clf = load(model_path)

	for col in ['workclass', 'occupation', 'native.country']:
		user_input[col].fillna(user_input[col].mode()[0], inplace=True)
	categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
	for feature in categorical:
		le = preprocessing.LabelEncoder()
		user_input[feature] = le.fit_transform(user_input[feature])
	# Feature scaling
	scaler = StandardScaler()
	user_input = pd.DataFrame(scaler.fit_transform(user_input), columns=user_input.columns)

	return clf.predict(user_input)[0]

