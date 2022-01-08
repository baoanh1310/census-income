import os

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

def read_data(path):
	df = pd.read_csv(path, encoding='latin-1')
	return df

def prepare_data(df, test_size=0.3):
	# Encode ? as NaN
	df[df=='?'] = np.nan

	# Impute missing values with mode
	for col in ['workclass', 'occupation', 'native.country']:
		df[col].fillna(df[col].mode()[0], inplace=True)

	# Check again for missing values
	# print(df.isnull().sum())

	# Prepare train, test data
	X = df.drop(['income'], axis=1)
	y = df['income']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)

	categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
	for feature in categorical:
		le = preprocessing.LabelEncoder()
		X_train[feature] = le.fit_transform(X_train[feature])
		X_test[feature] = le.transform(X_test[feature])

	# Feature scaling
	scaler = StandardScaler()
	X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
	X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

	return X_train, y_train, X_test, y_test

def train_logreg(X_train, y_train, X_test, y_test, model_path):
	logreg = LogisticRegression()
	logreg.fit(X_train, y_train)
	y_pred = logreg.predict(X_test)

	# Save and load model
	if not os.path.exists(model_path):
		dump(logreg, model_path)

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

