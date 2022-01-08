import os

import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

from config import BASIC_MODEL_PATH

# Import dataset
filename = './data/adult.csv'
df = pd.read_csv(filename, encoding='latin-1')

# Preprocessing data
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
for feature in categorical:
	le = preprocessing.LabelEncoder()
	X_train[feature] = le.fit_transform(X_train[feature])
	X_test[feature] = le.transform(X_test[feature])

# Feature scaling
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Training
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Accuracy score with all the features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# Save and load model
if not os.path.exists(BASIC_MODEL_PATH):
	dump(logreg, BASIC_MODEL_PATH)

clf = load(BASIC_MODEL_PATH)

print("Loaded model accuracy score: ", accuracy_score(y_test, clf.predict(X_test)))