import os

from config import BASIC_MODEL_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, DECISION_TREE_MODEL_PATH, KNN_MODEL_PATH, DROP_MODEL_PATH
from utils import *

# Import dataset
train_df = read_data(TRAIN_DATA_PATH)
test_df = read_data(TEST_DATA_PATH)

# Prepare data
X_train, y_train, X_test, y_test = prepare_data(train_df, test_df)

# Training
train_logreg(X_train, y_train, X_test, y_test, BASIC_MODEL_PATH)

# Evaluation
print("Evaluating for Logistic Regression Model...")
evaluate(BASIC_MODEL_PATH, X_test, y_test)

# Training drop model
X_train, y_train, X_test, y_test = prepare_drop_data(train_df, test_df)
train_logreg(X_train, y_train, X_test, y_test, DROP_MODEL_PATH)
# Evaluation
print("Evaluating for Drop Logistic Regression Model...")
evaluate(DROP_MODEL_PATH, X_test, y_test)

# Train decision tree
# train_decision_tree(X_train, y_train, X_test, y_test, DECISION_TREE_MODEL_PATH)
# print("Evaluating for Decision Tree Model...")
# evaluate(DECISION_TREE_MODEL_PATH, X_test, y_test)

# Train KNN
# train_knn(X_train, y_train, X_test, y_test, KNN_MODEL_PATH)
# print("Evaluating KNN Model...")
# evaluate(KNN_MODEL_PATH, X_test, y_test)