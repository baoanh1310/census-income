import os

from config import BASIC_MODEL_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH
from utils import read_data, prepare_data, train_logreg, evaluate

# Import dataset
train_df = read_data(TRAIN_DATA_PATH)
test_df = read_data(TEST_DATA_PATH)

# Prepare data
X_train, y_train, X_test, y_test = prepare_data(train_df, test_df)

# Training
train_logreg(X_train, y_train, X_test, y_test, BASIC_MODEL_PATH)

# Evaluation
evaluate(BASIC_MODEL_PATH, X_test, y_test)