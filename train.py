import os

from config import BASIC_MODEL_PATH, DATA_PATH
from utils import read_data, prepare_data, train_logreg, evaluate

# Import dataset
df = read_data(DATA_PATH)

# Prepare data
X_train, y_train, X_test, y_test = prepare_data(df)

# Training
train_logreg(X_train, y_train, X_test, y_test, BASIC_MODEL_PATH)

# Evaluation
evaluate(BASIC_MODEL_PATH, X_test, y_test)