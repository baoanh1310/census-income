from joblib import load
import pandas as pd
import numpy as np

from config import BASIC_MODEL_PATH, DATA_PATH
from utils import read_data, prepare_data, predict

df = read_data(DATA_PATH)
X_train, y_train, X_test, y_test = prepare_data(df)

predict(BASIC_MODEL_PATH, X_test, y_test, 10)