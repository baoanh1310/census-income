from joblib import load
import pandas as pd
import numpy as np

from config import BASIC_MODEL_PATH, DATA_PATH
from utils import read_data, prepare_data, predict

df = read_data('./data/test.csv')
df[df=='?'] = np.nan
print(df.info())