import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold

data = pd.read_csv('../data/kc_house_data_clean.csv')
