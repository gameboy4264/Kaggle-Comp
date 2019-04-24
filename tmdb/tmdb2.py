# Kaggle Competition 1 -- Housing Prices Document
# Version 7.0 -- RMSE of Log

# 0 导入常用库
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error
from math import sqrt
from math import log

from sklearn.metrics import mean_squared_log_error




train_X, val_X, train_y, val_y = train_test_split(xtrain, ytrain, random_state=1)
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y)
val_preds = my_model.predict(val_X)

val_preds = my_model.predict(val_X)
val_preds[val_preds<0] = 0
rmsle = np.sqrt(mean_squared_log_error( val_preds, val_y))
print("RMSLE: %2f" %sqrt(rmsle))
