# Kaggle Competition 1 -- Housing Prices Document
# Version 7.0 -- RMSE of Log

# 0 导入常用库
import pandas as pd
import numpy as np
from math import sqrt
from xgboost import XGBRegressor

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score


"""
1- Preprocessing and exploring
1.1- Imports
1.2- Types
1.3 - Missing Values
1.4 - Exploring
1.5 - Feature Engineering
1.6 - Prepare for models

2- Nested Cross Validation
3- Submission
"""



# 1 数据导入
file_path = 'input/train.csv'
data = pd.read_csv(file_path)
data_cate = data.select_dtypes(['object'])
data_without_cate = data.select_dtypes(exclude=['object'])
one_hot_data_cate = pd.get_dummies(data_cate)
data_dummy = pd.concat([data_without_cate, one_hot_data_cate], axis=1)

# 2 选取变量
y = data_dummy.Survived
X = data_dummy.drop(['Survived'], axis=1)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# 3 建模讯模
my_pipeline = make_pipeline(Imputer(), XGBRegressor(n_estimators=1000, learning_rate=0.05))
my_pipeline.fit(train_X, train_y)
# 4 误差验证
val_preds = my_pipeline.predict(val_X)
val_pre = [int(item>0.5) for  item in val_preds]
score = accuracy_score(val_pre, val_y)
print("Accuracy: %2f" %score)

# 5 全数据训练模型
pipeline_on_full_data = make_pipeline(Imputer(), XGBRegressor(n_estimators=1000, learning_rate=0.05))
pipeline_on_full_data.fit(X, y, early_stopping_rounds=5, verbose=False)

# 6 模型预测结果组
test_data_path = 'input/test.csv'
test_data = pd.read_csv(test_data_path)
one_hot_test_data = pd.get_dummies(test_data)
fianl_train, final_test = X.align(one_hot_test_data, join='left', axis=1)
test_preds = my_pipeline.predict(final_test)

test_pre = [int(item>0.5) for  item in test_preds]

# 7 预测结果导出
output = pd.DataFrame({
	'PassengerId': test_data.PassengerId,
	'Survived': test_pre
	})
output.to_csv('submission.csv')

# 8 预测结果提交
