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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# 1 数据导入
file_path = 'input/train.csv'
data = pd.read_csv(file_path)
data_cate = data.select_dtypes(['object'])
data_without_cate = data.select_dtypes(exclude=['object'])
one_hot_data_cate = pd.get_dummies(data_cate)
data_dummy = pd.concat([data_without_cate, one_hot_data_cate], axis=1)

# 2 选取变量
y = data_dummy.SalePrice
X = data_dummy.drop(['SalePrice'], axis=1)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# 3 建模讯模
my_pipeline = make_pipeline(Imputer(), XGBRegressor(n_estimators=1000, learning_rate=0.05))
my_pipeline.fit(train_X, train_y)

# 4 误差验证
val_preds = my_pipeline.predict(val_X)
msel = mean_squared_error(np.log(val_preds), np.log(val_y))
print("RMSE: %2f" %sqrt(msel))

# 5 全数据训练模型
pipeline_on_full_data = make_pipeline(Imputer(), XGBRegressor(n_estimators=1000, learning_rate=0.05))
pipeline_on_full_data.fit(X, y, early_stopping_rounds=5, verbose=False)

# 6 模型预测结果组
test_data_path = 'input/test.csv'
test_data = pd.read_csv(test_data_path)
one_hot_test_data = pd.get_dummies(test_data)
fianl_train, final_test = X.align(one_hot_test_data, join='left', axis=1)
test_preds = pipeline_on_full_data.predict(final_test)

# 7 预测结果导出
output = pd.DataFrame({
	'Id': test_data.Id,
	'SalePrice': test_preds
	})
output.set_index('Id').to_csv('submission.csv')

# 8 预测结果提交
