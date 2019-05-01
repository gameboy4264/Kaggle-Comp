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


###   rms = sqrt(mean_squared_error(y_actual, y_predicted))  ###

# 1 读取数据 + 数据处理
file_path = 'input/train.csv'
data = pd.read_csv(file_path)
data_cate = data.select_dtypes(['object'])
data_without_cate = data.select_dtypes(exclude=['object'])
one_hot_data_cate = pd.get_dummies(data_cate)
data_dummy = pd.concat([data_without_cate, one_hot_data_cate], axis=1)

# 2 选取变量
y = data_dummy.SalePrice
X = data_dummy.drop(['SalePrice'], axis=1)

# 3 变量分组
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# 4 建模
my_pipeline = make_pipeline(Imputer(), XGBRegressor(n_estimators=1000, learning_rate=0.05))

# 5 训练模型
my_pipeline.fit(X, y)

# 6 模型预测
val_preds = my_pipeline.predict(val_X)

# 7 误差验证
msel = mean_squared_error(np.log(val_preds), np.log(val_y))
# scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_squared_error')
print("RMSE: %2f" %sqrt(msel))

# 8 全数据训练模型
#my_model_on_full_data = XGBRegressor(n_estimators=1000, learning_rate=0.05)
#my_model_on_full_data.fit(X, y, early_stopping_rounds=5, verbose=False)

# 9 模型预测结果组
test_data_path = 'input/test.csv'
test_data = pd.read_csv(test_data_path)
one_hot_test_data = pd.get_dummies(test_data)
fianl_train, final_test = X.align(one_hot_test_data, join='left', axis=1)
test_preds = my_pipeline.predict(final_test)

# 10 预测结果导出
output = pd.DataFrame({
	'Id': test_data.Id,
	'SalePrice': test_preds
	})
output.set_index('Id').to_csv('submission.csv')

# 11 预测结果提交
