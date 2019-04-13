# Kaggle Competition 1 -- Housing Prices Document
# Version 8.0 -- Data Preprocess

# 0 导入常用库
import pandas as pd
import numpy as np
from math import sqrt
from math import log

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import Imputer
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# 1 读取数据 + 数据处理
train_data = pd.read_csv('input/train.csv')
test_data = pd.read_csv('input/test.csv')

train_data_cate = train_data.select_dtypes(['object'])
train_data_without_cate = train_data.select_dtypes(exclude=['object'])
one_hot_train_data_cate = pd.get_dummies(train_data_cate)
train_data_dummy = pd.concat([train_data_without_cate, one_hot_train_data_cate], axis=1)

test_data_cate = test_data.select_dtypes(['object'])
test_data_without_cate = test_data.select_dtypes(exclude=['object'])
one_hot_test_data_cate = pd.get_dummies(test_data_cate)
test_data_dummy = pd.concat([test_data_without_cate, one_hot_test_data_cate], axis=1)

final_train, final_test = train_data_dummy.align(test_data_dummy, join='left', axis=1)

# 2 选取变量
y = final_train.SalePrice
X = final_train.drop(['SalePrice'], axis=1)

# 3 变量分组
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


my_imputer = Imputer()
imputed_train_X = my_imputer.fit_transform(train_X)
imputed_val_X = my_imputer.transform(val_X)
final_test_X = final_test.drop(['SalePrice'], axis=1)
imputed_test_X = my_imputer.transform(final_test_X)

# 4 建模 + 5 训模

xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, early_stopping_rounds=5, verbose=False, eval_set=[(val_X, val_y)])
xgb_model.fit(imputed_train_X, train_y)
val_preds = xgb_model.predict(imputed_val_X)
xgb_msel = mean_squared_error(np.log(val_preds), np.log(val_y))

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(imputed_train_X, train_y)
val_preds = forest_model.predict(imputed_val_X)
forest_msel = mean_squared_error(np.log(val_preds), np.log(val_y))

tree_model = DecisionTreeRegressor(random_state=1)
tree_model.fit(imputed_train_X, train_y)
val_preds = tree_model.predict(imputed_val_X)
tree_msel = mean_squared_error(np.log(val_preds), np.log(val_y))


print("RMSE of xgb_model: %2f" %sqrt(xgb_msel))
print("RMSE of forest_model: %2f" %sqrt(forest_msel))
print("RMSE of tree_model: %2f" %sqrt(tree_msel))

# 8 全数据训练模型
#my_model_on_full_data = XGBRegressor(n_estimators=1000, learning_rate=0.05)
#my_model_on_full_data.fit(X, y, early_stopping_rounds=5, verbose=False)

# 9 模型预测结果组
test_preds = my_model.predict(imputed_test_X)


# 10 预测结果导出
output = pd.DataFrame({
	'Id': test_data.Id,
	'SalePrice': test_preds
	})
output.set_index('Id').to_csv('submission.csv')

# 11 预测结果提交







"""
one_hot_test_data = pd.get_dummies(test_data)
fianl_train, final_test = X.align(one_hot_test_data, join='left', axis=1)



my_imputer = Imputer()
train_data = my_imputer.fit_transform(train_data)
test_data = my_imputer.transform(test_data)


my_imputer = Imputer()
imputer_train = my_imputer.fit_transform(final_train)
imputer_test = my_imputer.transform(final_test)

"""