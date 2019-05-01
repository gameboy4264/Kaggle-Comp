# Kaggle Competition 1 -- Housing Prices Document
# Version 4.0 -- Pipeline

# 0 导入常用库
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error


# 1 读取数据
file_path = 'input/*.csv'
data = pd.read_csv(file_path)

# 2 选取变量
y = data.Price
useful_features = []
X = data[useful_features]

# 3 变量分组
# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# 4 建模
my_pipeline = make_pipeline(Imputer(), XGBRegressor(n_estimators=1000, learning_rate=0.05))

# 5 训练模型
my_pipeline.fit(X, y)

# 6 模型预测
# val_preds = my_pipeline.predict(val_X)

# 7 误差验证
# mae = mean_absolute_error(val_preds, val_y)
scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
mean_score = -1 * scores.mean()
print("Mean Absolute Error: %2f" %mean_score)

# 8 全数据训练模型
#my_model_on_full_data = XGBRegressor(n_estimators=1000, learning_rate=0.05)
#my_model_on_full_data.fit(X, y, early_stopping_rounds=5, verbose=False)

# 9 模型预测结果组
test_data_path = '*'
test_data = pd.read_csv(test_data_path)
test_X = test_data[useful_features]
test_preds = my_pipeline.predict(test_X)

# 10 预测结果导出
output = pd.DataFrame({
	'Id': test_data.Id,
	'SalePrice': test_preds
	})

# 11 预测结果提交