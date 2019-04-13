# Kaggle Competition 1 -- Housing Prices Document
# Version 2.0 -- XGBRegressor

# 0 导入常用库
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# 1 读取数据
file_path = 'input/*.csv'
data = pd.read_csv(file_path)

# 2 选取变量
y = data.Price
useful_features = []
X = data[useful_features]

# 3 变量分组
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# 4 建模
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

# 5 训练模型
my_model.fit(train_X, train_y, early_stopping_rounds=5, verbose=False)

# 6 模型预测
val_preds = my_model.predict(val_X)

# 7 误差验证
mae = mean_absolute_error(val_preds, val_y)
print("Validation MAE: %2f" %mae)

# 8 全数据训练模型
my_model_on_full_data = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model_on_full_data.fit(X, y, early_stopping_rounds=5, verbose=False)

# 9 模型预测结果组
test_data_path = '*'
test_data = pd.read_csv(test_data_path)
test_X = test_data[useful_features]
test_preds = my_model_on_full_data.predict(test_X)

# 10 预测结果导出
output = pd.DataFrame({
	'Id': test_data.Id,
	'SalePrice': test_preds
	})

# 11 预测结果提交