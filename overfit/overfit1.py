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

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


# 1 数据导入
file_path = 'input/train.csv'
data = pd.read_csv(file_path)

# 2 选取变量
y = data.target
X = data.drop(['target'], axis=1)
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
test_preds = my_pipeline.predict(test_data)

test_pre = [int(item>0.5) for  item in test_preds]

# 7 预测结果导出
output = pd.DataFrame({
	'id': test_data.id,
	'target': test_pre
	})
output.set_index('id')
output.to_csv('submission.csv')

# 8 预测结果提交










'''


svc=make_pipeline(StandardScaler(),SVC(random_state=1))
r=[0.0001,0.001,0.1,1,10,50,100]
PSVM=[{'svc__C':r, 'svc__kernel':['linear']},
      {'svc__C':r, 'svc__gamma':r, 'svc__kernel':['rbf']}]
GSSVM=GridSearchCV(estimator=svc, param_grid=PSVM, scoring='accuracy', cv=2)
scores_svm=cross_val_score(GSSVM, X.astype(float), y,scoring='accuracy', cv=5)

np.mean(scores_svm)


warnings.filterwarnings(action="ignore")
model=GSSVM.fit(X, y)
pred=model.predict(xtest)
output=pd.DataFrame({'PassengerId':test2['PassengerId'],'Survived':pred})
output.to_csv('submission.csv', index=False)