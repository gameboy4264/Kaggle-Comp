#导入常用库
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
from math import sqrt

#数据导入
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
sample = pd.read_csv('input/sample_submission.csv')


train.dtypes.sort_values()
train.select_dtypes(include='int64').head()
train.select_dtypes(include='float64').head()
train.select_dtypes(include='object').head()
train.isnull().sum()[train.isnull().sum()>0]  #应把train和test组合起来看

#处理缺失值

train.belongs_to_collection = train.belongs_to_collection.fillna("unknow")
test.belongs_to_collection = test.belongs_to_collection.fillna("unknow")

train.genres = train.genres.fillna("unknow")
test.genres = test.genres.fillna("unknow")

train.homepage = train.homepage.fillna("unknow")
test.homepage = test.homepage.fillna("unknow")

train.overview = train.overview.fillna("unknow")
test.overview = test.overview.fillna("unknow")

train.poster_path = train.poster_path.fillna("unknow")
test.poster_path = test.poster_path.fillna("unknow")

train.production_companies = train.production_companies.fillna("unknow")
test.production_companies = test.production_companies.fillna("unknow")

train.production_countries = train.production_countries.fillna("unknow")
test.production_countries = test.production_countries.fillna("unknow")

train.runtime = train.runtime.fillna(train.runtime.mean())
test.runtime = test.runtime.fillna(train.runtime.mean())

train.spoken_languages = train.spoken_languages.fillna("unknow")
test.spoken_languages = test.spoken_languages.fillna("unknow")

train.tagline = train.tagline.fillna("unknow")
test.tagline = test.tagline.fillna("unknow")

train.Keywords = train.Keywords.fillna("unknow")
test.Keywords = test.Keywords.fillna("unknow")

train.cast = train.cast.fillna("unknow")
test.cast = test.cast.fillna("unknow")

train.crew = train.crew.fillna("unknow")
test.crew = test.crew.fillna("unknow")

train.title = train.title.fillna("unknow")
test.title = test.title.fillna("unknow")

train.status = train.status.fillna(train.status.mode())
test.status = test.status.fillna(train.status.mode())

train.release_date = train.release_date.fillna("unknow")
test.release_date = test.release_date.fillna("unknow")

test.release_date[test.release_date=="unknow"] = 0000-00-00

# Feature Engeering

train['genres2'] = train.genres.apply(lambda x: x.count('id'))
test['genres2'] = test.genres.apply(lambda x: x.count('id'))

train['cast2'] = train.cast.apply(lambda x: x.count('cast_id'))
test['cast2'] = test.cast.apply(lambda x: x.count('cast_id'))

train['Keywords2'] = train.Keywords.apply(lambda x: x.count('id'))
test['Keywords2'] = test.Keywords.apply(lambda x: x.count('id'))

train['title2'] = train.title.apply(lambda x: len(x))
test['title2'] = test.title.apply(lambda x: len(x))

train['tagline2'] = train.tagline.apply(lambda x: len(x))
test['tagline2'] = test.tagline.apply(lambda x: len(x))

train['spoken_languages2'] = train.spoken_languages.apply(lambda x: x.count('name'))
test['spoken_languages2'] = test.spoken_languages.apply(lambda x: x.count('name'))

train['production_countries2'] = train.production_countries.apply(lambda x: x.count('name'))
test['production_countries2'] = test.production_countries.apply(lambda x: x.count('name'))

train['crew2'] = train.crew.apply(lambda x: x.count('id'))
test['crew2'] = test.crew.apply(lambda x: x.count('id'))

train['overview2'] = train.overview.apply(lambda x: len(x))
test['overview2'] = test.overview.apply(lambda x: len(x))

train['original_title2'] = train.original_title.apply(lambda x: len(x))
test['original_title2'] = test.original_title.apply(lambda x: len(x))

train['homepage2'] = train.homepage.apply(lambda x: 0 if x=='unknow' else 1)
test['homepage2'] = test.homepage.apply(lambda x: 0 if x=='unknow' else 1)

train['belongs_to_collection2'] = train.belongs_to_collection.apply(lambda x: 0 if x=='unknow' else 1)
test['belongs_to_collection2'] = test.belongs_to_collection.apply(lambda x: 0 if x=='unknow' else 1)

train['production_companies2'] = train.production_companies.apply(lambda x: x.count('id'))
test['production_companies2'] = test.production_companies.apply(lambda x: x.count('id'))

train['release_date_year'] = train.release_date.apply(lambda x: pd.to_datetime(x).year)
train['release_date_month'] = train.release_date.apply(lambda x: pd.to_datetime(x).month)
train['release_date_day'] = train.release_date.apply(lambda x: pd.to_datetime(x).day)
train['release_date_weekday'] = train.release_date.apply(lambda x: pd.to_datetime(x).weekday())
test['release_date_year'] = test.release_date.apply(lambda x: pd.to_datetime(x).year)
test['release_date_month'] = test.release_date.apply(lambda x: pd.to_datetime(x).month)
test['release_date_day'] = test.release_date.apply(lambda x: pd.to_datetime(x).day)
test['release_date_weekday'] = test.release_date.apply(lambda x: pd.to_datetime(x).weekday())

#热力图探索
"""
fig = plt.figure(figsize=(18,14))
sns.heatmap(train.corr(), annot=True)
"""

#选取特征
train.drop(['id', 'belongs_to_collection', 'genres', 'homepage',
       'imdb_id', 'original_language', 'original_title', 'overview', 'poster_path', 'production_companies',
       'production_countries', 'release_date', 'spoken_languages' , 'tagline', 'title', 'Keywords', 'cast', 'crew'], axis=1, inplace=True)

test.drop(['id', 'belongs_to_collection', 'genres', 'homepage',
       'imdb_id', 'original_language', 'original_title', 'overview', 'poster_path', 'production_companies',
       'production_countries', 'release_date', 'spoken_languages' , 'tagline', 'title', 'Keywords', 'cast', 'crew'], axis=1, inplace=True)


tmdb = pd.concat([train, test], sort=False)
tmdb = pd.get_dummies(tmdb)
len_train = len(train)
len_test = len(test)
train = tmdb[:len_train]
test = tmdb[len_train:]

train.revenue = train.revenue.astype('int')

xtrain=train.drop("revenue", axis=1)
ytrain=train['revenue']
xtest=test.drop("revenue", axis=1)


#模型训练

#模型1-XGBRegressor
train_X, val_X, train_y, val_y = train_test_split(xtrain, ytrain, random_state=1)
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y)
val_preds = my_model.predict(val_X)

val_preds = my_model.predict(val_X)
val_preds[val_preds<0] = 0
rmsle = np.sqrt(mean_squared_log_error( val_preds, val_y))
print("RMSLE: %2f" %sqrt(rmsle))

#结果提交
test2 = test.copy()
pred = my_model.predict(xtest)
output = pd.DataFrame({'id': test2.id, 'revenue': pred})
output.to_csv('submission1.csv', index=False)

#模型2-交叉检验 + XGBRegressor
from sklearn.model_selection import cross_val_score
XGB = XGBRegressor(n_estimators=1000, learning_rate=0.05)
score_xgb = cross_val_score(XGB, xtrain, ytrain, scoring='neg_mean_squared_log_error', cv=5)
#ValueError: Mean Squared Logarithmic Error cannot be used when targets contain negative values.
np.mean(score_xgb)

model = XGB.fit(xtrain, ytrain)
pred = model.predict(xtest)
output = pd.DataFrame({'id': test2.id, 'revenue': pred})
output.to_csv('submission3.csv', index=False)




#模型3-各类简单回归模型交叉检验尝试
from sklearn import tree
model_tree = tree.DecisionTreeRegressor()
score_tree = cross_val_score(model_tree, xtrain, ytrain, scoring='neg_mean_squared_log_error', cv=5)
sqrt(-np.mean(score_tree))

from sklearn import linear_model
model_lr = linear_model.LinearRegression()
score_lr = cross_val_score(model_lr, xtrain, ytrain, scoring='neg_mean_squared_log_error', cv=5)
#ValueError: Mean Squared Logarithmic Error cannot be used when targets contain negative values.
sqrt(-np.mean(score_lr))

from sklearn import svm
model_svr = svm.SVR()
score_svr = cross_val_score(model_svr, xtrain, ytrain, scoring='neg_mean_squared_log_error', cv=5)
sqrt(-np.mean(score_svr))

from sklearn import neighbors
model_KN = neighbors.KNeighborsRegressor()
score_KN = cross_val_score(model_KN, xtrain, ytrain, scoring='neg_mean_squared_log_error', cv=5)
sqrt(-np.mean(score_KN))

from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)
score_RFR = cross_val_score(model_RandomForestRegressor, xtrain, ytrain, scoring='neg_mean_squared_log_error', cv=5)
sqrt(-np.mean(score_RFR))

from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)
score_Ada = cross_val_score(model_AdaBoostRegressor, xtrain, ytrain, scoring='neg_mean_squared_log_error', cv=5)
sqrt(-np.mean(score_Ada))

from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)
score_GBRT = cross_val_score(model_GradientBoostingRegressor, xtrain, ytrain, scoring='neg_mean_squared_log_error', cv=5)
#ValueError: Mean Squared Logarithmic Error cannot be used when targets contain negative values.
sqrt(-np.mean(score_GBRT))

from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor()
score_bagging = cross_val_score(model_BaggingRegressor, xtrain, ytrain, scoring='neg_mean_squared_log_error', cv=5)
sqrt(-np.mean(score_bagging))

from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()
score_ExtraTree = cross_val_score(model_ExtraTreeRegressor, xtrain, ytrain, scoring='neg_mean_squared_log_error', cv=5)
sqrt(-np.mean(score_ExtraTree))



#模型4-各类简单回归模型函数尝试

from sklearn import tree
model_tree = tree.DecisionTreeRegressor()

from sklearn import linear_model
model_lr = linear_model.LinearRegression()

from sklearn import svm
model_svr = svm.SVR()

from sklearn import neighbors
model_KN = neighbors.KNeighborsRegressor()

from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)

from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)

from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)

from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor()

from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()


def try_model(model):
	model_name = str(model)
	train_X, val_X, train_y, val_y = train_test_split(xtrain, ytrain, random_state=1)
	model.fit(train_X, train_y)
	val_preds = model.predict(val_X)
	val_preds[val_preds<0] = 0
	rmsle = np.sqrt(mean_squared_log_error( val_preds, val_y))
	print("%s - RMSLE: %2f" % (model_name, sqrt(rmsle)))

model_list = [model_tree, model_lr, model_svr, model_KN, model_RandomForestRegressor, 
			  model_AdaBoostRegressor, model_GradientBoostingRegressor, model_BaggingRegressor,
			  model_ExtraTreeRegressor]

for model in model_list:
	try_model(model)







"""
train_X, val_X, train_y, val_y = train_test_split(xtrain, ytrain, random_state=1)
model_tree.fit(train_X, train_y)
val_preds = model_tree.predict(val_X)
val_preds[val_preds<0] = 0
rmsle = np.sqrt(mean_squared_log_error( val_preds, val_y))
print("RMSLE: %2f" %sqrt(rmsle))

model_BaggingRegressor.fit(xtrain, ytrain)
pred = model_BaggingRegressor.predict(xtest)
output = pd.DataFrame({'id': test2.id, 'revenue': pred})
output.to_csv('submission5.csv', index=False)
"""








