import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyspark as ps
from pyspark import SparkContext
from pyspark.sql import HiveContext
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.feature import StandardScaler
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.tree import GradientBoostedTrees
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel 
from pyspark.mllib.evaluation import RegressionMetrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from nltk.corpus import stopwords
from numpy import inf


def join_dfs(votes_df, bills_df, bills_json_df):
	v = votes_df.iloc[:,np.where(votes_df.columns.values == 'num_yes')[0][0]:]
	y = v.pop('percent_yes_D')
	y.percent_yes_D.fillna(0, inplace = True)
	v = v.iloc[:,np.where(v.columns.values == 'is_amendment')[0][0]:] #grab appropriate features
	v.pop('percent_yes_D')
	vbills = v.join(bills_json_df, how = 'left')
	t = TfidfVectorizer(max_features = 1000, stop_words = stop_words)
	bill_sparse = t.fit_transform(bills_df[0])
	bill_dense_tfidf = pd.DataFrame(bill_sparse.todense())
	bill_dense_tfidf.set_index(bills_df.index, inplace = True)
	vb = vbills.join(bill_dense_tfidf, how = 'left')
	vb = vb.apply(lambda x: (x-np.mean(x)) / np.std(x)) #normalize data
	vb.fillna(-1, inplace = True)
	hive_contxt.creatDataFrame()
	X = vb.astype(np.float64)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	return X_train, X_test, y_train, y_test # As dataframes

X_train['label'] = y_train
X_train = pd.read_pickle('XandYtrainforspark')
X_test = pd.read_pickle('Xtestforspark')
y_test = pd.read_pickle('ytestforspark')

sc = SparkContext()
hive_contxt = HiveContext(sc)
df = hive_contxt.createDataFrame(X_train)
features = df.map(lambda row: row[:1079])
standardizer = StandardScaler()
model = standardizer.fit(features)
features_transform = model.transform(features)
lab = df.map(lambda row: row[1079])
transformedData = lab.zip(features_transform)
transformedData = transformedData.map(lambda row: LabeledPoint(row[0],[row[1]]))
trainingData, testingData = transformedData.randomSplit([.8,.2],seed=1234)
model = RandomForest.trainRegressor(trainingData, {}, numTrees = 6, seed=42)
t = testingData.map(lambda lp: lp.features)
t_labels = testingData.map(lambda lp: lp.label)
pred = model.predict(t)
pred = pred.collect()

t_labels = t_labels.collect()

mean_squared_error(t_labels,pred)
r2_score(t_labels,pred)
PredandLabel = pred.zip(t_labels)

model = GradientBoostedTrees.trainRegressor(trainingData, {}, numIterations=10, learningRate = 0.01)
t = testingData.map(lambda lp: lp.features)
t_labels = testingData.map(lambda lp: lp.label)
pred = model.predict(t)
pred = pred.collect()

t_labels = t_labels.collect()

mean_squared_error(t_labels,pred)
r2_score(t_labels,pred)
PredandLabel = pred.zip(t_labels)





# df_train = hive_contxt.createDataFrame(transformedData, ['label', 'features'])
# #rf = RandomForestRegressor(numTrees=100, maxDepth=30, seed=42)
# lr = LinearRegression()
# model = lr.fit(df_train)
# df_test = hive_contxt.createDataFrame(X_test)
# model = standardizer.fit(df_test)
# dftest_transform = model.transform(df_test)
# test0 = hive_contxt.createDataFrame(dftest_transform, ['features'])
# model.transform(test0).head().prediction

if __name__ == '__main__':
	congressional_stop_words = open('congressional_stop_words.txt').read().split('\n')
	stop_words = stopwords.words('english')
	stop_words = stop_words + congressional_stop_words
	bills_json_df = pd.read_pickle('bills_json_df')
	votes_df = get_bill_id()
	votes_df = get_votes_data(votes_df)
	votes_df = group_by_chamber_latest(votes_df)
	votes_df.set_index('bill_id', inplace = True)
	bills_df = pd.read_pickle('bills_df')
	X, y = join_dfs(votes_df, bills_df, bills_json_df)