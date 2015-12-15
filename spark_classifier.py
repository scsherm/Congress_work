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
	v = votes_df.iloc[:,np.where(votes_df.columns.values == 'is_amendment')[0][0]:]
	v['vote'] = [1 for i in range(len(v))]
	t = TfidfVectorizer(max_features = 1000, stop_words = stop_words)
	bill_sparse = t.fit_transform(bills_df[0])
	bill_dense_tfidf = pd.DataFrame(bill_sparse.todense())
	bill_dense_tfidf.set_index(bills_df.index, inplace = True)
	all_bills = bill_dense_tfidf.join(bills_json_df, how = 'left')
	all_bills = all_bills.join(v.vote, how = 'left')
	all_bills.vote.fillna(0, inplace = True)
	y = all_bills.pop('vote')
	y.fillna(0, inplace = True)
	all_bills.fillna(-1, inplace = True)
	all_bills.pop('vote')
	X = all_bills.astype(np.float64)
	return X, y


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