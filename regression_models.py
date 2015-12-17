import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
import numpy as np
from numpy import inf
import pandas as pd 
from nltk.corpus import stopwords
import cPickle as pickle
from votes_df_clean import get_precent_party, get_bill_id, get_votes_data, group_by_chamber_latest
from bills_df_json_clean import to_df, get_party_dict, get_sponsor_party, get_new_attributes

def join_dfs(votes_df, bills_df, bills_json_df):
	v = votes_df.iloc[:,np.where(votes_df.columns.values == 'num_yes')[0][0]:]
	y = v.pop('percent_yes_D')
	y.fillna(0, inplace = True)
	y = y.values
	v = v.iloc[:,np.where(v.columns.values == 'is_amendment')[0][0]:] #grab appropriate features
	v.pop('percent_yes_D')
	vbills = v.join(bills_json_df, how = 'left')
	infile = open('bills_tfidf_sparse.pkl', 'rb')
	bill_sparse = pickle.load(infile)
	bill_dense_tfidf = pd.DataFrame(bill_sparse.todense())
	bill_dense_tfidf.set_index(bills_df.index, inplace = True)
	vb = vbills.join(bill_dense_tfidf, how = 'left')
	vb = vb.apply(lambda x: (x-np.mean(x)) / np.std(x)) #normalize data
	vb.fillna(-1, inplace = True)
	X = vb.values.astype(np.float64)
	return X, y

def rf_regression_model(X, y):
	print 'running rf_regression_model...'
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	rfr = RandomForestRegressor(n_estimators = 5000, n_jobs = -1, oob_score = True)
	rfr.fit(X_train, y_train)
	pred = rfr.predict(X_test)
	mse = mean_squared_error(y_test, pred)
	rmse = np.sqrt(mse)
	r2 = r2_score(y_test, pred)
	return rfr, mse, rmse, r2

def bagging_regression_model(X, y):
	print 'running bagging_regression_model...'
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	bagr = BaggingRegressor(n_estimators = 5000, n_jobs = -1, oob_score = True, bootstrap_features = True)
	bagr.fit(X_train, y_train)
	pred = bagr.predict(X_test)
	mse = mean_squared_error(y_test, pred)
	rmse = np.sqrt(mse)
	r2 = r2_score(y_test, pred)
	return bagr, mse, rmse, r2

def linear_regression_model(X, y):
	print 'running linear_regression_model...'
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	lr = LinearRegression()
	lr.fit(X_train, y_train)
	pred = lr.predict(X_test)
	mse = mean_squared_error(np.exp(y_test), pred)
	rmse = np.sqrt(mse)
	r2 = r2_score(y_test, pred)
	return lr, mse, rmse, r2

def GB_regression_model_search(X, y):
	print 'running GB_regression_model_search...'
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	param_grid = [{'learning_rate': [.01, .05], 'n_estimators': [1000], 'max_depth': [5, 10, 15]}]
	GB = GradientBoostingRegressor()
	GBr = GridSearchCV(GB, param_grid, verbose = 2, cv = 3, n_jobs = -1) #3 k-folds
	GBr.fit(X_train, y_train)
	pred = GBr.predict(X_test)
	mse = mean_squared_error(y_test, pred)
	rmse = np.sqrt(mse)
	r2 = r2_score(y_test, pred)
	return GBr, mse, rmse, r2

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
	rfr, rfr_mse, rfr_rmse, rfr_r2 = rf_regression_model(X, y)
	bagr, bagr_mse, bagr_rmse, bagr_r2 = bagging_regression_model(X, y)
	lr, lr_mse, lr_rmse, lr_r2 = linear_regression_model(X, y)
	GBr, GBr_mse, GBr_rmse, GBr_r2 = GB_regression_model_search(X, y)
	