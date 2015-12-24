import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
import numpy as np
from numpy import inf
import pandas as pd 
from nltk.corpus import stopwords
import cPickle as pickle
from votes_df_clean import get_precent_party, get_bill_id, get_votes_data, group_by_chamber_latest
from bills_df_json_clean import to_df, get_party_dict, get_sponsor_party, get_new_attributes

def join_dfs(votes_df, bills_df, bills_json_df):
	'''Joins the data and return X, y numpy arrays for training'''

	v = votes_df.iloc[:,np.where(votes_df.columns.values == 'num_yes')[0][0]:]
	y = v.pop('percent_yes_R')
	y.fillna(0, inplace = True)
	y = y.values
	v = v.iloc[:,np.where(v.columns.values == 'is_amendment')[0][0]:] #grab appropriate features
	v.pop('percent_yes_R')
	vbills = v.join(bills_json_df, how = 'left')
	bill_sparse = joblib.load('bills_tfidf_sparse.pkl')
	bill_dense_tfidf = pd.DataFrame(bill_sparse.todense())
	bill_dense_tfidf.set_index(bills_df.index, inplace = True)
	vb = vbills.join(bill_dense_tfidf, how = 'left')
	vb = vb.apply(lambda x: (x-np.mean(x)) / np.std(x)) #normalize data
	vb.fillna(-1, inplace = True)
	X = vb.values.astype(np.float64)
	return X, y


def rf_regression_model(X, y):
	'''Runs random forest regressor model and calulates the mse, rmse, and r2 score'''

	print 'running rf_regression_model...'

	#Split train test 80/20
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	#Set random forest object and fit model
	rfr = RandomForestRegressor(n_estimators = 	10000, n_jobs = -1, oob_score = True)
	rfr.fit(X_train, y_train)

	#Calulate prediction
	pred = rfr.predict(X_test)

	#Calculate error
	mse = mean_squared_error(y_test, pred)
	rmse = np.sqrt(mse)
	r2 = r2_score(y_test, pred)
	return rfr, mse, rmse, r2


def bagging_regression_model(X, y):
	'''Runs bagging regressor model and calulates the mse, rmse, and r2 score'''

	print 'running bagging_regression_model...'

	#Split train test 80/20
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	#Set bagging object and fit model
	bagr = BaggingRegressor(n_estimators = 5000, n_jobs = -1, oob_score = True, bootstrap_features = True)
	bagr.fit(X_train, y_train)

	#Calulate prediction
	pred = bagr.predict(X_test)

	#Calculate error
	mse = mean_squared_error(y_test, pred)
	rmse = np.sqrt(mse)
	r2 = r2_score(y_test, pred)
	return bagr, mse, rmse, r2


def linear_regression_model(X, y):
	'''Runs linear regression model and calulates the mse, rmse, and r2 score'''

	print 'running linear_regression_model...'

	#Split train test 80/20
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	#Set linear regression object and fit model
	lr = LinearRegression()
	lr.fit(X_train, y_train)

	#Calulate prediction
	pred = lr.predict(X_test)

	#Calculate error
	mse = mean_squared_error(np.exp(y_test), pred)
	rmse = np.sqrt(mse)
	r2 = r2_score(y_test, pred)
	return lr, mse, rmse, r2


def GB_regression_model_search(X, y):
	'''Runs gradient boosted regressor model with grid search and calulates the mse, rmse, and r2 score'''

	print 'running GB_regression_model_search...'

	#Split train test 80/20
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	#Set params for grid search
	param_grid = [{'learning_rate': [.01, .001], 'n_estimators': [1000], 'max_depth': [3,5,7]}]

	#Set gradient boosted object
	GB = GradientBoostingRegressor()
	#Perform grid search accross all cores and fit model with best params
	GBr = GridSearchCV(GB, param_grid, verbose = 2, cv = 2, n_jobs = -1) #2 k-folds
	GBr.fit(X_train, y_train)

	#Calulate prediction
	pred = GBr.predict(X_test)

	#Calculate error
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
	rfr3, rfr_mse3, rfr_rmse3, rfr_r23 = rf_regression_model(X, y)
	bagr3, bagr_mse3, bagr_rmse3, bagr_r23 = bagging_regression_model(X, y)
	lr3, lr_mse3, lr_rmse3, lr_r23 = linear_regression_model(X, y)
	GBr3, GBr_mse3, GBr_rmse3, GBr_r23 = GB_regression_model_search(X, y)
	