from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd 


def rf_regression_model_grid_search(votes_df, bills_df):
	v = votes_df.iloc[:,np.where(votes_df.columns.values == 'num_yes')[0][0]:]
	y = v.pop('percent_yes_D')
	y.fillna(-1, inplace = True)
	y = y.values
	v = v.iloc[:,np.where(v.columns.values == 'is_amendment')[0][0]:]
	mx_feats = [10000, 50000, 100000]
	n_estimators = [1000, 5000, 10000]
	params_list = []
	for i in mx_feats:
		for j in n_estimators:
			t = TfidfVectorizer(max_features = i)
			bill_sparse = t.fit_transform(bills_df[0])
			bill_dense_tfidf = pd.DataFrame(bill_sparse.todense())
			bill_dense_tfidf.set_index(bills_df.index, inplace = True)
			vb = v.join(bill_dense_tfidf, how = 'left')
			vb.fillna(-1, inplace = True)
			X = vb.values.astype(np.float64)
			rfr = RandomForestRegressor(n_estimators = j)
			score = cross_val_score(rfr, X, y, scoring = 'mean_squared_error').mean()
			params_list.append((score,i,j))
			print 'MSE: {}, RMSE: {}, max_features: {}, n_estimators: {}'.format(score, np.sqrt(score), i, j)
	best_params = min(params_list, key = lambda x: x[0])
	return best_params

if __name__ == '__main__':
	best_params = rf_regression_model_grid_search(votes_df, bills_df)