from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd 
from votes_df_clean import get_precent_party, get_bill_id, get_votes_data, group_by_chamber_latest

b = bills_json_df.iloc[:,np.where(bills_json_df.columns.values == 'num_cosponsors')[0][0]:]


def rf_regression_model_grid_search(votes_df, bills_df):
	v = votes_df.iloc[:,np.where(votes_df.columns.values == 'num_yes')[0][0]:]
	y = v.pop('percent_yes_D')
	y.fillna(-1, inplace = True)
	y = y.values
	v = v.iloc[:,np.where(v.columns.values == 'is_amendment')[0][0]:]
	v.pop('percent_yes_D')
	mx_feats = [10000, 50000, 100000]
	params_list = []
	for i in mx_feats:
		t = TfidfVectorizer(max_features = 1000)
		bill_sparse = t.fit_transform(bills_df[0])
		bill_dense_tfidf = pd.DataFrame(bill_sparse.todense())
		bill_dense_tfidf.set_index(bills_df.index, inplace = True)
		vb = v.join(bill_dense_tfidf, how = 'left')
		vb = vb.apply(lambda x: (x-np.mean(x)) / np.std(x))
		vb.fillna(-1, inplace = True)
		X = vb.values.astype(np.float64)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
		rfr = RandomForestRegressor(n_estimators = 10000, n_jobs = -1, oob_score = True)
		score = cross_val_score(rfr, X, y, scoring = 'mean_squared_error').mean()
		params_list.append((score,i))
		print 'MSE: {}, RMSE: {}, max_features: {}'.format(score, np.sqrt(score), i)
	best_params = min(params_list, key = lambda x: x[0])
	return best_params

if __name__ == '__main__':
	votes_df = get_bill_id()
	votes_df = get_votes_data(votes_df)
	votes_df = group_by_chamber_latest(votes_df)
	votes_df.set_index('bill_id', inplace=True)
	bills_df = pd.read_pickle('bills_df')
	best_params = rf_regression_model_grid_search(votes_df, bills_df)