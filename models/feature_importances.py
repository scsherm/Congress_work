import sys
sys.path.append('/Users/scsherm/Documents/Congress_work/Unbalanced_Data')
sys.path.append('/Users/scsherm/Documents/Congress_work/topic_modeling')
sys.path.append('/Users/scsherm/Documents/Congress_work/models')
sys.path.append('/Users/scsherm/Documents/Congress_work/data_cleaning')
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
import seaborn as sns
from votes_df_clean import get_precent_party, get_bill_id, get_votes_data, group_by_chamber_latest
from bills_df_json_clean import to_df, get_party_dict, get_sponsor_party, get_new_attributes

# Plot feature importance
feature_importance = GBr.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)[::-1]
sorted_idx = sorted_idx[:10]
pos = np.arange(sorted_idx.shape[0]) + .5
sns.barplot(pos, feature_importance[sorted_idx], orient='h')
cols_names = vb.iloc[:,sorted_idx].columns.values
for i in xrange(len(cols_names)):
	if type(cols_names[i]) == int:
		cols_names[i] = tfidf_vectorizer.get_feature_names()[cols_names[i]]
plt.yticks(pos, cols_names, rotation=45)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')

plt.show()
if __name__ == '__main__':