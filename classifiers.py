import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd 
from nltk.corpus import stopwords
import cPickle as pickle
from votes_df_clean import get_precent_party, get_bill_id, get_votes_data, group_by_chamber_latest
from bills_df_json_clean import to_df, get_party_dict, get_sponsor_party, get_new_attributes

def join_dfs(votes_df, bills_df, bills_json_df):
	'''joins the matrices to get back the bills data with the response as True or False
	for whether the bill was voted on'''
	v = votes_df.iloc[:,np.where(votes_df.columns.values == 'is_amendment')[0][0]:]
	v['vote'] = [1 for i in range(len(v))]
	infile = open('bills_tfidf_sparse.pkl', 'rb')
	bill_sparse = pickle.load(infile)
	bill_dense_tfidf = pd.DataFrame(bill_sparse.todense())
	bill_dense_tfidf.set_index(bills_df.index, inplace = True)
	all_bills = bill_dense_tfidf.join(bills_json_df, how = 'left')
	all_bills = all_bills.join(v.vote, how = 'left')
	all_bills.vote.fillna(0, inplace = True)
	y = all_bills.pop('vote')
	y.fillna(0, inplace = True)
	y = y.values
	all_bills.fillna(-1, inplace = True)
	all_bills.pop('vote')
	X = all_bills.values.astype(np.float64)
	return X, y

def clf_model(X, y, model = RandomForestClassifier(n_estimators = 5000, n_jobs = -1, oob_score = True)):
	'''runs a classifier model for the given model (with paramters)'''
	print 'running {}...'.format(model)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	clf = model
	clf.fit(X_train, y_train)
	pred = clf.predict_proba(X_test) #get back probabilities
	pred2 = clf.predict(X_test) #get back predictions
	fpr, tpr, thresholds = roc_curve(y_test, pred[:,1])
	#get the AUC
	AUC = roc_auc_score(y_test, pred[:,1])
	#get the AUC for precision and recall curve
	AUC2 = average_precision_score(y_test, pred[:,1])
	recall = recall_score(y_test, pred2)
	precision = precision_score(y_test, pred2)
	#plot AUC
	plt.plot(fpr, tpr, label = 'AUC = {}'.format(round(AUC,4)))
	v = np.linspace(0,1)
	plt.plot(v,v, linestyle = '--', color = 'b')
	plt.xlabel("False Postive Rate")
	plt.ylabel("True Postive Rate")
	plt.title('ROC Curve')
	plt.xlim(-0.1,1)
	plt.ylim(0,1.1)
	plt.axhline(1, color = 'k', linestyle = '--')
	plt.axvline(0, color = 'k', linestyle = '--')
	return clf, recall, AUC, precision, AUC2

def GB_classifier_model_search(X, y):
	'''runs grid search for the gradient boosting classifer'''
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	param_grid = [{'learning_rate': [.01, .05], 'n_estimators': [1000], 'max_depth': [5, 10, 15]}]
	GB = GradientBoostingClassifier()
	print 'running GradientBoostingClassifier with grid search...'
	GBc = GridSearchCV(GB, param_grid, verbose = 2, cv = 3, n_jobs = -1) #10 k-folds
	GBc.fit(X_train, y_train)
	pred = GBc.predict_proba(X_test) #get back probabilities
	pred2 = GBc.predict(X_test) #get back predictions
	fpr, tpr, thresholds = roc_curve(y_test, pred[:,1])
	#get the AUC
	AUC = roc_auc_score(y_test, pred[:,1])
	#get the AUC for precision and recall curve
	AUC2 = average_precision_score(y_test, pred[:,1])
	recall = recall_score(y_test, pred2)
	precision = precision_score(y_test, pred2)
	#plot AUC
	plt.plot(fpr, tpr, label = 'AUC = {}'.format(round(AUC,4)))
	v = np.linspace(0,1)
	plt.plot(v,v, linestyle = '--', color = 'b')
	plt.xlabel("False Postive Rate")
	plt.ylabel("True Postive Rate")
	plt.title('ROC Curve')
	plt.xlim(-0.1,1)
	plt.ylim(0,1.1)
	plt.axhline(1, color = 'k', linestyle = '--')
	plt.axvline(0, color = 'k', linestyle = '--')
	plt.legend() 
	return GBc, recall, AUC, precision, AUC2


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
	rfc, rfc_recall, rfc_AUC, rfc_precision, rfc_AUC2 = clf_model(X, y, model = RandomForestClassifier(n_estimators = 5000, n_jobs = -1, oob_score = True))
	plt.savefig("RandomForestClassifier", format = 'png')
	GNB, GNB_recall, GNB_AUC, GNB_precision, GNB_AUC2 = clf_model(X, y, model = GaussianNB())
	plt.savefig("GaussianNB", format = 'png')
	MNB, MNB_recall, MNB_AUC, MNB_precision, MNB_AUC2 = clf_model(X, y, model = MultinomialNB())
	plt.savefig("MultinomialNB", format = 'png')
	BNB, BNB_recall, BNB_AUC, BNB_precision, BNB_AUC2 = clf_model(X, y, model = BernoulliNB())
	plt.savefig("BernoulliNB", format = 'png')
	logitr, logitr_recall, logitr_AUC, logitr_precision, logitr_AUC2 = clf_model(X, y, model = LogisticRegression())
	plt.savefig("LogisticRegression", format = 'png')
	GBc, GBc_recall, GBc_AUC, GBc_precision, GBc_AUC2 = GB_classifier_model_search(X, y)
	plt.savefig("GradientBoostingClassifier", format = 'png')

