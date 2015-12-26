import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split, cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import recall_score, precision_score, average_precision_score, roc_curve, roc_auc_score
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adagrad, Adadelta, Adam
from keras.layers.advanced_activations import ThresholdedReLU
from keras.utils import np_utils
import numpy as np
import pandas as pd 
from over_sampling import SMOTE
from under_sampling import UnderSampler
from nltk.corpus import stopwords
import cPickle as pickle
from votes_df_clean import get_precent_party, get_bill_id, get_votes_data, group_by_chamber_latest
from bills_df_json_clean import to_df, get_party_dict, get_sponsor_party, get_new_attributes
# import theano
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'


def join_dfs(votes_df, bills_df, bills_json_df):
	'''joins the matrices to get back the bills data with the response as True or False
	for whether the bill was voted on'''

	#Get voted values
	v = votes_df.iloc[:,np.where(votes_df.columns.values == 'is_amendment')[0][0]:]
	v['vote'] = [1 for i in range(len(v))]

	#load tfidf
	bill_sparse = joblib.load('bills_tfidf_sparse.pkl')
	bill_dense_tfidf = pd.DataFrame(bill_sparse.todense())
	bill_dense_tfidf.set_index(bills_df.index, inplace = True)

	#join text with features
	all_bills = bill_dense_tfidf.join(bills_json_df, how = 'left')
	all_bills = all_bills.join(v.vote, how = 'left')
	all_bills.vote.fillna(0, inplace = True)

	#grab response variable
	y = all_bills.pop('vote')
	y.fillna(0, inplace = True)
	y = y.values
	all_bills.fillna(-1, inplace = True)
	all_bills.pop('vote')
	X = all_bills.values.astype(np.float64)
	return X, y


def clf_model(X, y, m_label, model = RandomForestClassifier(n_estimators = 5000, n_jobs = -1, oob_score = True)):
    '''runs a classifier model for the given model (with paramters)'''
    print 'running {}...'.format(model)

    #split 80/20 train test
    sss = StratifiedShuffleSplit(y, n_iter = 1, test_size = 0.2, random_state = 42)
    for train_index, test_index in sss:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    #Oversampling of unbalanced dataset
    sm = SMOTE(kind = 'regular', verbose = True)
    X_train, y_train = sm.fit_transform(X_train, y_train)
    X_train, y_train = sm.fit_transform(X_train, y_train)


    u = UnderSampler()
    X_train, y_train = u.fit_transform(X_train, y_train)

    #fit model
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
    #plt.plot(fpr, tpr, label = '{} AUC = {}'.format(m_label,round(AUC,3)))
    return clf, recall, AUC, precision, AUC2


def GB_classifier_model_search(X, y, m_label):
	'''runs grid search for the gradient boosting classifer'''
	#split 80/20 train test
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	#create param grid for search
	param_grid = [{'learning_rate': [.01, .001], 'n_estimators': [1000], 'max_depth': [3,5,7]}]
	GB = GradientBoostingClassifier()
	print 'running GradientBoostingClassifier with grid search...'
	GBc = GridSearchCV(GB, param_grid, verbose = 2, cv = 2, n_jobs = -1) #2 k-folds
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
	plt.plot(fpr, tpr, label = '{} AUC = {}'.format(m_label,round(AUC,3)))
	v = np.linspace(0,1)
	plt.plot(v,v, linestyle = '--', color = 'k')
	plt.xlabel("False Postive Rate")
	plt.ylabel("True Postive Rate")
	plt.title('ROC Curve')
	plt.xlim(-0.05,1)
	plt.ylim(0,1.05)
	plt.axhline(1, color = 'k', linestyle = '--')
	plt.axvline(0, color = 'k', linestyle = '--')
	plt.legend()
	return GBc, recall, AUC, precision, AUC2

def run_neural_net(X, y):
    print 'running neural network...'
    model = Sequential()
    
    #split 80/20 train test
    sss = StratifiedShuffleSplit(y, n_iter = 1, test_size = 0.2, random_state = 42)
    for train_index, test_index in sss:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    #Oversampling of unbalanced dataset
    sm = SMOTE(kind = 'regular', verbose = True)
    X_train, y_train = sm.fit_transform(X_train, y_train)
    X_train, y_train = sm.fit_transform(X_train, y_train)

    y_train = y_train.reshape(y_train.shape[0],1)
    y_test = y_test.reshape(y_test.shape[0],1)
    y_train, y_test = [np_utils.to_categorical(x) for x in (y_train, y_test)]
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    #tr = ThresholdedReLU(theta = 0.3)
    model.add(Dense(input_dim=X.shape[1], output_dim=1000, init='uniform',activation='relu'))
    #model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=1000, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=1000, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=2, init='uniform'))
    model.add(Activation('softmax'))
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(X_train, y_train, nb_epoch=10, batch_size = 200)
    score = model.evaluate(X_test, y_test, show_accuracy=True)
    pred = model.predict_proba(X_test) #get back probabilities
    pred2 = model.predict_classes(X_test) #get back predictions
    fpr, tpr, thresholds = roc_curve(y_test[:,1], pred[:,1])
    
    #get the AUC
    AUC = roc_auc_score(y_test[:,1], pred[:,1])
    
    #get the AUC for precision and recall curve
    AUC2 = average_precision_score(y_test[:,1], pred[:,1])
    recall = recall_score(y_test[:,1], pred2)
    precision = precision_score(y_test[:,1], pred2)
    print score
    return model, X_train, y_train, X_test, y_test, score


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
    rfc2, rfc_recall2, rfc_AUC2, rfc_precision2, rfc_AUC22 = clf_model(X, y, m_label = 'RFC', model = RandomForestClassifier(n_estimators = 5000, n_jobs = -1, oob_score = True))
    GNB2, GNB_recall2, GNB_AUC2, GNB_precision2, GNB_AUC22 = clf_model(X, y, m_label = 'GNB', model = GaussianNB())
    MNB2, MNB_recall2, MNB_AUC2, MNB_precision2, MNB_AUC22 = clf_model(X, y, m_label = 'MNB', model = MultinomialNB())
    BNB2, BNB_recall2, BNB_AUC2, BNB_precision2, BNB_AUC22 = clf_model(X, y, m_label = 'BNB', model = BernoulliNB())
    logitr2, logitr_recall2, logitr_AUC2, logitr_precision2, logitr_AUC22 = clf_model(X, y, m_label = 'Logr', model = LogisticRegression())
    model, X_train, y_train, X_test, y_test, score = run_neural_net(X, y)


