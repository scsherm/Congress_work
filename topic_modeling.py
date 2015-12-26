from __future__ import print_function
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.externals import joblib
import cPickle as pickle
import pandas as pd 
from nltk.corpus import stopwords
import pattern.en as en
import numpy as np
import unicodedata


class StemmedTfidfVectorizer(TfidfVectorizer):
    '''add lemmatization and ignore digits for TfidfVectorizer'''

    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (en.lemma(word) for word in analyzer(doc) if str.isdigit(unicodedata.normalize('NFKD', word).encode('ascii','ignore')) == False)


class StemmedTfVectorizer(CountVectorizer):
    '''add lemmatization and ignore digits for CountVectorizer'''

    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        return lambda doc: (en.lemma(word) for word in analyzer(doc) if str.isdigit(unicodedata.normalize('NFKD', word).encode('ascii','ignore')) == False)


def print_top_words(model, feature_names, n_top_words):
    '''print topics/indicies and store them in dictionary'''

    topic_dict = {} #empty dict
    #enumerate lda/nmf components
    for topic_idx, topic in enumerate(model.components_):
        topic_dict[topic_idx] = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        #print("Topic {}:".format(topic_idx))
        #print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    #print()
    return topic_dict


def fit_nmf(tfidf, k, tfidf_vectorizer, n_top_words = 10):
    '''takes in a tfidf sparse vector and finds the top topics'''
    nmf = NMF(n_components=k, random_state=1, alpha=.1, l1_ratio=.5)
    nmf.fit(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    nmf_topic_dict = print_top_words(nmf, tfidf_feature_names, n_top_words)
    return nmf, nmf_topic_dict


def fit_lda(tf, k, tf_vectorizer, n_top_words = 10):
    '''takes in a tf sparse vector and finds the top topics'''
    lda = LatentDirichletAllocation(n_topics=k, max_iter=5, learning_method='online', learning_offset=50., random_state=0)
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    lda_topic_dict = print_top_words(lda, tf_feature_names, n_top_words)
    return lda, lda_topic_dict


if __name__ == '__main__':
    congressional_stop_words = open('congressional_stop_words.txt').read().split('\n')
    stp_words = stopwords.words('english')
    stp_words = stp_words + congressional_stop_words
    n_topics = 10
    n_top_words = 10
    bills_df_congress = pd.read_pickle('bills_df_congress')
    bills_df_congress.drop(bills_df_congress.index[np.where(bills_df_congress.congress.isnull())[0][0]], inplace = True)
    congress_topic_dict_nmf = {}
    congress_topic_dict_lda = {}
    nmf_words_weights = {}
    lda_words_weights = {}
    for congress in bills_df_congress.congress.unique():
        print("running nmf/lda for congress {}".format(congress))
        b = bills_df_congress.query('congress == @congress')
        tfidf_vectorizer = StemmedTfidfVectorizer(max_features = 1000, stop_words = stp_words, max_df = 0.5)
        tf_vectorizer = StemmedTfVectorizer(max_features = 1000, stop_words = stp_words, max_df = 0.5)
        b_tfidf = tfidf_vectorizer.fit_transform(b[0])
        b_tf = tf_vectorizer.fit_transform(b[0])
        nmf, nmf_topic_dict = fit_nmf(b_tfidf, n_topics, tfidf_vectorizer)
        lda, lda_topic_dict = fit_lda(b_tf, n_topics, tf_vectorizer)
        congress_topic_dict_nmf[congress] = nmf_topic_dict
        congress_topic_dict_lda[congress] = lda_topic_dict
        nmf_words_weights[congress] = [zip(tfidf_vectorizer.get_feature_names(), i) for i in nmf.components_*1000]
        lda_words_weights[congress] = [zip(tf_vectorizer.get_feature_names(), i) for i in lda.components_]
    tfidf = joblib.load('bills_tfidf_sparse.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    tf = joblib.load('bills_tf_sparse.pkl')
    tf_vectorizer = joblib.load('tf_vectorizer.pkl')
    nmf, nmf_topic_dict = fit_nmf(tfidf, n_topics, tfidf_vectorizer)
    lda, lda_topic_dict = fit_lda(tf, n_topics, tf_vectorizer)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    tf_feature_names = tf_vectorizer.get_feature_names()






