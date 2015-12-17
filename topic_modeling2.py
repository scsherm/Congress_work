from __future__ import print_function
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import cPickle as pickle
import pandas as pd 
from nltk.corpus import stopwords
import Stemmer
import nltk.stem 
import pattern.en as en
import unicodedata


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (en.lemma(word) for word in analyzer(doc) if str.isdigit(unicodedata.normalize('NFKD', word).encode('ascii','ignore')) == False)

class StemmedTfVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        return lambda doc: (en.lemma(word) for word in analyzer(doc) if str.isdigit(unicodedata.normalize('NFKD', word).encode('ascii','ignore')) == False)

def print_top_words(model, feature_names, n_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict[topic_idx] = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    return topic_dict

def fit_nmf(tfidf):
    nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5)
    nmf.fit(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)
    return nmf

def fit_lda(tf):
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5, learning_method='online', learning_offset=50., random_state=0)
    lda.fit(tf)
    return lda


if __name__ == '__main__':
    congressional_stop_words = open('congressional_stop_words.txt').read().split('\n')
    stop_words = stopwords.words('english')
    stop_words = stop_words + congressional_stop_words
    n_topics = 10
    n_top_words = 20
    with open('bills_tfidf_sparse.pkl', 'rb') as infile:
        tfidf = pickle.load(infile)
    with open('tfidf_vectorizer.pkl', 'rb') as infile:
        tfidf_vectorizer = pickle.load(infile)
    with open('bills_tf_sparse.pkl', 'rb') as infile:
        tf = pickle.load(infile)
    with open('tf_vectorizer.pkl', 'rb') as infile:
        tf_vectorizer = pickle.load(infile)
    nmf = fit_nmf(tfidf)
    lda = fit_lda(tf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    nmf_topic_dict = print_top_words(nmf, tfidf_feature_names, n_top_words)
    tf_feature_names = tf_vectorizer.get_feature_names()
    lda_topic_dict = print_top_words(lda, tf_feature_names, n_top_words)

    bills_df = pd.read_pickle('bills_df')
    t = StemmedTfidfVectorizer(max_features = 1000, stop_words = stop_words, max_df = 0.7)
    bill_sparse = t.fit_transform(bills_df[0])




