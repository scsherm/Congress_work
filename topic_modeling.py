import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from votes_df_clean import get_precent_party, get_bill_id, get_votes_data, group_by_chamber_latest
from bills_df_json_clean import to_df, get_party_dict, get_sponsor_party, get_new_attributes
import re
import yaml
from pymongo import MongoClient
import pyspark as ps
from pyspark.mllib.clustering import LDAModel
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext
from pyspark.sql import HiveContext
from pyspark.mllib.util import MLUtils
from gensim.models.ldamodel import LdaModel
from gensim import corpora
from gensim import matutils
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
import pandas as pd
from gensim import corpora, models, similarities

def join_dfs(bills_json_df, bills_df):
	bills_json_df.set_index('bill_id', inplace = True)
	bills = bills_df.join(bills_json_df.congress, how = 'left')
	return bills
sc = SparkContext()
hive_contxt = HiveContext(sc)
df = hive_contxt.createDataFrame(bills)

docs = bills[0]
congressional_stop_words = open('congressional_stop_words.txt').read().split('\n')
stop_words = stopwords.words('english')
stop_words = stop_words + congressional_stop_words

def tokenize_and_normalize(chunks):
    #words = [ tokenize.word_tokenize(sent.encode("utf8")) for sent in tokenize.sent_tokenize(chunks) ]
    words  = []
    try:
    	for sent in tokenize.sent_tokenize(chunks):
    		words.append(tokenize.word_tokenize(sent))
    except:
    	pass

    flatten = [ inner for sublist in words for inner in sublist ]
    stripped = [] 

    for word in flatten: 
        if word not in stop_words:
            try:
                stripped.append(word.encode('latin-1').decode('utf8').lower())
            except:
                #print "Cannot encode: " + word
                pass
            
    return [ word for word in stripped if len(word) > 1 ] 

# def print_features(clf, vocab, n=10):
#     """ Print sorted list of non-zero features/weights. """
#     coef = clf.coef_[0]
#     print 'positive features: %s' % (' '.join(['%s/%.2f' % (vocab[j], coef[j]) for j in np.argsort(coef)[::-1][:n] if coef[j] > 0]))
#     print 'negative features: %s' % (' '.join(['%s/%.2f' % (vocab[j], coef[j]) for j in np.argsort(coef)[:n] if coef[j] < 0]))

# lda = fit_lda(X,vocab)

# lda.show_topics(num_topics=10,num_words=50,formatted=False)

parsed = [ tokenize_and_normalize(s) for s in docs ]

dictionary = corpora.Dictionary(parsed)
corpus = [dictionary.doc2bow(text) for text in parsed]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

%time lda=LdaModel(corpus_tfidf, id2word=dictionary, num_topics=15, update_every=0, passes=200)
lda.print_topics(15, 15)

if __name__ == '__main__':
	bills_df = pd.read_pickle('bills_df')
	bills_json_df = to_df()