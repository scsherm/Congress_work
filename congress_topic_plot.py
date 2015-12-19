import pandas as pd 
import numpy as np 
from topic_modeling import StemmedTfidfVectorizer, StemmedTfVectorizer, print_top_words, fit_nmf, fit_lda
from sklearn.externals import joblib
from wordcloud import WordCloud
from scipy import ndimage
import matplotlib.pyploy as plt
from itertools import repeat

def get_H_W(nmf, lda, tfidf, tf):
	#create H matrix (docs x topics)
	H = nmf.components_
	H2 = lda.components_
	#create W matrix (topics x features)
	W = nmf.transform(tfidf)
	W2 = lda.transform(tf)

	bills_df_congress = pd.read_pickle('bills_df_congress')

	#create dataframes
	W = pd.DataFrame(W)
	W2 = pd.DataFrame(W2)
	#set index as bill_id
	W.set_index(bills_df_congress.index, inplace = True)
	W2.set_index(bills_df_congress.index, inplace = True)
	#add congress
	W['congress'] = bills_df_congress.congress
	W2['congress'] = bills_df_congress.congress
	#drop nan value
	W.drop(W.index[np.where(W.isnull())[0][0]], inplace = True)
	W2.drop(W2.index[np.where(W2.isnull())[0][0]], inplace = True)

	mean_topic_con_nmf = W.groupby('congress').mean()
	mean_topic_con_lda = W2.groupby('congress').mean()
	return H, W, H2, W2, mean_topic_con_nmf, mean_topic_con_lda


def plot_all_topics(mean_topic_con_nmf):
	#create list for xtick labels
	con = ['103\n(1993-1994)', '104\n(1995-1996)', '105\n(1997-1998)', '106\n(1999-2000)', '107\n(2001-2002)', '108\n(2003-2004)', \
	'109\n(2005-2006)', '110\n(2007-2008)', '111\n(2009-2010)', '112\n(2011-2012)', '113\n(2013-2014)', '114\n(2015-2016)']

	#identify each topic with index and plt
	for idx in mean_topic_con_nmf:
		plt.plot(mean_topic_con_nmf[idx], label=idx)
		ax = plt.gca()
		ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
		ax.set_xticklabels(con)
	plt.legend()

def explore_topic(mean_topic_con_nmf, idx, tfidf_vectorizer, H):
	#get feature names form vectorizer
	feature_names = tfidf_vectorizer.get_feature_names()

	#plot topic over time
	plt.plot(mean_topic_con_nmf[idx])
	plt.title('Topic {}'.format(idx))
	plt.show()
	plt.close()

	#plot wordcloud
	topic_arr = H[idx]
	topic_arr_names = zip(feature_names, topic_arr*1000)
	words = map(lambda x: list(repeat(x[0], int(x[1]))), topic_arr_names)
	words = [word for sublist in words for word in sublist]
	text = ' '.join(words)
	wc = WordCloud(background_color = 'white', width = 800, height = 1800).generate(text)
	plt.imshow(wc)
	plt.axis('off')


if __name__ == '__main__':
	tfidf = joblib.load('bills_tfidf_sparse.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    tf = joblib.load('bills_tf_sparse.pkl')
    tf_vectorizer = joblib.load('tf_vectorizer.pkl')
    nmf, nmf_topic_dict = fit_nmf(tfidf)
    lda, lda_topic_dict = fit_lda(tf)
	H, W, H2, W2, mean_topic_con_nmf, mean_topic_con_lda = get_H_W(nmf, lda, tfidf, tf)
	plot_all_topics(mean_topic_con_nmf)
	plt.savefig('topics_over_congress_nmf.png', format = 'png')
	plt.close()
	plot_all_topics(mean_topic_con_lda)
	plt.savefig('topics_over_congress_lda.png', format = 'png')
	plt.close()
