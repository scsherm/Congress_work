import pandas as pd 
import numpy as np 
from topic_modeling import StemmedTfidfVectorizer, StemmedTfVectorizer, print_top_words, fit_nmf, fit_lda
from sklearn.externals import joblib
from wordcloud import WordCloud
from scipy import ndimage
import matplotlib.pyplot as plt
from itertools import repeat, permutations
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from scipy.spatial.distance import pdist
from progress.bar import Bar
from multiprocessing import Pool


def get_H_W(nmf, tfidf):
	'''Creates the H and W matricies from the nmf object and calculates
	the mean weight of topic for each congress number (i.e. the years)'''
	#create H matrix (docs x topics)
	H = nmf.components_
	#create W matrix (topics x features)
	W = nmf.transform(tfidf)

	bills_df_congress = pd.read_pickle('bills_df_congress')

	#create dataframes
	W = pd.DataFrame(W)

	#set index as bill_id
	W.set_index(bills_df_congress.index, inplace = True)

	#add congress
	W['congress'] = bills_df_congress.congress

	#drop nan value
	W.drop(W.index[np.where(W.isnull())[0][0]], inplace = True)

	#groupby congress
	mean_topic_con_nmf = W.groupby('congress').mean()
	return H, W, mean_topic_con_nmf


def plot_all_topics(mean_topic_con_nmf):
	#create list for xtick labels
	con = ['1993', '1995', '1997', '1999', '2001', '2003', '2005', '2007', '2009', '2011', '2013', '2015']

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
	wc = WordCloud(background_color = 'black', width = 1800, height = 800).generate(text)
	plt.imshow(wc)
	plt.axis('off')


def get_metric(W, k, tfidf):
	'''Calculates how pure a document is related to the topic it is labeled as'''

	W.pop('congress')
	labels = np.array([np.argmax(row) for row in W.values])
	rel_importance = np.array([row[np.argmax(row)] / row.sum() for row in W.values])
	rel_importance = np.nan_to_num(rel_importance)
	purity_metric_l = []
	for i in xrange(k):
		mean_purity = rel_importance[labels == i].mean()
		std_purity = rel_importance[labels == i].std()
		purity_metric_l.append(mean_purity/std_purity)
	return np.array(purity_metric_l)


def topic_purity_maximizer(tfidf, tfidf_vectorizer):
	ntopic_pur_list = []
	for k in xrange(40,60):
		n_topics = k
		nmf, nmf_topic_dict = fit_nmf(tfidf, n_topics, tfidf_vectorizer)
		H, W, mean_topic_con_nmf = get_H_W(nmf, tfidf)
		purity_metric_l = get_metric(W, k)
		ntopic_pur_list.append((purity_metric_l.mean(), k))
	ntopic_pur_list.sort(key = lambda x: x[0])
	return ntopic_pur_list


def doc_word_count(tfidf, reverse_lookup, word1, word2 = None):
	'''Returns either the number of documents associated with two words or one'''
	tfidf_arr = tfidf.toarray()
	if word2:
		return len(np.where((tfidf_arr[:,reverse_lookup[word1]] > 0) & (tfidf_arr[:,reverse_lookup[word2]] > 0))[0])
	else:
		return len(np.where(tfidf_arr[:,reverse_lookup[word1]] > 0)[0])


def coherence_score(tfidf, reverse_lookup, word1, word2, e = 0.01):
	'''Calculates the coherence score, given two words, tfidf and the associated feature names'''

	return np.log((doc_word_count(tfidf, reverse_lookup, word1, word2) + e)/doc_word_count(tfidf, reverse_lookup, word2))


def topic_coherence(tfidf, reverse_lookup, topic_words):
	'''Calculatesthe pairwise total coherence score for a given topic and its top words'''

	result = 0.0
	perm = permutations(topic_words, 2)
	while True:
		try:
			word1, word2 = perm.next()
			result += coherence_score(tfidf, reverse_lookup, word1, word2)
		except:
			pass


def run_topic_coherence(tfidf, reverse_lookup):
	average_coherence_k = []
	for k in xrange(10,41,5):
		print ('running for {} topics...'.format(k))
		score_list = []
		nmf, nmf_topic_dict = fit_nmf(tfidf, 10, tfidf_vectorizer)
		for topic_words in nmf_topic_dict.values():
			val = topic_coherence(tfidf, reverse_lookup, topic_words.split())
			score_list.append(val)
			print (val, topic_words)
		average_coherence_k.append((score_list.mean(), k))
	return average_coherence_k


def get_cos_dist_H(H, k):
	'''Returns the cosine similarity between topics of the H matrix'''
	perm = permutations(H, 2)
	d = []
	while True:
		try:
			arr1, arr2 = perm.next()
			d.append(pairwise_distances(arr1.reshape(1,-1), arr2.reshape(1,-1), metric='cosine', n_jobs = -1))
		except:
			return np.array(d)


# def run_cos_H(tfidf, tfidf_vectorizer):
# 	'''Returns the mean cosine similarity between topics of the H matrix'''

# 	avg_d_k = []
# 	bar = Bar('Processing', max = 50)
# 	for k in xrange(101,151):
# 		nmf, nmf_topic_dict = fit_nmf(tfidf, k, tfidf_vectorizer)
# 		H = nmf.components_
# 		d = pairwise_distances(H, metric='cosine', n_jobs = -1)
# 		idx = np.tril_indices(d.shape[0], k=-1)
# 		avg_d_k.append((d[idx].mean(), k))
# 		bar.next()
# 	bar.finish()
# 	return avg_d_k


def get_cos_dist_W(W, k, tfidf):
	'''Returns an array of the cosine similarities of each topic'''

	#Establishes the topic label for each row in the W matrix
	labels = np.array([np.argmax(row) for row in W])
	
	dist_l = []
	#progress tracking
	bar = Bar('Processing')
	for i in xrange(k):
		vec = pairwise_distances(tfidf[labels == i], metric='cosine')
		idx = np.tril_indices(vec.shape[0], k=-1)
		dist_l.append(vec[idx].mean())
		bar.next()
	bar.finish()
	return np.array(dist_l)


# def run_cos_W(tfidf, tfidf_vectorizer):
# 	'''Returns the average intertopic cosine similarity and the corresponding k value'''

# 	avg_d_k = []
# 	bar = Bar('Processing')
# 	for k in xrange(101,151):
# 		nmf, nmf_topic_dict = fit_nmf(tfidf, k, tfidf_vectorizer)
# 		W = nmf.transform(tfidf) #Docs x topics matrix
# 		d = get_cos_dist_W(W, k, tfidf)
# 		avg_d_k.append((d.mean(), k))
# 		bar.next()
# 	bar.finish()
# 	return avg_d_k


def run_cos_W(k):
	'''Returns the average intertopic cosine similarity and the corresponding k value'''

	#load tfidf and vectorizer
	tfidf = joblib.load('bills_tfidf_sparse.pkl')
	tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

	#run nmf
	nmf, nmf_topic_dict = fit_nmf(tfidf, k, tfidf_vectorizer)

	#get W matrix
	W = nmf.transform(tfidf) 

	#get the mean cosine similarities per topic
	d = get_cos_dist_W(W, k, tfidf)
	print d.mean(), k
	return (d.mean(), k) 


def parallel_run_cos_W(parameter_list):
	'''Runs the nmf function and computes the cosine similarity on every topic
	in the tfidf matrix in parallel  on 16 cores'''

	pool = Pool(processes = 32)
	results = pool.map(run_cos_W, parameter_list)
	return results


def run_cos_H(k):
	'''Returns the mean cosine similarity between topics of the H matrix'''
	
	tfidf = joblib.load('bills_tfidf_sparse.pkl')
	tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
	nmf, nmf_topic_dict = fit_nmf(tfidf, k, tfidf_vectorizer)
	H = nmf.components_
	d = pairwise_distances(H, metric='cosine')
	idx = np.tril_indices(d.shape[0], k=-1)
	return (d[idx].mean(), k)

def parallel_run_cos_H(parameter_list):
	'''Runs the nmf function and computes the cosine similarity between topics
	of the H matrix in parallel  on 8 cores'''

	pool = Pool(processes = 8)
	results = pool.map(run_cos_H, parameter_list)
	return results





if __name__ == '__main__':
    tfidf = joblib.load('bills_tfidf_sparse.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    tf = joblib.load('bills_tf_sparse.pkl')
    tf_vectorizer = joblib.load('tf_vectorizer.pkl')
    n_topics = 70
    ntopic_pur_list = topic_purity_maximizer(tfidf, tfidf_vectorizer)
    #tf = joblib.load('bills_tf_sparse.pkl')
    #tf_vectorizer = joblib.load('tf_vectorizer.pkl')
    nmf, nmf_topic_dict = fit_nmf(tfidf, n_topics, tfidf_vectorizer)
    #lda, lda_topic_dict = fit_lda(tf)
	H, W, mean_topic_con_nmf = get_H_W(nmf, tfidf)
	reverse_lookup = {word: idx for idx, word in enumerate(np.array(tfidf_vectorizer.get_feature_names()))}
	average_coherence_k = run_topic_coherence(tfidf, reverse_lookup)
	avg_d_k_H_164 = parallel_run_cos_H(range(101,165))
	avg_d_k_W_164 = parallel_run_cos_W(range(101,165))



