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


def get_H_W(nmf, tfidf, W):
	'''Creates the H and W matricies from the nmf object and calculates
	the mean weight of topic for each congress number (i.e. the years)'''

	#create H matrix (docs x topics)
	#H = nmf.components_
	#create W matrix (topics x features)
	#W = nmf.transform(tfidf)

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
	return mean_topic_con_nmf


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

	#create list for xtick labels
	con = ['1993', '1995', '1997', '1999', '2001', '2003', '2005', '2007', '2009', '2011', '2013', '2015']

	#plot topic over time
	plt.plot(mean_topic_con_nmf[idx])
	plt.title('Topic {}'.format(idx))
	ax = plt.gca()
	ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
	ax.set_xticklabels(con)
	plt.show()
	plt.close()

	#plot wordcloud
	topic_arr = H[idx]
	topic_arr_names = zip(feature_names, topic_arr*1000)
	words = map(lambda x: list(repeat(x[0], int(x[1]))), topic_arr_names)
	words = [word for sublist in words for word in sublist]
	text = ' '.join(words)
	wc = WordCloud(background_color = 'white', width = 1800, height = 800).generate(text)
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


def get_cos_dist_W(W, k, tfidf):
	'''Returns an array of the cosine similarities of each topic'''

	#Establishes the topic label for each row in the W matrix
	labels = np.array([np.argmax(row) for row in W])
	
	dist_l = []
	#progress tracking
	bar = Bar('Processing')
	for i in xrange(k):
		if len(W[labels == i]):
			vec = pairwise_distances(tfidf[labels == i], metric='cosine')
			idx = np.tril_indices(vec.shape[0], k=-1)
			dist_l.append(vec[idx].mean())
		bar.next()
	bar.finish()
	return np.array(dist_l)


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

	pool = Pool(processes = 36)
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
	print d[idx].mean(), k
	return (d[idx].mean(), k)


def parallel_run_cos_H(parameter_list):
	'''Runs the nmf function and computes the cosine similarity between topics
	of the H matrix in parallel  on 8 cores'''

	pool = Pool(processes = 8)
	results = pool.map(run_cos_H, parameter_list)
	return results


def frob_norm(k):
	'''Return the frobenius norm for a given number of topics'''

	#load tfidf and vectorizer
	tfidf = joblib.load('bills_tfidf_sparse.pkl')
	tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

	#run nmf
	nmf, nmf_topic_dict = fit_nmf(tfidf, k, tfidf_vectorizer)
	print nmf.reconstruction_err_, k
	return (nmf.reconstruction_err_, k)


def parallel_frob_norm(parameter_list):
	'''Runs the nmf function and computes the frobenius norm between V
	and the HW product in parallel on 36 cores'''

	pool = Pool(processes = 36)
	results = pool.map(frob_norm, parameter_list)
	return results


def plot_multi_topics_bar(mean_topic_con_nmf):
	'''Plots different topics on the same graph witha separate legend'''

	plt.bar(np.arange(12), mean_topic_con_nmf[110], label = 'Terror', color = 'b')
	plt.bar(np.arange(12), mean_topic_con_nmf[76], label = 'Iraq', color = 'g')
	plt.bar(np.arange(12), mean_topic_con_nmf[188], label = 'Budget', color = 'r')
	plt.bar(np.arange(12), mean_topic_con_nmf[205], label = 'Katrina', color = 'c')
	#plt.plot(mean_topic_con_nmf[214], label = 'STEM')
	#plt.plot(mean_topic_con_nmf[220], label = 'Internet')
	plt.bar(np.arange(12), mean_topic_con_nmf[277], label = 'Greenhouse', color = 'm')
	plt.bar(np.arange(12), mean_topic_con_nmf[286], label = 'Care', color = 'y')
	con = ['1993', '1995', '1997', '1999', '2001', '2003', '2005', '2007', '2009', '2011', '2013', '2015']
	#plt.title('Topics by Congress/Year')
	plt.xlabel('Congress/Year')
	plt.ylabel('Prevalence')
	ax = plt.gca()
	ax.set_xticks(np.arange(12))
	ax.set_xticklabels(con)
	# create a second figure for the legend
	figLegend = plt.figure(figsize = (1.5,1.3))
	# produce a legend for the objects in the other figure
	plt.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left', ncol = 3, mode="expand")	
	plt.show()


def plot_multi_topics(mean_topic_con_nmf):
	'''Plots different topics on the same graph witha separate legend'''

	with plt.style.context('fivethirtyeight'):
		plt.plot(mean_topic_con_nmf[110], label = 'Terror', color = 'b')
		plt.plot(mean_topic_con_nmf[76], label = 'Iraq', color = 'g')
		plt.plot(mean_topic_con_nmf[188], label = 'Budget', color = 'r')
		plt.plot(mean_topic_con_nmf[205], label = 'Katrina', color = 'c')
		#plt.plot(mean_topic_con_nmf[214], label = 'STEM', color = 'm')
		#plt.plot(mean_topic_con_nmf[220], label = 'Internet', color = 'y')
		plt.plot(np.arange(12), mean_topic_con_nmf[277], label = 'Greenhouse', color = 'k')
		plt.plot(np.arange(12), mean_topic_con_nmf[286], label = 'Care', color = 'mediumspringgreen')
		con = ['1993', '1995', '1997', '1999', '2001', '2003', '2005', '2007', '2009', '2011', '2013', '2015']
		#plt.title('Topics by Congress/Year')
		plt.xlabel('Congress/Year')
		plt.ylabel('Prevalence')
		ax = plt.gca()
		ax.set_xticks(np.arange(12))
		ax.set_xticklabels(con)
		# create a second figure for the legend
		figLegend = plt.figure(figsize = (1.5,1.3))
		# produce a legend for the objects in the other figure
		plt.figlegend(*ax.get_legend_handles_labels(), loc = 'upper left', ncol = 3, mode="expand")	
	plt.show()


def plot_cosin_sim():
	'''Plots the cosine similarity for H & W matrices'''

	#load data
	H_sim = joblib.load('avg_d_k_H_10to60.pkl')+joblib.load('avg_d_k_H_61to100.pkl')+joblib.load('avg_d_k_H_101to164.pkl')+joblib.load('avg_d_k_H_165to205.pkl')+joblib.load('avg_d_k_H_205to520.pkl')
	W_sim = joblib.load('avg_d_k_W_10to60.pkl')+joblib.load('avg_d_k_W_61to100.pkl')+joblib.load('avg_d_k_W_101to172.pkl')+joblib.load('avg_d_k_W_175to355.pkl')+joblib.load('avg_d_k_W_355to520.pkl')
	
	#Separate K and distances
	H_sim = [x for x in H_sim if x[1] % 5 == 0]
	H_sim_x = map(lambda x: x[1], H_sim)
	H_sim_y = map(lambda x: x[0], H_sim)

	#Plot H
	plt.plot(H_sim_x, H_sim_y, label = 'Between Topics', color = 'g')

	#Separate K and distances
	W_sim = [x for x in W_sim if x[1] % 5 == 0]
	W_sim_x = map(lambda x: x[1], W_sim)
	W_sim_y = map(lambda x: x[0], W_sim)

	#Plot W
	plt.plot(W_sim_x, W_sim_y, label = 'Within Topics', color = 'b')

	#Legend and labels
	plt.xlabel('Number of Topics')
	plt.ylabel('Average Cosine Similarity')
	plt.vlines(300, 0.65, 1, color = 'r')
	plt.legend()
	plt.show()

if __name__ == '__main__':
    tfidf = joblib.load('bills_tfidf_sparse_full.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer_full.pkl')
    tf = joblib.load('bills_tf_sparse.pkl')
    tf_vectorizer = joblib.load('tf_vectorizer.pkl')
    n_topics = 300
    ntopic_pur_list = topic_purity_maximizer(tfidf, tfidf_vectorizer)
    #tf = joblib.load('bills_tf_sparse.pkl')
    #tf_vectorizer = joblib.load('tf_vectorizer.pkl')
    W = joblib.load('W_300_full.pkl')
    nmf = joblib.load('nmf_300_full.pkl')
    nmf_topic_dict = joblib.load('nmf_topic_dict_300_full.pkl')
    nmf, nmf_topic_dict = fit_nmf(tfidf, n_topics, tfidf_vectorizer)
    H = nmf.components_ 
    #lda, lda_topic_dict = fit_lda(tf)
	mean_topic_con_nmf = get_H_W(nmf, tfidf)
	reverse_lookup = {word: idx for idx, word in enumerate(np.array(tfidf_vectorizer.get_feature_names()))}
	average_coherence_k = run_topic_coherence(tfidf, reverse_lookup)
	avg_d_k_H_525 = parallel_run_cos_H(range(205,525,5))
	avg_d_k_W_525 = parallel_run_cos_W(range(355,525,5))
	frob_norm_list = parallel_frob_norm(range(30,206,5))











