import pandas as pd 
import numpy as np 

H = nmf.model.components_
W = nmf.transform(tfidf)
W2 = lda.transform(tf)
H2 = lda.model.components_
bills_df_congress = pd.read_pickle('bills_df_congress')
W = pd.DataFrame(W)
W2 = pd.DataFrame(W2)
W.set_index(bills_df_congress.index, inplace = True)
W2.set_index(bills_df_congress.index, inplace = True)
W.drop(W.index[np.where(W.isnull())[0][0]], inplace = True)
W2.drop(W2.index[np.where(W2.isnull())[0][0]], inplace = True)
mean_topic_con_nmf = W.groupby('congress').mean()
mean_topic_con_lda = W2.groupby('congress').mean()




con = ['103\n(1993-1994)', '104\n(1995-1996)', '105\n(1997-1998)', '106\n(1999-2000)', '107\n(2001-2002)', '108\n(2003-2004)', \
'109\n(2005-2006)', '110\n(2007-2008)', '111\n(2009-2010)', '112\n(2011-2012)', '113\n(2013-2014)', '114\n(2015-2016)']
for idx in mean_topic_con_nmf:
    plt.plot(mean_topic_con_nmf[idx], label=idx)
    ax = plt.gca()
    ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
    ax.set_xticklabels(con)
plt.legend()
plt.savefig('topics_over_congress_nmf.png', format = 'png')
plt.close()
plt.plot(mean_topic_con_lda)
ax = plt.gca()
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
ax.set_xticklabels(con)
plt.savefig('topics_over_congress_lda.png', format = 'png')
plt.close()