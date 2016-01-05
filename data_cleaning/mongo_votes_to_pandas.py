import pandas as pd 
from pymongo import MongoClient


def to_df():
	'''pulls the votes json data from mongo into a pandas dataframe'''
	client = MongoClient() #initiate mongo client
	db = client.votes_database #connect to votes database
	data = db.votes_table #connect ot votes table
	votes_df = pd.DataFrame(list(data.find()))
	return votes_df

if __name__ == '__main__':
	votes_df = to_df()