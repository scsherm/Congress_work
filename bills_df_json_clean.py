import pandas as pd 
from pymongo import MongoClient
import sunlight as sl
import json
sl.config.API_KEY = '531ee226fadb4c298d0594f0543cabd8'


def to_df():
	'''Pull data from mongo to df in pandas'''
	client = MongoClient() #initiate mongo client
	db = client.bills_database #connect to votes database
	data = db.bills_table #connect ot votes table
	bills_json_df = pd.DataFrame(list(data.find()))
	return bills_json_df


def get_df():
	'''read from stored csv'''
	bills_json_df = pd.read_csv('bills_json_df')
	return bills_json_df

def get_party(line):
	try:
		party = sl.congress.legislators(first_name = line[1], last_name = line[0])[0]['party']
		print 'got party'
		return party
	except:
		print 'exception'
		return {}

def get_percent_party(bills_json_df):
	bills_json_df['sponsor_name'] = bills_json_df.sponsor.map(lambda x: x.get('name').split(', ') if type(x) == dict else {})
	bills_json_df['sponsor_party'] = bills_json_df.sponsor_name.map(lambda line: get_party(line))

def get_new_attributes():
	bills_json_df['num_cosponsors'] = bills_json_df.cosponsors.map(lamnda x: len(x))
	bills_json_df = pd.concat([bills_json_df,pd.get_dummies(bills_json_df.subjects_top_term)], axis = 1)
	bills_json_df.drop('Accounting and auditing', axis = 1, inplace = True)
	bills_json_df['request'] = bills_json_df['by_request']
	bills_json_df.set_index('bill_id', inplace = True)
	

if __name__ == '__main__':
	bills_json_df = get_df()