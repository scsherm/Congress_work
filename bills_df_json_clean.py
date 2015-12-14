import pandas as pd 
from pymongo import MongoClient
import sunlight as sl
import json
import re
import numpy as np 
import yaml
sl.config.API_KEY = '531ee226fadb4c298d0594f0543cabd8'


def to_df():
	'''Pull data from mongo to df in pandas'''
	client = MongoClient() #initiate mongo client
	db = client.bills_database #connect to votes database
	data = db.bills_table #connect ot votes table
	bills_json_df = pd.DataFrame(list(data.find()))
	return bills_json_df

def get_party_dict():
	'''read in legislators yaml file and create a dictionary of their party affiliation'''
	legislators = yaml.load(open('legislators_historical.yaml'))
	legislators_party_dict = {}
	for i in legislators:
		legislators_party_dict[i['name']['last'], i['name']['first']] = i['terms'][len(i['terms'])-1].get('party')
	return legislators_party_dict


def get_sponsor_party(bills_json_df, legislators_party_dict):
	'''get the sponsor party (if possible), and return dummies for the sponsor party'''
	bills_json_df['sponsor_name'] = bills_json_df.sponsor.map(lambda x: x.get('name') if type(x) == dict else {})
	bills_json_df['sponsor_name'] = bills_json_df.sponsor_name.map(lambda x: re.sub(r'[^\w\s]', '', x).split(' ') if type(x) == unicode else {})
	bills_json_df['sponsor_name'] = bills_json_df.sponsor_name.map(lambda x: (x[0], x[1]) if type(x) == list else [])
	bills_json_df['sponsor_party'] = bills_json_df.sponsor_name.map(lambda line: legislators_party_dict.get(line, np.nan) if type(line) == tuple else np.nan)
	bills_json_df = pd.concat([bills_json_df,pd.get_dummies(bills_json_df.sponsor_party)], axis = 1)
	return bills_json_df

def get_new_attributes(bills_json_df):
	'''collect attributes of certain columns'''
	bills_json_df['num_cosponsors'] = bills_json_df.cosponsors.map(lambda x: len(x))
	bills_json_df['num_committees'] = bills_json_df.committees.map(lambda x: len(x))
	bills_json_df['num_amendments'] = bills_json_df.amendments.map(lambda x: len(x))
	bills_json_df = pd.concat([bills_json_df,pd.get_dummies(bills_json_df.subjects_top_term)], axis = 1)
	bills_json_df.drop('Accounting and auditing', axis = 1, inplace = True)
	bills_json_df['request'] = bills_json_df['by_request']
	bills_json_df.set_index('bill_id', inplace = True)
	return bills_json_df
	

if __name__ == '__main__':
	bills_json_df = to_df()
	legislators_party_dict = get_party_dict()
	bills_json_df = get_sponsor_party(bills_json_df, legislators_party_dict)
	bills_json_df = get_new_attributes(bills_json_df)


