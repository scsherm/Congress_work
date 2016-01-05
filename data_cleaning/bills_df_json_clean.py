import sys
sys.path.append('/Users/scsherm/Documents/Congress_work/Unbalanced_Data')
sys.path.append('/Users/scsherm/Documents/Congress_work/topic_modeling')
sys.path.append('/Users/scsherm/Documents/Congress_work/models')
sys.path.append('/Users/scsherm/Documents/Congress_work/data_cleaning')
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
	bills_json_df['num_D_S'] = bills_json_df.congress
	bills_json_df['num_R_S'] = bills_json_df.congress
	bills_json_df['num_D_H'] = bills_json_df.congress
	bills_json_df['num_R_H'] = bills_json_df.congress
	bills_json_df.set_index('bill_id', inplace = True)
	return bills_json_df


def add_percent_party(bills_json_df, congress, p_cham, value):
	bills_json_df[p_cham][np.where(bills_json_df.congress == congress)[0]] = value
	return bills_json_df


if __name__ == '__main__':
	bills_json_df = to_df()
	legislators_party_dict = get_party_dict()
	bills_json_df = get_sponsor_party(bills_json_df, legislators_party_dict)
	bills_json_df = get_new_attributes(bills_json_df)
	bills_json_df = add_percent_party(bills_json_df, '103', 'num_D_S', 57)
	bills_json_df = add_percent_party(bills_json_df, '103', 'num_R_S', 43)
	bills_json_df = add_percent_party(bills_json_df, '104', 'num_D_S', 48)
	bills_json_df = add_percent_party(bills_json_df, '104', 'num_R_S', 52)
	bills_json_df = add_percent_party(bills_json_df, '105', 'num_D_S', 45)
	bills_json_df = add_percent_party(bills_json_df, '105', 'num_R_S', 55)
	bills_json_df = add_percent_party(bills_json_df, '106', 'num_D_S', 45)
	bills_json_df = add_percent_party(bills_json_df, '106', 'num_R_S', 55)
	bills_json_df = add_percent_party(bills_json_df, '107', 'num_D_S', 50)
	bills_json_df = add_percent_party(bills_json_df, '107', 'num_R_S', 50)
	bills_json_df = add_percent_party(bills_json_df, '108', 'num_D_S', 48)
	bills_json_df = add_percent_party(bills_json_df, '108', 'num_R_S', 51)
	bills_json_df = add_percent_party(bills_json_df, '109', 'num_D_S', 44)
	bills_json_df = add_percent_party(bills_json_df, '109', 'num_R_S', 55)
	bills_json_df = add_percent_party(bills_json_df, '110', 'num_D_S', 49)
	bills_json_df = add_percent_party(bills_json_df, '110', 'num_R_S', 49)
	bills_json_df = add_percent_party(bills_json_df, '111', 'num_D_S', 58)
	bills_json_df = add_percent_party(bills_json_df, '111', 'num_R_S', 42)
	bills_json_df = add_percent_party(bills_json_df, '112', 'num_D_S', 51)
	bills_json_df = add_percent_party(bills_json_df, '112', 'num_R_S', 47)
	bills_json_df = add_percent_party(bills_json_df, '113', 'num_D_S', 53)
	bills_json_df = add_percent_party(bills_json_df, '113', 'num_R_S', 45)
	bills_json_df = add_percent_party(bills_json_df, '114', 'num_D_S', 44)
	bills_json_df = add_percent_party(bills_json_df, '114', 'num_R_S', 54)
	bills_json_df = add_percent_party(bills_json_df, '103', 'num_D_H', 258)
	bills_json_df = add_percent_party(bills_json_df, '103', 'num_R_H', 176)
	bills_json_df = add_percent_party(bills_json_df, '104', 'num_D_H', 204)
	bills_json_df = add_percent_party(bills_json_df, '104', 'num_R_H', 230)
	bills_json_df = add_percent_party(bills_json_df, '105', 'num_D_H', 207)
	bills_json_df = add_percent_party(bills_json_df, '105', 'num_R_H', 226)
	bills_json_df = add_percent_party(bills_json_df, '106', 'num_D_H', 211)
	bills_json_df = add_percent_party(bills_json_df, '106', 'num_R_H', 223)
	bills_json_df = add_percent_party(bills_json_df, '107', 'num_D_H', 212)
	bills_json_df = add_percent_party(bills_json_df, '107', 'num_R_H', 221)
	bills_json_df = add_percent_party(bills_json_df, '108', 'num_D_H', 205)
	bills_json_df = add_percent_party(bills_json_df, '108', 'num_R_H', 229)
	bills_json_df = add_percent_party(bills_json_df, '109', 'num_D_H', 202)
	bills_json_df = add_percent_party(bills_json_df, '109', 'num_R_H', 231)
	bills_json_df = add_percent_party(bills_json_df, '110', 'num_D_H', 236)
	bills_json_df = add_percent_party(bills_json_df, '110', 'num_R_H', 199)
	bills_json_df = add_percent_party(bills_json_df, '111', 'num_D_H', 257)
	bills_json_df = add_percent_party(bills_json_df, '111', 'num_R_H', 178)
	bills_json_df = add_percent_party(bills_json_df, '112', 'num_D_H', 193)
	bills_json_df = add_percent_party(bills_json_df, '112', 'num_R_H', 242)
	bills_json_df = add_percent_party(bills_json_df, '113', 'num_D_H', 201)
	bills_json_df = add_percent_party(bills_json_df, '113', 'num_R_H', 234)
	bills_json_df = add_percent_party(bills_json_df, '114', 'num_D_H', 188)
	bills_json_df = add_percent_party(bills_json_df, '114', 'num_R_H', 247)


