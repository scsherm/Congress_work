from pymongo import MongoClient
from get_filepaths import get_filepaths_votes
import json

def mongo_dump():
	#initiate mongo
	client = MongoClient() 
	client.drop_database('votes_database')
	db = client['votes_database'] #create database
	tab = db['votes_table'] #create table
	vote_file_list = get_filepaths_votes()
	for filename in vote_file_list:
		with open(filename) as f:
			print f
			tab.insert(json.loads(f.read()))

if __name__ == '__main__':
	mongo_dump()