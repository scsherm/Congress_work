from pymongo import MongoClient
from get_filepaths import get_filepaths_type
import json

def mongo_dump(datatype = 'votes'):
	'''for a given data type (bills or votes) will dump all json files to 
	mongoDB'''
	#initiate mongo
	client = MongoClient()
	try: 
		client.drop_database('{}_database'.format(datatype))
	except:
		pass
	db = client['{}_database'.format(datatype)] #create database
	tab = db['{}_table'.format(datatype)] #create table
	datatype_file_list = get_filepaths_type(datatype = datatype)
	for filename in datatype_file_list:
		with open(filename) as f:
			print f
			tab.insert(json.loads(f.read()))

if __name__ == '__main__':
	mongo_dump(datatype = 'bills')