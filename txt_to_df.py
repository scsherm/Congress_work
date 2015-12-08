from get_filepaths import get_filepaths
from get_filepaths import get_filepaths_votes
import pandas as pd 
from pathlib import Path
import numpy as np
import json 

def group_texts():
	full_file_list = get_filepaths()
	txt_dict = {}
	for i in full_file_list:
		if i:
			print i #tracking
			max_find = np.array([len(open(file).read()) for file in i])
			max_find_num = np.where(max_find == max(max_find))[0][0]
			with open(i[max_find_num]) as f:
				txt_dict[i[max_find_num].rsplit('/')[8] + '-' + i[max_find_num].rsplit('/')[5]] = [f.read()]
	return txt_dict

def map_votes_to_bills(txt_dict):
	vote_file_list = get_filepaths_votes()
	for filename in vote_file_list:
		with open(filename) as f:
			vote_dict = json.loads(f.read())
			if 'bill' in vote_dict.keys():
				print filename
				bill_id = vote_dict['bill']['type'] + str(vote_dict['bill']['number']) + '-' + str(vote_dict['bill']['congress'])
				try:
					txt_dict[bill_id].append(vote_dict)
				except:
					pass
	return txt_dict



if __name__ == '__main__':
	txt_dict = group_texts()
	txt_dict = map_votes_to_bills(txt_dict)