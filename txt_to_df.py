from get_filepaths import get_filepaths
import pandas as pd 
from pathlib import Path
import numpy as np 

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




	# 	with open(filename) as f:
	# 		print f
	# 		txt_list.append(f.read())
	# bill_txt_df = pd.DataFrame(txt_list)
	# return bill_txt_df

if __name__ == '__main__':
	txt_dict = group_texts()