from get_filepaths import get_filepaths
import pandas as pd 
from pathlib import Path
import numpy as np 

def group_texts():
	full_file_list = get_filepaths()
	txt_list = []
	for i in full_file_list:
		if i:
			print i #tracking
			max_find = np.array([len(open(file).read()) for file in i])
			max_find_num = np.where(max_find == max(max_find))[0][0]
			with open(i[max_find_num]) as f:
				txt_list.append(f.read())
	return txt_list




	# 	with open(filename) as f:
	# 		print f
	# 		txt_list.append(f.read())
	# bill_txt_df = pd.DataFrame(txt_list)
	# return bill_txt_df

if __name__ == '__main__':
	bill_txt_df = to_df()