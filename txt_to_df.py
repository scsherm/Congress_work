from get_filepaths import get_filepaths
import pandas as pd 
from pathlib import Path

def group_texts():
	txt_file_list = get_filepaths()
	txt_list = []
	fl = [txt_file_list[0]]
	for i in range(len(txt_file_list)):
		print txt_file_list[i] #tracking
		if i > 0:
			if Path(txt_file_list[i]).parents[1] == Path(txt_file_list[i-1]).parents[1]:
				with open(txt_file_list[i]) as f:
					fl.append(f.read())
			else:
				print 'Next Section' #tracking
				txt_list.append(fl)
				with open(txt_file_list[i]) as f:
					fl = [f.read()]
	txt_list.append(fl)
	return txt_list




		with open(filename) as f:
			print f
			txt_list.append(f.read())
	bill_txt_df = pd.DataFrame(txt_list)
	return bill_txt_df

if __name__ == '__main__':
	bill_txt_df = to_df()