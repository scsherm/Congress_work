from get_filepaths import get_filepaths
import pandas as pd 

def to_df():
	txt_file_list = get_filepaths()
	txt_list = []
	for filename in txt_file_list:
		with open(filename) as f:
			print f
			txt_list.append(f.read())
	bill_txt_df = pd.DataFrame(txt_list)
	return bill_txt_df

if __name__ == '__main__':
	bill_txt_df = to_df()