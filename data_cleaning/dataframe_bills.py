import sys
sys.path.append('/Users/scsherm/Documents/Congress_work/Unbalanced_Data')
sys.path.append('/Users/scsherm/Documents/Congress_work/topic_modeling')
sys.path.append('/Users/scsherm/Documents/Congress_work/models')
sys.path.append('/Users/scsherm/Documents/Congress_work/data_cleaning')
import pandas as pd 
import numpy as np 
from group_texts import group_texts

def bills_to_df(txt_dict):
	'''pulls the txt data into a pandas dataframe'''
	bills_df = pd.DataFrame.from_dict(txt_dict, orient = 'index')
	return bills_df

if __name__ == '__main__':
	txt_dict = group_texts()
	bills_df = bills_to_df(txt_dict)