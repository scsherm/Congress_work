import pandas as pd 
import numpy as np 

def get_precent_party(line):
	'''return the percent of type of vote for a particular party'''
	if type(line) == list:
		countD, countR, count = 0, 0, 0
		if len(line) > 0:
			if type(line[0]) == dict:
				for item in line:
					if item['party'] == 'D':
						countD += 1
					elif item['party'] == 'R':
						countR += 1
					else:
						count += 1
				return {'countD': countD/float(countD+countR+count), 'countR': countR/float(countD+countR+count), 'count': count/float(countD+countR+count)}
			else:
				return {'countD': None, 'countR': None, 'count': None}
		else:
			return {'countD': None, 'countR': None, 'count': None}
	else:
		return {'countD': None, 'countR': None, 'count': None}


def get_bill_id():
	'''creates the bill_id from the bill column dictionary'''
	votes_df = pd.read_pickle('votes_df')
	#remove null values (not bills)
	votes_df = votes_df.iloc[np.where(votes_df.bill.notnull())[0],:]
	#extract bill_id from bill column dict
	votes_df['bill_id'] = votes_df.bill.map(lambda x: x['type']+str(x['number'])+'-'+str(x['congress']))
	return votes_df

def get_votes_data(votes_df):
	'''creates dummies, converts dates, and gets counts for votes'''
	votes_df['date'] = pd.to_datetime(votes_df.date)
	votes_df['num_yes'] = votes_df.votes.map(lambda x: len(x.get('Yea', x.get('Aye', []))))
	votes_df['num_no'] = votes_df.votes.map(lambda x: len(x.get('No', x.get('Nay', []))))
	votes_df['num_not_voting'] = votes_df.votes.map(lambda x: len(x.get('Not Voting', [])))
	votes_df['num_present'] = votes_df.votes.map(lambda x: len(x.get('Present', [])))
	votes_df['percent_yes_D'] = votes_df.votes.map(lambda x: get_precent_party(x.get('Yea', x.get('Aye', [])))['countD'])
	votes_df['percent_no_D'] = votes_df.votes.map(lambda x: get_precent_party(x.get('No', x.get('Nay', [])))['countD'])
	votes_df['percent_yes_R'] = votes_df.votes.map(lambda x: get_precent_party(x.get('Yea', x.get('Aye', [])))['countR'])
	votes_df['percent_no_R'] = votes_df.votes.map(lambda x: get_precent_party(x.get('No', x.get('Nay', [])))['countR'])
	votes_df['percent_not_voting_D'] = votes_df.votes.map(lambda x: get_precent_party(x.get('Not Voting', []))['countD'])
	votes_df['percent_not_voting_R'] = votes_df.votes.map(lambda x: get_precent_party(x.get('Not Voting', []))['countR'])
	votes_df['percent_present_D'] = votes_df.votes.map(lambda x: get_precent_party(x.get('Present', []))['countD'])
	votes_df['percent_present_R'] = votes_df.votes.map(lambda x: get_precent_party(x.get('Present', []))['countR'])
	votes_df['is_amendment'] = votes_df.amendment.notnull()
	votes_df = pd.concat([votes_df,pd.get_dummies(votes_df.category)], axis = 1)
	votes_df.drop('unknown', axis = 1, inplace = True)
	votes_df = pd.concat([votes_df,pd.get_dummies(votes_df.requires)], axis = 1)
	votes_df.drop('3/5', axis = 1, inplace = True)
	#votes_df = pd.concat([votes_df,pd.get_dummies(votes_df.result)], axis = 1)
	#votes_df.drop('Motion to Reconsider Rejected', axis = 1, inplace = True)
	votes_df = pd.concat([votes_df,pd.get_dummies(votes_df.session)], axis = 1)
	votes_df.drop('2002', axis = 1, inplace = True)
	return votes_df


def group_by_chamber_latest(votes_df):
	'''returns the latest votes for both, the house and senate'''
	return votes_df.ix[votes_df.groupby(['bill_id','chamber'])['date'].idxmax()]


if __name__ == '__main__':
	votes_df = get_bill_id()
	votes_df = get_votes_data(votes_df)
	votes_df = group_by_chamber_latest(votes_df)
	votes_df.set_index('bill_id', inplace=True)
