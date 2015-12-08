import os

def get_filepaths_votes():
	vote_file_list = []
	#scan subfolders for txt files
	for congr_num in os.listdir("/Volumes/scsherm/congress/data"):
		if not congr_num.startswith('.'):
			for root, dirs, files in os.walk("/Volumes/scsherm/congress/data/" + congr_num + "/votes"):
				for file in files:
					if file.endswith(".json"):
						print os.path.join(root, file)
						vote_file_list.append(os.path.join(root, file))
	return vote_file_list

def get_filepaths():
	full_file_list = []

	#scan subfolders for txt files
	for root, dirs, files in os.walk("/Volumes/scsherm/congress/data"):
		if os.path.abspath(root).endswith(('/hconres', '/hjres', '/hr', '/hres', '/s', '/sconres', '/sjres', '/sres')):
			for directory in os.listdir(os.path.abspath(root)):
				d_file_list = []
				if not directory.startswith('.'):
					for r, d, fs in os.walk(os.path.abspath(root) + '/' + directory):
						for file in fs:
							if file.endswith(".txt"):
								print os.path.join(r, file)
								d_file_list.append(os.path.join(r, file))
				full_file_list.append(d_file_list)
	return full_file_list


if __name__ == '__main__':
	full_file_list = get_filepaths()