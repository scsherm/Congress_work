import os

def get_filepaths():
	txt_file_list = []
	#scan subfolders for txt files
	for root, dirs, files in os.walk("/Volumes/scsherm/congress/data"):
		print os.path.abspath(root) 
		if os.path.abspath(root).__contains__('/votes'):
			print 'yes'
			for file in files:
				if file.endswith(".json"):
					txt_file_list.append(os.path.join(root, file))
	return txt_file_list

if __name__ == '__main__':
	txt_file_list = get_filepaths()