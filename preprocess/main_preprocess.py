import sys
from preprocess.preprocess_one import *

def main_preprocess(preprocess_method,out_path,targets,NODE_TYPE_DICT):
	if preprocess_method == "1":
		return preprocess_one(out_path,targets,NODE_TYPE_DICT)
	
	elif preprocess_method == "2":
		return preprocess_two(out_path,targets,NODE_TYPE_DICT)

	else:
		sys.exit("== There is no such model name ==")