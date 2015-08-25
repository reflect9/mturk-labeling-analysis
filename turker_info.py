import pprint as pp
import json, math, re, csv, itertools, os
from datetime import datetime
import numpy as np


# COUNTING UNIQUE TUEKERS



exit()

path_to_batch_results = "./batch_results"
batch_files = os.listdir(path_to_batch_results)
dict_turkers = {}
for bf_name in batch_files:
	print bf_name
	with open(path_to_batch_results+"/"+bf_name, mode='r') as bf:
	    reader = csv.DictReader(bf)
	    for row in reader:
	    	if row['Title'] not in dict_turkers: dict_turkers[row['Title']]={}
	    	if row['WorkerId'] not in dict_turkers[row['Title']]: 
	    		dict_turkers[row['Title']][row['WorkerId']]=[]
	    	dict_turkers[row['Title']][row['WorkerId']].append(row['Answer.surveycode'])

# pp.pprint(dict_turkers)


tasks_per_turker = [5*len(hitlist) for tid, hitlist in dict_turkers['Word labeling tasks'].iteritems()]
print tasks_per_turker

print len(tasks_per_turker)
print np.mean(tasks_per_turker), np.std(tasks_per_turker), np.max(tasks_per_turker), np.min(tasks_per_turker)


for k,d in dict_turkers.iteritems():
	print len(d.keys())


# tasks_per_turker = [len(hitlist) for tid, hitlist in dict_turkers['Pick the best label for a set of documents'].iteritems()]
# print tasks_per_turker

# print np.mean(tasks_per_turker)
# print np.std(tasks_per_turker)
# print np.max(tasks_per_turker)
# print np.min(tasks_per_turker)



# with open('turkers.json','w') as outfile:
# 	json.dump(dict_turkers, outfile, indent = 4)


# file_topicJSON = open("nyt-50-topics-documents.json","r")
# topicJSON = json.loads(file_topicJSON.read())