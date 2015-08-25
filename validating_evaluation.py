import numpy as np
from numpy import arange,array,ones,linalg
import scipy.stats as stats
import matplotlib.pyplot as plt
import pprint as pp
import json, math, re, csv, itertools
import nltk.stem.snowball as snowball
from bs4 import BeautifulSoup
import random



modes = ["word","histogram","wordcloud","topic-in-a-box"]
wordNums = ["5","10","20"]



with open('csv_backup_4/Evaluation.csv', mode='r') as infile:
    reader = csv.DictReader(infile)
    evals = [rows for rows in reader]

# evals_done = [ev for ev in evals if ev['done']=="True" and "bad" not in ev['worst'] and "bad" not in ev["best"]]
evals_done = [ev for ev in evals if ev['done']=="True" and "2015-03-15" in ev['created']]


pp.pprint([ev for ev in evals_done if "u'0-5-word-4-short'" in ev['players']])

updated_bad = []

print "BAD BAD BAD BAD BAD BADBAD BAD BADBAD BAD BADBAD BAD BAD"
for ev in evals_done:
	if "bad" in ev['best'] or "bad" in ev['worst']:
		updated_bad.append(ev['updated'])

print updated_bad

dict_eval = {}
for topicIdx in range(50):
	dict_eval[topicIdx]={}
	for wordNum in wordNums:
		dict_eval[topicIdx][wordNum]={}
		for shortOrLong in ["short","long"]:
			dict_eval[topicIdx][wordNum][shortOrLong]={}
			for descNum in range(5):
				dict_eval[topicIdx][wordNum][shortOrLong][descNum]=[]

print "===================================================================================="

for ev in evals_done:
	info = ev["players"].split(",")[0].replace("[u'","").replace("'","").split("-")
	topicIdx = int(info[0])
	wordNum = info[1]
	mode = info[2]
	descNum = int(info[3])
	shortOrLong = info[4]
	if mode!="word":
		print "WARNING! "+ str(info)
	else:
		# print info
		dict_eval[topicIdx][wordNum][shortOrLong][descNum].append(ev)

for topicIdx in range(50):
	for wordNum in wordNums:
		for shortOrLong in ["short","long"]:
			for descNum in range(5):
				if len(dict_eval[topicIdx][wordNum][shortOrLong][descNum])!=1:
					print "u'%d-%s-%d-%s : "%(topicIdx,wordNum,descNum,shortOrLong) + str(len(dict_eval[topicIdx][wordNum][shortOrLong][descNum]))
					for ev in dict_eval[topicIdx][wordNum][shortOrLong][descNum]:
						print str(ev['players']) + str(ev["best"]) + str(ev["worst"]) 
						# if "bad" in ev["best"] or "bad" in ev["worst"]:
						# 	print "Found bad in result"
						# 	print ev
					# print dict_eval[topicIdx][wordNum][shortOrLong][descNum]
					# print "-----"


# print "MULTIPLE EVALUATIONS"
# topicIdx = 0
# for wordNum in wordNums:
# 	for mode in modes:
# 		for descNum in range(5):
# 			for shortOrLong in ["short","long"]:
# 				evl = [ev for ev in evals_done if "u'%d-%s-%s-%d-%s"%(topicIdx,wordNum,mode,descNum,shortOrLong)]
# 				if len(evl)!=1:
# 					print "u'%d-%s-%s-%d-%s"%(topicIdx,wordNum,mode,descNum,shortOrLong)
# 					print evl 


# dict_eval = {}
# for wordNum in wordNums:
# 	dict_eval[wordNum]={}
# 	for mode in modes+["algorithm"]:
# 		dict_eval[wordNum][mode]={}
# 		for shortOrLong in ["short","long"]:
# 			dict_eval[wordNum][mode][shortOrLong]={'best':0, 'worst':0}

	