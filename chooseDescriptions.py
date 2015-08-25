import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pprint as pp
import json, math, re, csv, itertools
import nltk.stem.snowball as snowball
from bs4 import BeautifulSoup
from datetime import datetime


modes = ["word","histogram","wordcloud","topic-in-a-box"]
wordNums = ["5","10","20"]
file_topicJSON = open("nyt-50-topics-documents.json","r")
topicJSON = json.loads(file_topicJSON.read())
path_to_documents = "../findingtopic/dataset/documents/"

with open('csv_backup_3/Answer.csv', mode='r') as infile:
    reader = csv.DictReader(infile)
    answers = [rows for rows in reader]

# INITIALIZE RESULT SET
descriptions = {}
for ti in range(50):
	descriptions[ti] = {}
	for wordNum in wordNums:
		descriptions[ti][wordNum]={}
		for mode in modes:
			descriptions[ti][wordNum][mode]=[]
# pp.pprint(descriptions)

for answer in answers:
	# print answer
	if answer['mode']!="topic-in-a-box":
		descriptions[int(answer['topicIdx'])][str(answer['wordNum'])][answer['mode']].append(answer)
	else:
		if int(answer['topicIdx'])>0:
			fixedTID = int(answer['topicIdx'])-1
		else:
			fixedTID = 49 	
		answer['topicIdx']=str(fixedTID)
		descriptions[fixedTID][str(answer['wordNum'])][answer['mode']].append(answer)

# print descriptions[0]

# pot = descriptions[35]["5"]["word"]
# times = [datetime.strptime(x['created'], "%Y-%m-%dT%H:%M:%S") for x in pot]
# pot = sorted(pot, key=lambda x: datetime.strptime(x['created'], "%Y-%m-%dT%H:%M:%S") )
# descriptions[35]["5"]["word"] = pot
# pp.pprint(descriptions[35]["5"]["topic-in-a-box"])

for ti in range(50):
	for wordNum in wordNums:
		for mode in modes:
			pot = descriptions[ti][wordNum][mode]
			pot_sorted = sorted(pot, key=lambda x: datetime.strptime(x['created'], "%Y-%m-%dT%H:%M:%S"))
			descriptions[ti][wordNum][mode] = pot_sorted[:5]
				# pp.pprint(pot)				


with open('csv_backup_3/Descriptions.csv', mode='w') as wp:
	a = csv.writer(wp, delimiter='|')
	a.writerow(['topicIdx','wordNum','shortOrLong','mode','dID','key','label'])
	for ti in range(50):
		for wordNum in wordNums:
			for shortOrLong in ['short','long']:
				for mode in modes:
					dlist = descriptions[ti][wordNum][mode]
					for di, desc in enumerate(dlist):
						label = desc[shortOrLong]
						a.writerow([desc['topicIdx'],desc['wordNum'],shortOrLong,desc['mode'],di, desc['key'],label])



# with open('csv_backup_3/Descriptions.json', mode='w') as infile:
#     json.dump(descriptions, infile, indent=4)
			

# for ti in range(50):
# 	for wordNum in wordNums:
# 		for mode in modes:
# 			pot = descriptions[ti][wordNum][mode]
# 			pp.pprint([p['created'] for p in pot])
