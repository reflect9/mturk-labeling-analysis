import numpy as np
from numpy import arange,array,ones,linalg
import scipy.stats as stats
import matplotlib.pyplot as plt
import pprint as pp
import json, math, re, csv, itertools, sys
import nltk.stem.snowball as snowball
from bs4 import BeautifulSoup
import random
from collections import Counter

modes = ["word","histogram","wordcloud","topic-in-a-box"]
modes_short = ["word","histogram","wordcloud","topic-in-a-box","algorithm"]
wordNums = ["5","10","20"]
lengths = ["short","long"]
cardinality = ["short","long"]

topic_coherence = [0.05, 0.05, 0.08, 0.08, 0.04, 0.07, 0.03, 0.05, 0.12, 0.01, 0.09, 0.11, 0.14, 0.13, 0.18, 0.11, 0.01, 0.18, 0.12, 0.06, 0.03, 0.06, 0.12, 0.02, 0.12, 0.21, 0.2, 0.05, 0.08, 0.05, 0.09, 0.2, 0.04, 0.03, 0.12, 0.15, 0.08, 0.14, 0.05, 0.14, 0.14, 0.03, 0.11, 0.13, 0.1, 0.07, 0.08, 0.06, 0.09, 0.07]

with open('csv_backup_3/Answer.csv', mode='r') as infile:
    reader = csv.DictReader(infile)
    answers = [rows for rows in reader]

with open('csv_backup_5/Answer_fixed.csv', mode='r') as infile:
    reader = csv.DictReader(infile)
    answers_fixed = [rows for rows in reader]

with open('csv_backup_5/Evaluation.csv', mode='r') as infile:
    reader = csv.DictReader(infile)
    evals = [rows for rows in reader]

with open('csv_backup_5/LazyTurker.csv', mode='r') as infile:
    reader = csv.DictReader(infile)
    lazyTurkers = [rows for rows in reader]





def splitKeyNames(knlist):
	knlist = re.sub(r'[\[\]]', '', knlist)
	knlist = re.sub(r"u\'", "", knlist)
	knlist = re.sub(r"\'", "", knlist)
	return knlist.split(", ")

def getProperty(kn):
	if kn=="algorithm": 
		return {"mode":"algorithm"}
	if "-word-" in kn:	mode="word"
	if "-wordcloud-" in kn:	mode="wordcloud"
	if "-histogram-" in kn:	mode="histogram"
	if "-topic-in-a-box-" in kn:	mode="topic-in-a-box"
	tokens = kn.split("-")
	topicIdx = tokens[0]
	wordNum = tokens[1]
	descNum = tokens[len(tokens)-2]
	shortOrLong = tokens[len(tokens)-1]
	return {'mode':mode, 'topicIdx': topicIdx, 'wordNum':wordNum, 'descNum':descNum, 'shortOrLong':shortOrLong }

def getProperty2(kn):
	if "-algorithm-" in kn:	mode="algorithm"
	if "-word-" in kn:	mode="word"
	if "-wordcloud-" in kn:	mode="wordcloud"
	if "-histogram-" in kn:	mode="histogram"
	if "-topic-in-a-box-" in kn:	mode="topic-in-a-box"
	tokens = kn.split("-")
	topicIdx = tokens[0]
	wordNum = tokens[1]
	descNum = tokens[len(tokens)-2]
	shortOrLong = tokens[len(tokens)-1]
	return {'mode':mode, 'topicIdx': topicIdx, 'wordNum':wordNum, 'descNum':descNum, 'shortOrLong':shortOrLong }

def calc_entropy(s):
	p, lns = Counter(s), float(len(s))
	return -sum( count/lns * math.log(count/lns, 2) for count in p.values())

def aggregate_votes(eval_list):
	dict_eval = {}
	for wordNum in wordNums:
		dict_eval[wordNum]={}
		for mode in modes+["algorithm"]:
			dict_eval[wordNum][mode]={}
			for shortOrLong in ["short","long"]:
				dict_eval[wordNum][mode][shortOrLong]={'best':0, 'worst':0, "best_divided":0, "worst_divided":0}
	for ev in eval_list:
		# print "------------------------"
		# print ev
		# print "BEST: "+ev['best']
		best =  splitKeyNames(ev['best'])
		# print best
		for kn in best:	
			mode = getProperty(kn)['mode']
			# print mode
			# print dict_eval[ev['wordNum']][mode][ev['shortOrLong']]
			dict_eval[ev['wordNum']][mode][ev['shortOrLong']]['best'] += 1
			dict_eval[ev['wordNum']][mode][ev['shortOrLong']]['best_divided'] += 1.0/len(best)
			# print "-->"+str(dict_eval[ev['wordNum']][mode][ev['shortOrLong']])
		# print "WORST: "+ev['worst']
		worst =  splitKeyNames(ev['worst'])
		# print worst
		for kn in worst:	
			mode = getProperty(kn)['mode']
			# print mode
			# print dict_eval[ev['wordNum']][mode][ev['shortOrLong']]
			dict_eval[ev['wordNum']][mode][ev['shortOrLong']]['worst'] += 1
			dict_eval[ev['wordNum']][mode][ev['shortOrLong']]['worst_divided'] += 1.0/len(worst)
			# print "-->"+ str(dict_eval[ev['wordNum']][mode][ev['shortOrLong']])
	return dict_eval
	
# ### pairwise chi square test
# list = [315.8,342.7, 309.3, 338.2, 192]
# list = [226.8,221,261,241.8,549.3]
# for i in list:
# 	for j in list:
# 		if list.index(j)<=list.index(i): continue
# 		print "v%d vs v%d => %s" % (list.index(i)+1, list.index(j)+1, str(stats.chisquare([i,j])))
# exit()

# #######################################################################################################################
# #######################################################################################################################
# #######################################################################################################################
# # ANALYSIS 1. DESCRIPTIVE REPORT

# print "TOTAL # OF EVALUATIONS: " + str(len(evals))
# print "# OF EVALUATIONS PER TOPIC: 60"
# dict = {i:0 for i in ["short","long"]}
# for ev in evals:
# 	dict[ev['shortOrLong']]+=1 
# print "# OF EVALUATIONS PER SHORT OR LONG: " + str([v for i,v in dict.iteritems()])

# data =[]
# for ev in evals:
# 	try:
# 		if ev['shortOrLong']=="long":
# 			data.append(float(ev['duration']))
# 	except ValueError:
# 		pass
# 		# print "Not a float:"+ev['duration']
# print "TIME SPENT PER TASK:   Avg.:%f, STD:%f, Median:%f, Max:%f, Min:%f"%(np.mean(data), np.std(data), np.median(data), np.max(data), np.min(data))




# #######################################################################################################################
# #######################################################################################################################
# #######################################################################################################################
# # ANALYSIS 2. POSTHOC ANALYSIS

### LENGTH OF SHORT LABELS AND QUALITY
###   : Compare length of best and worst short labels

# file_descJSON = open("Descriptions.json","r")
# descJSON = json.loads(file_descJSON.read())

# file_evaluationJSON = open("csv_backup_5/evaluation_dict.json","r")
# evaluationJSON = json.loads(file_evaluationJSON.read())

# algorithm_labels = {0: ['Smoking', 'Tobacco smoking', 'History of smoking'], 1: ['Eliot Spitzer', "State Children's Health Insurance Program", 'Eliot Spitzer political surveillance controversy'], 2: ['Television', 'Fox News Channel', 'Television news in the United States'], 3: ['Catholic Church', 'Catholic Church and Nazi Germany', 'Criticism of the Catholic Church'], 4: ['Street', 'Wall Street', '14th Street (Manhattan)'], 5: ['Hotel Jerome', 'Hotel', 'Dunbar Hotel'], 6: ['George W. Bush and the Iraq War', 'Iraq War', 'Iraq'], 7: ['Real estate', 'Real estate appraisal', 'AG Real Estate'], 8: ["Friedman's k-percent rule", 'Percent allocation management module', 'Five-Percent Nation'], 9: ['United States Ski Team', 'United States Ski and Snowboard Association', 'FIS Alpine Ski World Cup'], 10: ['Airport security', 'Air travel disruption after the 2010 Eyjafjallaj\xc3\xb6kull eruption', 'United Airlines Flight 93'], 11: ['Like a Virgin', 'Be Like That', 'Like a Prayer (song)'], 12: ['Internet', 'Web 2.0', 'Social web'], 13: ['Police', 'Police of The Wire', 'Misconduct in the Philadelphia Police Department'], 14: ['Health care in the United States', 'Health care in Canada', 'Health'], 15: ['Art film', 'Honolulu Museum of Art', 'Mexican art'], 16: ['Refusal of work', 'Care work', 'Job interview'], 17: ['Federal Court of Justice of Germany', 'Criminal law in the Waite Court', 'Court'], 18: ['History of the Los Angeles Lakers', 'Los Angeles Clippers', 'History of the Los Angeles Dodgers'], 19: ['Book', 'Night (book)', 'The New York Review of Books'], 20: ['Jersey City', 'Union City', 'Atlantic City'], 21: ['Alaska Public Safety Commissioner dismissal', 'News International phone hacking scandal', 'Recent history of the District of Columbia Fire and Emergency Medical Services Department'], 22: ['Family', 'Hitler family', 'Bront\xc3\xab family'], 23: ['Death of Michael Jackson', 'Death of the Family', 'Death squad'], 24: ['Super Bowl', 'Super Bowl XLII', 'Super Bowl XLI'], 25: ['High School for Gifted Students', 'High School for Gifted Students', 'Hunter College High School'], 26: ['Music', 'Rock music', '1960s in music'], 27: ['History of fashion design', 'Fashion', 'Fashion design'], 28: ['Foreign trade of the United States', 'Foreign policy of the United States', 'Nuclear weapons and the United States'], 29: ['Advertising', 'Article marketing', 'Criticism of advertising'], 30: ['Iraqi Army', 'Iraqi insurgency (2003\xe2\x80\x9311)', 'Iraq War'], 31: ['Food', 'Wine and food matching', 'Food and dining in the Roman Empire'], 32: ['Martin Dies', 'Wife selling', 'The Years'], 33: ['Like a Virgin', 'Fula people', 'Marathi people'], 34: ['Tree', 'Snow', 'Severe weather'], 35: ['Joint-stock company', 'Executive compensation in the United States', 'Executive compensation'], 36: ['Construction of the World Trade Center', 'Chicago Board of Trade Building', 'World Trade Center'], 37: ['Democratic-Republican Party', 'Southern Democrats', 'Republican Party presidential primaries'], 38: ['2011 Lamar Hunt U.S. Open Cup Final', 'Venus Williams', 'Serena Williams'], 39: ['Sustainable energy', 'Oil sands', 'Peak oil'], 40: ['New York Knicks', 'Knicks\xe2\x80\x93Nets rivalry', 'History of the Brooklyn Nets'], 41: ['Execution of Saddam Hussein', 'Saddam Hussein', 'Trial of Saddam Hussein'], 42: ['In Death characters', 'Death and state funeral of Ronald Reagan', 'Death and state funeral of Gerald Ford'], 43: ['Sex trafficking of women and children in Thailand', 'Women in China', 'Women in Islam'], 44: ['Misnomer dance theater', 'Alvin Ailey American Dance Theater', 'Brazz Dance Theater'], 45: ['Ford Motor Company', 'Car', 'Ford Taurus'], 46: ['Indianapolis Museum of Art', 'John F. Kennedy Center for the Performing Arts', 'National Museum of Women in the Arts'], 47: ['Tax', 'Tobin tax', 'Carbon tax'], 48: ['Major League Baseball on Fox', 'Yankees\xe2\x80\x93Red Sox rivalry', 'Baseball'], 49: ['Hong Kong', 'China', 'Chinese Canadian']}

# best_labels = []
# worst_labels = []

# for kn, vote in evaluationJSON.iteritems():
# 	p = getProperty2(kn)
# 	if p['mode']=="algorithm":
# 		label = algorithm_labels[int(p['topicIdx'])][0]
# 	else:
# 		label = descJSON[str(p['topicIdx'])][p['wordNum']][p['mode']][int(p['descNum'])]["short"]
# 	# print kn, '\t\t', label, vote
# 	for i in range(vote['best']):
# 		best_labels.append(label)
# 	for i in range(vote['worst']):
# 		worst_labels.append(label)

# best_lengths = [len(label.split(" ")) for label in best_labels]
# worst_lengths = [len(label.split(" ")) for label in worst_labels]


# print "BEST LABELS" 
# print "LENGTH:   Avg.:%f, STD:%f, Median:%f, Max:%f, Min:%f"%(np.mean(best_lengths), np.std(best_lengths), np.median(best_lengths), np.max(best_lengths), np.min(best_lengths))
# print "WORST LABELS" 
# print "LENGTH:   Avg.:%f, STD:%f, Median:%f, Max:%f, Min:%f"%(np.mean(worst_lengths), np.std(worst_lengths), np.median(worst_lengths), np.max(worst_lengths), np.min(worst_lengths))

# # print best_lengths
# # print worst_lengths

# print stats.ttest_ind(best_lengths,worst_lengths)
# print len(best_lengths),  len(worst_lengths)

# print stats.mannwhitneyu(best_lengths,worst_lengths)
# exit()
# plt.figure(facecolor="white")
# n, bins, patches = plt.hist(best_lengths)
# # plt.axis([0,12,0,600])
# plt.ylabel("Frequency")
# plt.xlabel("# of Words of Best Labels")
# plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)	
# plt.show()


### SELF-REPORTED CONFIDENCE AND QUALITY
###     : Are they correlated? 

# file_descJSON = open("Descriptions.json","r")
# descJSON = json.loads(file_descJSON.read())
# file_evaluationJSON = open("csv_backup_5/evaluation_dict.json","r")
# evaluationJSON = json.loads(file_evaluationJSON.read())

# vote_conf_dict = {}
# for kn, vote in evaluationJSON.iteritems():
# 	p = getProperty2(kn)
# 	if p["mode"]=="algorithm": continue
# 	if str(vote) not in vote_conf_dict: 
# 		vote_conf_dict[str(vote)] = []
# 	# print kn,vote
# 	desc = descJSON[str(p['topicIdx'])][p['wordNum']][p['mode']][int(p['descNum'])]
# 	conf = desc["conf"]
# 	# print desc
# 	# print conf_score_dict
# 	vote_conf_dict[str(vote)].append(int(conf))
# 	# print vote_conf_dict

# # print(vote_conf_dict)
# for v, conf_list in vote_conf_dict.iteritems():
# 	print str(v)+": Avg.:%f, STD:%f"%(np.mean(conf_list), np.std(conf_list))

# exit()



### WORD-SPECIFICITY AND QUALITY
###     : Word-specificity affects label quality?


# #######################################################################################################################
# #######################################################################################################################
# #######################################################################################################################
# # ANALYSIS 3. BEST / WORST / BEST-WORST


# BEST AND WORST SCORE PER DESCRIPTION
# data = {}
# # INIT
# for topicIdx in range(50):
# 	for wordNum in wordNums:
# 		for mode in modes:
# 			for descNum in range(5):
# 				for shortOrLong in ["short","long"]:
# 					kn=str(topicIdx)+"-"+wordNum+"-"+mode+"-"+str(descNum)+"-"+shortOrLong
# 					data[kn]={"worst":0, "best":0, "best_divided":0, "worst_divided":0}
# 		mode = "algorithm"
# 		kn=str(topicIdx)+"-"+wordNum+"-"+mode+"-0-short"
# 		data[kn]={"worst":0, "best":0, "best_divided":0, "worst_divided":0}
# # INIT END
# for ev in evals:
# 	# print ev['best']
# 	best =  splitKeyNames(ev['best'])
# 	for kn in best:
# 		if kn=="algorithm":
# 			kn=ev["topicIdx"]+"-"+ev["wordNum"]+"-"+"algorithm"+"-0-short"
# 		# if kn not in data: 
# 		# 	data[kn]={'best':0, 'worst':0}
# 		data[kn]['best']+=1
# 		data[kn]['best_divided']+=1.0/len(best)
# 	worst =  splitKeyNames(ev['worst'])
# 	for kn in worst:
# 		if kn=="algorithm":
# 			kn=ev["topicIdx"]+"-"+ev["wordNum"]+"-"+"algorithm"+"-0-short"
# 		# if kn not in data: 
# 		# 	data[kn]={'best':0, 'worst':0}
# 		data[kn]['worst']+=1
# 		data[kn]['worst_divided']+=1.0/len(worst)

# with open('csv_backup_5/evaluation_dict.json','w') as f:
# 	f.write(json.dumps(data, indent=4))



# AGGREGATED BEST AND WORST FOR 5 MODES
# data = {mode:{'best':0, 'worst':0} for mode in modes+["algorithm"]}
# for ev in evals:
# 	best =  splitKeyNames(ev['best'])
# 	for kn in best:
# 		data[getProperty(kn)['mode']]['best']+=1
# 	worst =  splitKeyNames(ev['worst'])
# 	for kn in worst:
# 		data[getProperty(kn)['mode']]['worst']+=1
# for k,v in data.iteritems():
# 	v['score']=v['best']-v['worst']
# print data


# INITIALIZE DICTIONARY FOR EVALUATION RESULTS 
# dict_eval = {}
# for wordNum in wordNums:
# 	dict_eval[wordNum]={}
# 	for mode in modes+["algorithm"]:
# 		dict_eval[wordNum][mode]={}
# 		for shortOrLong in ["short","long"]:
# 			dict_eval[wordNum][mode][shortOrLong]={'best':0, 'worst':0, "best_divided":0, "worst_divided":0}
# for ev in evals:
# 	# print "------------------------"
# 	# print ev
# 	# print "BEST: "+ev['best']
# 	best =  splitKeyNames(ev['best'])
# 	# print best
# 	for kn in best:	
# 		mode = getProperty(kn)['mode']
# 		# print mode
# 		# print dict_eval[ev['wordNum']][mode][ev['shortOrLong']]
# 		dict_eval[ev['wordNum']][mode][ev['shortOrLong']]['best'] += 1
# 		dict_eval[ev['wordNum']][mode][ev['shortOrLong']]['best_divided'] += 1.0/len(best)
# 		# print "-->"+str(dict_eval[ev['wordNum']][mode][ev['shortOrLong']])
# 	# print "WORST: "+ev['worst']
# 	worst =  splitKeyNames(ev['worst'])
# 	# print worst
# 	for kn in worst:	
# 		mode = getProperty(kn)['mode']
# 		# print mode
# 		# print dict_eval[ev['wordNum']][mode][ev['shortOrLong']]
# 		dict_eval[ev['wordNum']][mode][ev['shortOrLong']]['worst'] += 1
# 		dict_eval[ev['wordNum']][mode][ev['shortOrLong']]['worst_divided'] += 1.0/len(worst)
		# print "-->"+ str(dict_eval[ev['wordNum']][mode][ev['shortOrLong']])

# for mode in modes_short:
# 	for wordNum in wordNums:
# 		for cardi in cardinality:
# 			print mode, wordNum
# 			print dict_eval[wordNum][mode][cardi]['best_divided'], dict_eval[wordNum][mode][cardi]['worst_divided']


# exit()

# # LET'S CREATE CSV
# with open('evaluation.csv', 'w') as fp:
# 	writer = csv.writer(fp, delimiter=',')
# 	row = []
# 	for mode in modes_short:
# 		for wordNum in wordNums:
# 			for sl in ["short","long"]:
# 				row.append(dict_eval[wordNum][mode][sl]['best'])
# 	writer.writerow(row)
# 	row = []
# 	for mode in modes_short:
# 		for wordNum in wordNums:
# 			for sl in ["short","long"]:
# 				row.append(dict_eval[wordNum][mode][sl]['worst'])
# 	writer.writerow(row)

# exit()

# # COMPARING 5 MODES 
# fig = plt.figure(1,facecolor="white")
# count=0
# ind = np.arange(5)
# width=0.35
# best_data = []
# worst_data = []
# for mi, mode in enumerate(modes+["algorithm"]):
# 	best = 0
# 	worst = 0
# 	for wordNum in wordNums:
# 		best += dict_eval[wordNum][mode]["short"]["best"]
# 		worst += dict_eval[wordNum][mode]["short"]["worst"]
# 	best_data.append(best)
# 	worst_data.append(worst)
# print best_data
# print worst_data

# exit()

# count+=1
# ax = fig.add_subplot(3,1,count)
# best_data = []
# worst_data = []
# for mi, mode in enumerate(modes+["algorithm"]):
# 	dd = d[mode]
# 	best_data.append(dd['short']['best'])
# 	worst_data.append(dd['short']['worst'])
# ax_best= ax.bar(ind, best_data, width, color='b')
# ax_worst= ax.bar(ind+width, worst_data, width, color='r')
# ax.set_ylim(0,250)
# ax.set_ylabel(str(wordNum) + " words")
# xTickMarks = [mode for mode in modes+["algorithm"]]
# ax.set_xticks(ind+width)
# xtickNames = ax.set_xticklabels(xTickMarks)
# plt.setp(xtickNames)



# # 15 BAR CHART OF SHORT LABELS (ALSO SHOWS ALGORITHM) 
# fig = plt.figure(1,facecolor="white")
# count=0
# ind = np.arange(5)
# width=0.35
# for wordNum in wordNums:
# 	d = dict_eval[wordNum]
# 	count+=1
# 	ax = fig.add_subplot(3,1,count)
# 	best_data = []
# 	worst_data = []
# 	for mi, mode in enumerate(modes+["algorithm"]):
# 		dd = d[mode]
# 		best_data.append(dd['short']['best_divided'])
# 		worst_data.append(dd['short']['worst_divided'])
# 	ax_best= ax.bar(ind, best_data, width, color='b')
# 	ax_worst= ax.bar(ind+width, worst_data, width, color='r')
# 	ax.set_ylim(0,220)
# 	ax.set_ylabel(str(wordNum) + " words")
# 	xTickMarks = ["Word List","Word List w/ Bars","Word Cloud","Topic-in-a-box","Algorithm"]	
#  	ax.set_xticks(ind+width)
# 	xtickNames = ax.set_xticklabels(xTickMarks)
# 	plt.setp(xtickNames)

# # plt.legend([ax_best, ax_worst], ('Best','Worst'), loc='upper right')

# fig.text(0.03, 0.5, '# of best / worst votes', fontsize=15, ha='center', va='center', rotation='vertical')
# fig.text(0.5, 0.98, 'Label Evaluation (Short)', fontsize=17, ha='center', va='top')
# # ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), modes, loc=1)

# plt.show()
			

# BAR CHART OF LONG LABELS (ALSO SHOWS ALGORITHM) 
# fig = plt.figure(1,facecolor="white")
# count=0
# ind = np.arange(4)
# width=0.29
# for wordNum in wordNums:
# 	d = dict_eval[wordNum]
# 	count+=1
# 	ax = fig.add_subplot(3,1,count)
# 	best_data = []
# 	worst_data = []
# 	for mi, mode in enumerate(modes):
# 		dd = d[mode]
# 		best_data.append(dd['long']['best_divided'])
# 		worst_data.append(dd['long']['worst_divided'])
# 	ax_best= ax.bar(ind, best_data, width, color='b')
# 	ax_worst= ax.bar(ind+width, worst_data, width, color='r')
# 	ax.set_ylim(0,220)
# 	ax.set_ylabel(str(wordNum) + " words")
# 	xTickMarks = ["Word List","Word List w/ Bars","Word Cloud","Topic-in-a-box"]	
# 	ax.set_xticks(ind+width)
# 	xtickNames = ax.set_xticklabels(xTickMarks)
# 	plt.setp(xtickNames)

# # plt.legend([ax_best, ax_worst], ('Best','Worst'), loc='upper right')

# fig.text(0.03, 0.5, '# of best / worst votes', fontsize=15, ha='center', va='center', rotation='vertical')
# fig.text(0.5, 0.98, 'Label Evaluation (Long)', fontsize=17, ha='center', va='top')
# # ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), modes, loc=1)

# plt.show()
		
# exit()

# # # BAR CHART OF SHORT LABELS: DIFFERENCE 
# fig = plt.figure(1,facecolor="white")
# count=0
# ind = np.arange(5)
# width=0.35
# for wordNum in wordNums:
# 	d = dict_eval[wordNum]
# 	count+=1
# 	ax = fig.add_subplot(3,1,count)
# 	diff_data = []
# 	for mi, mode in enumerate(modes+["algorithm"]):
# 		dd = d[mode]
# 		diff_data.append(dd['short']['best']-dd['short']['worst'])
# 	ax.bar(ind+width/2, diff_data, width=width, color='g')
# 	ax.set_ylim(-70,50)
# 	ax.set_ylabel(str(wordNum) + " words")
# 	xTickMarks = [mode for mode in modes+["algorithm"]]
# 	ax.set_xticks(ind+width)
# 	xtickNames = ax.set_xticklabels(xTickMarks)
# 	plt.setp(xtickNames)
# 	plt.gca().grid(True)

# # plt.legend([ax_best, ax_worst], ('Best','Worst'), loc='upper right')

# fig.text(0.03, 0.5, '# of best votes - # of worst votes', fontsize=15, ha='center', va='center', rotation='vertical')
# fig.text(0.5, 0.98, 'Label Evaluation (Short)', fontsize=17, ha='center', va='top')
# # ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), modes, loc=1)
# plt.show()



# # # BAR CHART OF LONG LABELS: DIFFERENCE 
# fig = plt.figure(1,facecolor="white")
# count=0
# ind = np.arange(4)
# width=0.35
# for wordNum in wordNums:
# 	d = dict_eval[wordNum]
# 	count+=1
# 	ax = fig.add_subplot(3,1,count)
# 	diff_data = []
# 	for mi, mode in enumerate(modes):
# 		dd = d[mode]
# 		diff_data.append(dd['long']['best']-dd['long']['worst'])
# 	ax.bar(ind+width/2, diff_data, width=width, color='g')
# 	ax.set_ylim(-70,50)
# 	ax.set_ylabel(str(wordNum) + " words")
# 	xTickMarks = [mode for mode in modes]
# 	ax.set_xticks(ind+width)
# 	xtickNames = ax.set_xticklabels(xTickMarks)
# 	plt.setp(xtickNames)
# 	plt.gca().grid(True)

# # plt.legend([ax_best, ax_worst], ('Best','Worst'), loc='upper right')

# fig.text(0.03, 0.5, '# of best votes - # of worst votes', fontsize=15, ha='center', va='center', rotation='vertical')
# fig.text(0.5, 0.98, 'Label Evaluation (Long)', fontsize=17, ha='center', va='top')
# # ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), modes, loc=1)
# plt.show()

	# ax = fig.add_subplot(111)


# player_dict = {mode:0 for mode in modes}
# player_dict['algorithm'] = 0
# winner_dict = {mode:0 for mode in modes}
# loser_dict = {mode:0 for mode in modes}
# winner_dict['algorithm'] = 0
# loser_dict['algorithm'] = 0
# for ev in evals:
# 	players = [mode for mode in player_dict.keys() if mode+"-" in ev['players']]
# 	for w in players: player_dict[w] += 1

# 	winners = [mode for mode in winner_dict.keys() if mode+"-" in ev['best']]
# 	for w in winners: winner_dict[w] += 1

# 	losers = [mode for mode in loser_dict.keys() if mode+"-" in ev['worst']]
# 	for w in losers: loser_dict[w] += 1
# 	# print ev['best'] + " , " + str(winner)

# pp.pprint(player_dict)
# pp.pprint(winner_dict)
# pp.pprint(loser_dict)
# pp.pprint([winner_dict[mode] - loser_dict[mode] for mode in modes])






# #######################################################################################################################
# #######################################################################################################################
# #######################################################################################################################
# # ANALYSIS 4. TOPIC COHERENCE AND OTHER FACTORS

## PARSE nyt-topics-oc-fixed.txt
# with open('nyt-topics-oc-npmi-fixed.txt','r') as fp:
# 	coh=[]
# 	for line in fp.readlines():
# 		m = re.match(r'\[(.*)\]',line)
# 		if m!=None:
# 			coh.append(abs(float(m.group(1))))
# 	print coh
# 	print len(coh)
# exit()

def voting_report(evals):
	# short overall
	best_votes = {mode:0 for mode in modes_short}
	worst_votes = {mode:0 for mode in modes_short}
	for ev in evals:
		best_kn_list = splitKeyNames(ev['best'])
		worst_kn_list = splitKeyNames(ev['worst'])
		# print "----"
		# print "BEST: " + str(best_kn_list)
		# print "WORST: " + str(worst_kn_list)
		# print best_votes
		# print worst_votes
		for kn in best_kn_list:
			mode = getProperty(kn)['mode']
			best_votes[mode]+=1.0/len(best_kn_list)
		for kn in worst_kn_list:
			mode = getProperty(kn)['mode']
			worst_votes[mode]+=1.0/len(worst_kn_list)
		# print best_votes
		# print worst_votes
	return {"best_votes":best_votes, "worst_votes":worst_votes}

top_topic_and_prob = sorted(enumerate(topic_coherence), key=lambda tup: tup[1])[:13]
bottom_topic_and_prob = sorted(enumerate(topic_coherence), key=lambda tup: tup[1])[49-12:]
top_topic_idx = [tup[0] for tup in top_topic_and_prob]
bottom_topic_idx = [tup[0] for tup in bottom_topic_and_prob]

# ### TOP AND BOTTOM QUARTILE 
quartile_dict = {}
quartile_dict['all'] = [ev for ev in evals if ev['shortOrLong']=="short" ]
quartile_dict['top'] = [ev for ev in evals if int(ev['topicIdx']) in top_topic_idx and ev['shortOrLong']=="short" ]
quartile_dict['bottom'] = [ev for ev in evals if int(ev['topicIdx']) in bottom_topic_idx and ev['shortOrLong']=="short"]


### DRAW BARCHART FOR TOP AND BOTTOM QUARTILES
fig = plt.figure(1,facecolor="white")
count=0
ind = np.arange(5)
width=0.35
for qk in ['top','bottom']:
	print qk.upper()+" TOPICS"
	count+=1
	ax = fig.add_subplot(2,1,count)
	d = quartile_dict[qk]
	votes = voting_report(d)
	# for mode in modes_short:
		# print mode+":"+str(votes['best_votes'][mode])+","+str(votes['worst_votes'][mode])\
	best_votes = [votes['best_votes'][mode] for mode in modes_short]
	worst_votes = [votes['worst_votes'][mode] for mode in modes_short]
	print best_votes
	print worst_votes
	ax_best= ax.bar(ind, best_votes, width, color='b')
	ax_worst= ax.bar(ind+width, worst_votes, width, color='r')
	ax.set_ylim(0,150)
	ax.set_ylabel(qk + " topics", fontsize=20)
	xTickMarks = ["Word List","Word List w/ Bars","Word Cloud","Topic-in-a-box","Algorithm"]	
 	ax.set_xticks(ind+width)
	xtickNames = ax.set_xticklabels(xTickMarks, fontsize=14)
	plt.yticks(range(0,141,20),fontsize=13)
	plt.setp(xtickNames)

# plt.legend([ax_best, ax_worst], ('Best','Worst'), loc='upper right')

# fig.text(0.03, 0.5, '# of best / worst votes', fontsize=15, ha='center', va='center', rotation='vertical')
# fig.text(0.5, 0.98, 'Label Evaluation (Short)', fontsize=17, ha='center', va='top')
# ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), modes, loc=1)


fig.set_size_inches(10, 8)
fig.savefig('top-bottom-votes.pdf')


exit()


### RUN CHI-SQUARE TEST WITHIN TOP AND BOTTOM QUARTILES
# for qk, d in quartile_dict.iteritems():
# 	print qk.upper()+" 13 TOPICS"
# 	for cardi in cardinality:
# 		for bw in ['best','worst']:
# 			evals = [ev for ev in quartile_dict[qk] if ev['shortOrLong']==cardi]
# 			report = voting_report(evals)
# 			if cardi=="short":
# 				votes = [report[bw+"_votes"][m] for m in modes_short]
# 				print "%6s %5s : v1[%d] v2[%d] v3[%d] v4[%d] v5[%d] : CHI-SQUARE: %s" % (cardi, bw, votes[0], votes[1], votes[2],votes[3], votes[4], str(stats.chisquare(votes)[1]))   
# 			else:
# 				votes = [report[bw+"_votes"][m] for m in modes]
# 				print "%6s %5s : v1[%d] v2[%d] v3[%d] v4[%d] : CHI-SQUARE: %s" % (cardi, bw, votes[0], votes[1], votes[2],votes[3], str(stats.chisquare(votes)[1]))   

# 			# print "%6s %5s : %20s"%(cardi, bw, str(stats.chisquare(votes)[1]))
# 	print ""


# ### COMPARE TIME, CONFIDENCE AND WORD LENGTH BETWEEN TOP AND BOTTOM QUARTILES
# quartile_dict['all'] = [ans for ans in answers_fixed]
# quartile_dict['top'] = [ans for ans in answers_fixed if int(ans['topicIdx']) in top_topic_idx]
# quartile_dict['bottom'] = [ans for ans in answers_fixed if int(ans['topicIdx']) in bottom_topic_idx]

# for qk, d in quartile_dict.iteritems():
# 	print qk.upper()+" 13 TOPICS"
# 	ans_list = [ans for ans in quartile_dict[qk]]
# 	duration_list = [float(ans['duration']) for ans in ans_list]
# 	confidence_list = [float(ans['conf']) for ans in ans_list]
# 	short_length_list = [len(ans['short'].split()) for ans in ans_list]
# 	long_length_list = [len(ans['long'].split()) for ans in ans_list]
# 	# print "%6s: Duration[%f], Confidence[%f], Label[%f], Sentence[%f]" % (wordNum, np.mean(duration_list), np.mean(confidence_list), np.mean(short_length_list), np.mean(long_length_list))   
# 	print "%6s: Duration[%f], Confidence[%f], Label[%f], Sentence[%f]" % ("all", np.mean(duration_list), np.mean(confidence_list), np.mean(short_length_list), np.mean(long_length_list))   
		
# 	for wordNum in wordNums:
# 		ans_list = [ans for ans in quartile_dict[qk] if ans['wordNum']==wordNum]
# 		duration_list = [float(ans['duration']) for ans in ans_list]
# 		confidence_list = [float(ans['conf']) for ans in ans_list]
# 		short_length_list = [len(ans['short'].split()) for ans in ans_list]
# 		long_length_list = [len(ans['long'].split()) for ans in ans_list]
# 		# print "%6s: Duration[%f], Confidence[%f], Label[%f], Sentence[%f]" % (wordNum, np.mean(duration_list), np.mean(confidence_list), np.mean(short_length_list), np.mean(long_length_list))   
# 		print "%6s: Duration[%f], Confidence[%f], Label[%f], Sentence[%f]" % (wordNum, np.mean(duration_list), np.mean(confidence_list), np.mean(short_length_list), np.mean(long_length_list))   
		
# 		# print "%6s %5s : %20s"%(cardi, bw, str(stats.chisquare(votes)[1]))
# 	print ""

### COMPARE SHARED WORDS BETWEEN TOP AND BOTTOM QUARTILES

f = open('csv_backup_5/topic_terms_with_frequency.json', mode='r')
termFreq = json.loads(f.read())

print np.mean(topic_coherence)
print np.std(topic_coherence)



topic_duration = {i:[] for i in range(50)}
confidence = {i:[] for i in range(50)}
for a in answers_fixed:
	if float(a['duration'])>0:
		topic_duration[int(a['topicIdx'])].append(float(a['duration']))
	confidence[int(a['topicIdx'])].append(float(a['conf']))


topic_duration_list = [np.mean(v) for k,v in topic_duration.iteritems()]
confidence_list = [np.mean(v) for k,v in confidence.iteritems()]

print [(i,v)  for i,v in enumerate(confidence_list)]
cl_with_idx = [(i,v)  for i,v in enumerate(confidence_list)]
print sorted(cl_with_idx, key=lambda (i,v): v)

top_topic_and_prob = sorted(enumerate(topic_coherence), key=lambda tup: tup[1])[:13]
bottom_topic_and_prob = sorted(enumerate(topic_coherence), key=lambda tup: tup[1])[49-12:]
top_topic_idx = [tup[0] for tup in top_topic_and_prob]
bottom_topic_idx = [tup[0] for tup in bottom_topic_and_prob]

# print top_topic_and_prob
# print top_topic_idx
# print bottom_topic_and_prob
# print bottom_topic_idx
# exit()

# ##### EXPORT DATA FOR R
# with open('csv_backup_5/coherence_time_conf_3.csv', 'w') as fp:
# 	writer = csv.writer(fp, delimiter=',')
# 	writer.writerow(['coherence','time','confidence','quartile'])
# 	for i in range(50):
# 		quartile="middle"
# 		if i in top_topic_idx:
# 			quartile="top"
# 		if i in bottom_topic_idx:
# 			quartile="bottom"
# 		writer.writerow([topic_coherence[i],topic_duration_list[i],confidence_list[i],quartile])
# exit()
# print [(confidence_list[i],topic_coherence[i]) for i in range(len(confidence_list))]


# print topic_duration_list
fig= plt.figure(facecolor="white")
plt.gcf().subplots_adjust(bottom=0.15)
ax = plt.subplot(1,2,1)
ax.tick_params(axis='x', labelsize=23)
ax.tick_params(axis='y', labelsize=23)
plt.axis([30,90,0,0.23])
plt.scatter(topic_duration_list, topic_coherence, color="black", s=40)
plt.ylabel("Topic Coherence", fontsize=30)
plt.xlabel("Avg. Labeling time", fontsize=30)
# plt.show()

# pp.pprint(confidence_list)
# plt.figure(facecolor="white")
ax = plt.subplot(1,2,2)
plt.axis([2.6,4.2,0,0.23])
ax.tick_params(axis='x', labelsize=23)
ax.tick_params(axis='y', labelsize=23)

plt.scatter(confidence_list, topic_coherence, color="black", s=40)
# plt.ylabel("Topic Coherence")
plt.xlabel("Avg. Confidence", fontsize=30)
# plt.show()

fig.set_size_inches(18, 8)
fig.savefig('coh_time_conf.pdf')

exit()




# #######################################################################################################################
# #######################################################################################################################
# #######################################################################################################################
# # ANALYSIS 5. ENTROPY Of VOTING; which setting was easy or difficult to decide best and worst. 


### PRECALCULATING VOTING ENTROPY
# file_evaluationJSON = open("csv_backup_5/evaluation_dict.json","r")
# eval_dict = json.loads(file_evaluationJSON.read())

# entropy = {}
# for topicIdx in range(50):
# 	entropy[topicIdx]={}
# 	for wordNum in wordNums:
# 		entropy[topicIdx][wordNum]={}
# 		for shortOrLong in cardinality:
# 			entropy[topicIdx][wordNum][shortOrLong] = {"best":[], "worst":[]}

# for ev in evals:
# 	# print "--------------------"
# 	# print entropy[int(ev['topicIdx'])][ev['wordNum']][ev['shortOrLong']]
# 	# print "best:"+str(ev['best'])
# 	# print "worst:"+str(ev['worst'])
# 	best =  splitKeyNames(ev['best'])
# 	for kn in best:
# 		mode = getProperty(kn)['mode']
# 		entropy[int(ev['topicIdx'])][ev['wordNum']][ev['shortOrLong']]['best'].append(mode)
# 	worst =  splitKeyNames(ev['worst'])
# 	for kn in worst:
# 		mode = getProperty(kn)['mode']
# 		entropy[int(ev['topicIdx'])][ev['wordNum']][ev['shortOrLong']]['worst'].append(mode)
# 	# print entropy[int(ev['topicIdx'])][ev['wordNum']][ev['shortOrLong']]
# for topicIdx in range(50):
# 	for wordNum in wordNums:
# 		for shortOrLong in cardinality:
# 			best_en = calc_entropy(entropy[topicIdx][wordNum][shortOrLong]["best"])
# 			worst_en = calc_entropy(entropy[topicIdx][wordNum][shortOrLong]["worst"])
# 			entropy[topicIdx][wordNum][shortOrLong]["best_entropy"] = best_en
# 			entropy[topicIdx][wordNum][shortOrLong]["worst_entropy"] = worst_en

# with open("csv_backup_5/entropy_by_topic.json","w") as f:
# 	f.write(json.dumps(entropy,indent=4));


### ENTROPY FOR BEST and WORST (overall)
### : entropy of best votes is higher or lower than entropy of worst votes. 
# best_entropy_list = []
# worst_entropy_list = []
# for topicIdx in range(50):
# 	for wordNum in wordNums:
# 		for shortOrLong in cardinality:
# 			best_entropy_list.append(entropy[topicIdx][wordNum][shortOrLong]["best_entropy"])
# 			worst_entropy_list.append(entropy[topicIdx][wordNum][shortOrLong]["worst_entropy"])


# bin = np.arange(0,2,0.1)
# plt.figure(facecolor="white")
# plt.subplot(1,2,1)
# n, bins, patches = plt.hist(best_entropy_list, bins = bin)
# plt.ylabel("Frequency")
# plt.axis([0,2,0,100])
# plt.xlabel("Entropy")
# plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)	
# plt.title("Best voting entropy: overall")

# plt.subplot(1,2,2)
# n, bins, patches = plt.hist(worst_entropy_list, bins = bin)
# plt.ylabel("Frequency")
# plt.xlabel("Entropy")
# plt.axis([0,2,0,100])
# plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)	
# plt.title("Worst voting entropy: overall")
# plt.show()

# print "For all settings"
# data = best_entropy_list
# print "BEST VOTE ENTROPY:   Avg.:%f, STD:%f, Median:%f, Max:%f, Min:%f, Normality:%s"%(np.mean(data), np.std(data), np.median(data), np.max(data), np.min(data), str(stats.normaltest(data)))
# data = worst_entropy_list
# print "WORST VOTE ENTROPY:   Avg.:%f, STD:%f, Median:%f, Max:%f, Min:%f, Normality:%s"%(np.mean(data), np.std(data), np.median(data), np.max(data), np.min(data), str(stats.normaltest(data)))
# print stats.ttest_ind(best_entropy_list, worst_entropy_list)


### ENTROPY FOR BEST and WORST (for different wordNums)
### : entropy of best votes is higher or lower than entropy of worst votes. 

# for wordNum in wordNums:
# 	for shortOrLong in cardinality:
# 		best_entropy_list = []
# 		worst_entropy_list = []
# 		for topicIdx in range(50):
# 			best_entropy_list.append(entropy[topicIdx][wordNum][shortOrLong]["best_entropy"])
# 			worst_entropy_list.append(entropy[topicIdx][wordNum][shortOrLong]["worst_entropy"])
# 		print wordNum+" words, " +shortOrLong+"" 
# 		data = best_entropy_list
# 		print "BEST VOTE ENTROPY:   Avg.:%f, STD:%f, Median:%f, Max:%f, Min:%f"%(np.mean(data), np.std(data), np.median(data), np.max(data), np.min(data))
# 		data = worst_entropy_list
# 		print "WORST VOTE ENTROPY:   Avg.:%f, STD:%f, Median:%f, Max:%f, Min:%f"%(np.mean(data), np.std(data), np.median(data), np.max(data), np.min(data))




### ENTROPY OF GOOD TOPIC AND BAD TOPICS
### : Are coherent topics easier to vote?  (we might need to separate best and worst.)





# 		if kn=="algorithm":
# 			kn=ev["topicIdx"]+"-"+ev["wordNum"]+"-"+"algorithm"+"-0-short"
# 		# if kn not in data: 
# 		# 	data[kn]={'best':0, 'worst':0}
# 		data[kn]['best']+=1
# 		data[kn]['best_divided']+=1.0/len(best)
# 	worst =  splitKeyNames(ev['worst'])
# 	for kn in worst:
# 		if kn=="algorithm":
# 			kn=ev["topicIdx"]+"-"+ev["wordNum"]+"-"+"algorithm"+"-0-short"
# 		# if kn not in data: 
# 		# 	data[kn]={'best':0, 'worst':0}
# 		data[kn]['worst']+=1
# 		data[kn]['worst_divided']+=1.0/len(worst)






# for ev in evals:
# 	# print ev['best']
# 	best =  splitKeyNames(ev['best'])
# 	for kn in best:
# 		if kn=="algorithm":
# 			kn=ev["topicIdx"]+"-"+ev["wordNum"]+"-"+"algorithm"+"-0-short"
# 		# if kn not in data: 
# 		# 	data[kn]={'best':0, 'worst':0}
# 		data[kn]['best']+=1
# 		data[kn]['best_divided']+=1.0/len(best)
# 	worst =  splitKeyNames(ev['worst'])
# 	for kn in worst:
# 		if kn=="algorithm":
# 			kn=ev["topicIdx"]+"-"+ev["wordNum"]+"-"+"algorithm"+"-0-short"
# 		# if kn not in data: 
# 		# 	data[kn]={'best':0, 'worst':0}
# 		data[kn]['worst']+=1
# 		data[kn]['worst_divided']+=1.0/len(worst)

# with open('csv_backup_5/evaluation_dict.json','w') as f:
# 	f.write(json.dumps(data, indent=4))



# #######################################################################################################################
# #######################################################################################################################
# #######################################################################################################################
# # ANALYSIS 6. LAZY TURKERS


# print "LAZY TURKERS: HITs that could not identify intentionally bad label"
# print "total missed HITs: %d" % len(lazyTurkers)

# dummy_tasks=[]

# for lt in lazyTurkers:
# 	tasks = json.loads(lt['evalResults'])
# 	dummy_tasks.append([t for t in tasks if t['memo']=="dummy"][0])
# 	print [t for t in tasks if t['memo']=="dummy"][0]

# for t in dummy_tasks:
# 	# print "---"
# 	# print t['worst']
# 	# print sum([[getProperty(kn)['mode'] for kn in splitKeyNames(kn)] for kn in t['worst']], [])
# 	t['worst_mode']= sum([[getProperty(kn)['mode'] for kn in splitKeyNames(kn)] for kn in t['worst']], [])



# for wordNum in wordNums:
# 	print "%s words: %d" % (wordNum, len([t for t in dummy_tasks if t['wordNum']==wordNum]))

# for mode in modes:
# 	print "# of worst vots for %s : %d" % (mode, len([t for t in dummy_tasks if mode in t['worst_mode'] ]))


# top_topic_and_prob = sorted(enumerate(topic_coherence), key=lambda tup: tup[1])[:13]
# bottom_topic_and_prob = sorted(enumerate(topic_coherence), key=lambda tup: tup[1])[49-12:]
# top_topic_idx = [tup[0] for tup in top_topic_and_prob]
# bottom_topic_idx = [tup[0] for tup in bottom_topic_and_prob]

# print "top topic dummy : %d" % len([t for t in dummy_tasks if int(t['topicIdx']) in top_topic_idx])
# print "bottom topic dummy : %d" % len([t for t in dummy_tasks if int(t['topicIdx']) in bottom_topic_idx])


# print len(evals)

# print [t['topicIdx'] for t in dummy_tasks]

# plt.figure(facecolor="white")
# n, bins, patches = plt.hist([int(t['topicIdx']) for t in dummy_tasks], bins=50)
# # plt.axis([0,12,0,600])
# plt.ylabel("Frequency")
# plt.xlabel("# of Words of Best Labels")
# plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)	
# plt.show()











