import numpy as np
from numpy import arange,array,ones,linalg
import scipy.stats as stats
import scipy.special as special
import matplotlib.pyplot as plt
import pprint as pp
import json, math, re, csv, itertools
import nltk.stem.snowball as snowball
from bs4 import BeautifulSoup
import random


modes = ["word","histogram","wordcloud","topic-in-a-box"]
wordNums = ["5","10","20"]
cardinality = ["short","long"]
file_topicJSON = open("nyt-50-topics-documents.json","r")
topicJSON = json.loads(file_topicJSON.read())
path_to_documents = "../findingtopic/dataset/documents/"

with open('csv_backup_3/Answer.csv', mode='r') as infile:
    reader = csv.DictReader(infile)
    answers = [rows for rows in reader]

with open('csv_backup_5/Answer_fixed.csv', mode='r') as infile:
    reader = csv.DictReader(infile)
    answers_fixed = [rows for rows in reader]


####### ETA SQUARE, ANOVA LIBRARY   
def FPvalue( *args):
	df_btwn, df_within = __degree_of_freedom_( *args)
	mss_btwn = __ss_between_( *args) / float( df_btwn)   
	mss_within = __ss_within_( *args) / float( df_within)
	F = mss_btwn / mss_within	
	P = special.fdtrc( df_btwn, df_within, F)
	return( F, P)

def EffectSize( *args):
	return( float( __ss_between_( *args) / __ss_total_( *args)))

def __concentrate_( *args):
	v = list( map( np.asarray, args))
	vec = np.hstack( np.concatenate( v))
	return( vec)

def __ss_total_( *args):
	vec = __concentrate_( *args)
	ss_total = sum( (vec - np.mean( vec)) **2)
	return( ss_total)

def __ss_between_( *args):
	grand_mean = np.mean( __concentrate_( *args))
	ss_btwn = 0
	for a in args:
		ss_btwn += ( len(a) * ( np.mean( a) - grand_mean) **2)
	return( ss_btwn)

def __ss_within_( *args):
	return( __ss_total_( *args) - __ss_between_( *args))

def __degree_of_freedom_( *args):
	args = list( map( np.asarray, args))
	df_btwn = len( args) - 1
	df_within = len( __concentrate_( *args)) - df_btwn - 1
	return( df_btwn, df_within)

def flatten(list_of_lists):
	return [val for sublist in list_of_lists for val in sublist]

# EXTRA. FIXING Answer.csv 

# for ans in answers:
# 	if ans['mode']=="topic-in-a-box":
# 		if ans['topicIdx']=='0':
# 			ans["topicIdx"]='49'
# 		else:
# 			ans["topicIdx"] = str(int(ans["topicIdx"])-1)
# print [a['short'] for a in answers if a['topicIdx']=="0"]
# with open('csv_backup_5/Answer_fixed.csv', 'w') as fp:
# 	writer = csv.writer(fp, delimiter=',')
# 	writer.writerow(answers[0].keys())
# 	for a in answers:
# 		writer.writerow(a.values())



# ANALYSIS 1. OVERALL DURATION 
# plt.figure(facecolor="white")
# durations = [int(a['duration']) for a in answers if int(a['duration'])>1]
# durations_log = [np.log2(int(a['duration'])) for a in answers if int(a['duration'])>1]
# n, bins, patches = plt.hist(durations, bins = np.logspace(0, 10, 50, base=2))
# plt.gca().set_xscale("log")
# # plt.axis([0,600,0,600])
# plt.ylabel("Frequency")
# plt.xlabel("Time spent per task (seconds)")
# plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)	
# plt.title("Overall Time Spent For Each Task ")

# plt.show()

# data =durations_log
# print "%s: #:%d, Avg:%f, Median:%f, Stdev:%f, min:%d , max:%d, normalityTest:%s" % ('DURATIONS', len(data), np.mean(data), np.median(data), np.std(data), min(data), max(data), str(stats.normaltest(data)))


##### EXPORT TIME DATA FOR R: twoway anova
# with open('csv_backup_5/labeling_time_R.csv', 'w') as fp:
# 	writer = csv.writer(fp, delimiter=',')
# 	writer.writerow(['visualization','cardinality','time','confidence'])
# 	for a in answers:
# 		writer.writerow([a['mode'],a['wordNum']+"w",a['duration'],a['conf']])
# exit()
# END OF EXPORTING


# ANALYSIS 1a. COMPARE DURATION BETWEEN MODES
# all_durations = []
# plt.figure(1,facecolor="white")
# for i in range(len(modes)):
# 	mode = modes[i]
# 	durations = [np.log(int(a['duration'])) for a in answers if a['mode']==mode]
# 	durations = [d for d in durations if d>1]
# 	all_durations.append(durations)
# 	fig = plt.subplot(4,1,i+1)
# 	plt.ylabel(mode)
# 	# plt.ylabel("Frequency")
# 	print "%20s: #:%d, Avg:%f, Median:%f, Stdev:%f, min:%d , max:%d, normalityTest:%s" % (mode, len(durations), np.mean(durations), np.median(durations), np.std(durations), min(durations), max(durations), str(stats.normaltest(durations)))
# 	n, bins, patches = plt.hist(durations, bins=range(0, 200 + 10, 10))
# 	# plt.axis([0,400,0,200])
# 	plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)	
# plt.gcf().suptitle("Time Spent For Each Task ")
# plt.show()
# plt.figure(2,facecolor="white")
# plt.boxplot(all_durations)
# # plt.ylim([0,120])
# plt.show()






# ANALYSIS 1b. COMPARE DURATION BETWEEN MODES AND WORDNUMS
# plt.figure(3,facecolor="white")
# count=0
# all_durations = []
# for i in range(len(modes)):
# 	for j in range(len(wordNums)):
# 		count+=1
# 		mode = modes[i]
# 		wordNum = wordNums[j]
# 		durations = [int(a['duration']) for a in answers if a['mode']==mode and a['wordNum']==wordNum and int(a['duration'])>1]
# 		durations_log = [np.log2(int(a['duration'])) for a in answers if a['mode']==mode and a['wordNum']==wordNum and int(a['duration'])>1]
# 		# durations = [d for d in durations if d>1]
# 		all_durations.append(durations_log)
# 		plt.subplot(len(modes),len(wordNums),count)
# 		if count==1 or count==4 or count==7 or count==10:
# 			plt.ylabel(mode)   
# 		if count<10:
# 			plt.gca().get_xaxis().set_visible(False)
# 		else:
# 			plt.xlabel("Time spent (seconds) using "+wordNum+" words")   
# 		data = durations
# 		print "%20s: Avg:%f, Stdev:%f, normalityTest:%s" % (mode+" "+str(wordNum), np.mean(data), np.std(data), str(stats.normaltest(data)[1]))
# 		# print "%20s: #:%d, Avg:%f, Median:%f, Stdev:%f, min:%d , max:%d, normalityTest:%s" % (mode+" "+str(wordNum), len(data), np.mean(data), np.median(data), np.std(data), min(data), max(data), str(stats.normaltest(data)))
# 		# n, bins, patches = plt.hist(durations, bins=range(0, 200 + 5, 5))
# 		n, bins, patches = plt.hist(durations, bins = np.logspace(0, 10, 50, base=2))
# 		plt.gca().set_xscale("log")
# 		plt.axis([0,1000,0,40])
# 		plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)	
# plt.show()
# plt.figure(4,facecolor="white")
# plt.title("Time Spent For Each Task")
# plt.boxplot(all_durations)
# # plt.ylim([0,320])
# plt.xticks(range(1,13), [x+"-"+str(y) for x in modes for y in wordNums], rotation='vertical')
# plt.subplots_adjust(bottom=0.3)
# plt.show()



### 4*3 bar chart of labeling time
# plt.figure(3,facecolor="white")
# ax = plt.subplot(111)
# time_average = []
# for mode in modes:
# 	for wordNum in wordNums:
# 		times = [float(a['duration']) for a in answers if a['mode']==mode and a['wordNum']==wordNum and float(a['duration'])>0]
# 		# print [a for a in answers if a['mode']==mode and a['wordNum']==wordNum and float(a['duration'])<1.0]
# 		# times = [int(a['duration']) for a in answers if a['mode']==mode and a['wordNum']==wordNum]
# 		# print [int(a['duration']) for a in answers if a['mode']==mode and a['wordNum']==wordNum and a['duration']<1.0]
# 		time_average.append(np.mean(times))
# print time_average
# std_error = [2.731,2.852,3.263,  4.633,3.060,3.604,  2.9,2.289,5.228,  3.109,3.390,4.441]
# ax.bar(np.arange(12),time_average, yerr=std_error, ecolor='red', color='black', width=0.5)

# plt.show()




# exit()
### ANOVA ON DURATIONS
# Across modes for 5 words 


# cardinality over each mode
# duration_dict={}
# print "One way ANOVA across cardinalities"
# for mode in modes: 
# 	duration_dict[mode]={}
# 	for wordNum in wordNums:
# 		duration_dict[mode][wordNum] = [log(int(a['duration'])) for a in answers if a['mode']==mode and a['wordNum']==wordNum and int(a['duration'])>1]
# 	data = [v for k,v in duration_dict[mode].iteritems()]
# 	print mode + ":" + str(stats.f_oneway(data[0],data[1],data[2])) + str(EffectSize(data[0],data[1],data[2]))
# # print duration_dict








# ANALYSIS 2a. COMPARE WORD LENGTH BEWTEEN MODES

### BASIC MEAN ANd SD
# print np.mean([len(a['short'].split(" ")) for a in answers])
# print np.std([len(a['short'].split(" ")) for a in answers])
# print np.mean([len(a['long'].split(" ")) for a in answers])
# print np.std([len(a['long'].split(" ")) for a in answers])

# exit()
# plt.figure(5,facecolor="white")
# count=0
# all_values = []
# for i in range(len(modes)):
# 	for j in range(len(wordNums)):
# 		count+=1
# 		mode = modes[i]
# 		wordNum = wordNums[j]
# 		values = [len(a['long'].split(" ")) for a in answers if a['mode']==mode and a['wordNum']==wordNum]
# 		all_values.append(values)
# 		plt.subplot(len(modes),len(wordNums),count)
# 		if count==1 or count==4 or count==7 or count==10:
# 			plt.ylabel(mode)   
# 		if count<10:
# 			plt.gca().get_xaxis().set_visible(False)
# 		else:
# 			plt.xlabel("Word lengths with "+wordNum+"-word settings")  
# 		print "%20s: #:%d, Avg:%f, Median:%f, Stdev:%f, min:%d , max:%d" % (mode+" "+str(wordNum), len(values), np.mean(values), np.median(values), np.std(values), min(values), max(values))
# 		n, bins, patches = plt.hist(values, bins=range(0, 30, 1))
# 		plt.axis([0,20,0,35])
# 		plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)	
# plt.gcf().suptitle("Length of Sentence Description (# of words)")
# plt.show()
# plt.figure(6,facecolor="white")
# plt.boxplot(all_values)
# plt.ylim([0,30])
# plt.xticks(range(1,13), [x+"-"+str(y) for x in modes for y in wordNums], rotation='vertical')
# plt.subplots_adjust(bottom=0.3)
# plt.show()

# ANALYSIS 2b. COMPARE SHORT DESCRIPTION LENGTH BEWTEEN MODES
# plt.figure(5,facecolor="white")
# count=0
# all_values = []
# for i in range(len(modes)):
# 	for j in range(len(wordNums)):
# 		count+=1
# 		mode = modes[i]
# 		wordNum = wordNums[j]
# 		values = [len(a['short'].split(" ")) for a in answers if a['mode']==mode and a['wordNum']==wordNum]
# 		all_values.append(values)
# 		plt.subplot(len(modes),len(wordNums),count)
# 		if count==1 or count==4 or count==7 or count==10:
# 			plt.ylabel(mode)   
# 		if count<10:
# 			plt.gca().get_xaxis().set_visible(False)
# 		else:
# 			plt.xlabel("Word lengths with "+wordNum+"-word settings")  
# 		print "%20s: #:%d, Avg:%f, Median:%f, Stdev:%f, min:%d , max:%d" % (mode+" "+str(wordNum), len(values), np.mean(values), np.median(values), np.std(values), min(values), max(values))
# 		n, bins, patches = plt.hist(values, bins=range(0, 30, 1))
# 		plt.axis([0,5,0,200])
# 		plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)	
# plt.gcf().suptitle("Length of Short Summary (# of words)")		
# plt.show()
# plt.figure(6)
# plt.boxplot(all_values)
# plt.ylim([0,5])

# plt.show()


##### EXPORT CONFIDENCE DATA FOR R: twoway anova
# for wordNum in wordNums:
# 	with open('csv_backup_5/confidence_table_%s.csv'%wordNum, 'w') as fp:
# 		writer = csv.writer(fp, delimiter=',')
# 		writer.writerow(['conf','index','mode'])
# 		alist = [a for a in answers if a['wordNum']==wordNum]
# 		adict = {mode:[] for mode in modes}
# 		for a in alist:
# 			adict[a['mode']].append(a)
# 		for i in range(np.min([len(l) for mk,l in adict.iteritems()])):
# 			for mode in modes: 
# 				a = adict[mode][i]
# 				writer.writerow([a['conf'],i, a['mode']])
# exit()
## END OF EXPORTING



# ANALYSIS 3. SELF-REPORTED CONFIDENCE
# data = [int(a['conf']) for a in answers]
# print "avg=%f, sd=%f" % (np.mean(data), np.std(data))

# for wordNum in wordNums:
# 	for mode in modes:
# 		data = [int(a['conf']) for a in answers if a['mode']==mode and a['wordNum']==wordNum]
# 		print "%s, %s : avg=%f, sd=%f" % (wordNum, mode, np.mean(data), np.std(data))
# exit()
# confidence = [int(a['conf']) for a in answers]
# n, bins, patches = plt.hist(confidence, bins=100)
# plt.figure(1,facecolor="white")
# plt.axis([0,1000,0,800])
# plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)	
# plt.show()
# print "%s: #:%d, Avg:%f, Median:%f, Stdev:%f, min:%d , max:%d" % ('confidence', len(confidence), np.mean(confidence), np.median(confidence), np.std(confidence), min(confidence), max(confidence))

#### Friedman chi square test
# for wordNum in wordNums:
# 	c_word = [int(a['conf']) for a in answers if a['mode']=="word"] + [4,4,4,4]
# 	c_histogram = [int(a['conf']) for a in answers if a['mode']=="histogram"]
# 	c_wordcloud = [int(a['conf']) for a in answers if a['mode']=="wordcloud"]
# 	c_topicbox = [int(a['conf']) for a in answers if a['mode']=="topic-in-a-box"]
# 	print len(c_word), len(c_histogram), len(c_wordcloud), len(c_topicbox)
# 	result = stats.friedmanchisquare(c_word, c_histogram, c_wordcloud, c_topicbox)
# 	print result 
# exit()

#### ONE WAY ANOVA OF MDOE ACROSS DIFFERENT CARDINALITY

# conf_dict = {}  # cardi -> mode
# print "One way ANOVA of confidence across modes for different cardi"
# for wordNum in wordNums: 
# 	conf_dict[wordNum]={}
# 	for mode in modes:
# 		conf_dict[wordNum][mode] = [int(a['conf']) for a in answers if a['mode']==mode and a['wordNum']==wordNum]
# 	data = [v for k,v in conf_dict[wordNum].iteritems()]
# 	print wordNum + ":" + str(stats.f_oneway(data[0],data[1],data[2],data[3])) + str(EffectSize(data[0],data[1],data[2],data[3]))
# # print duration_dict

# exit()
#####


# plt.figure(1,facecolor="white")
# count=0
# all_values = []
# for i in range(len(modes)):
# 	for j in range(len(wordNums)):
# 		count+=1
# 		mode = modes[i]
# 		wordNum = wordNums[j]
# 		values = [int(a['conf']) for a in answers if a['mode']==mode and a['wordNum']==wordNum]
# 		all_values.append(values)
# 		plt.subplot(len(modes),len(wordNums),count)
# 		if count==1 or count==4 or count==7 or count==10:
# 			plt.ylabel(mode)   
# 		if count<10:
# 			plt.gca().get_xaxis().set_visible(False)
# 		else:
# 			plt.xlabel("Confidence: "+wordNum+"-word settings")  
# 		print "%20s: \t#:%d, \tAvg:%f, \tMedian:%f, \tStdev:%f, \tmin:%d , \tmax:%d" % (mode+" "+str(wordNum), len(values), np.mean(values), np.median(values), np.std(values), min(values), max(values))
# 		n, bins, patches = plt.hist(values, bins=range(0, 30, 1))
# 		plt.axis([0,5,0,200])
# 		plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)	
# plt.gcf().suptitle("Self-Reported Confidence")				
# plt.show()

# fig = plt.figure(2,facecolor="white")
# ax = fig.add_subplot(111)
# means = {}
# stdev = {}
# for mode in modes:
# 	means[mode]=[]
# 	stdev[mode]=[]
# for mode in modes:
# 	for wordNum in wordNums:
# 		conf_values = [int(a['conf']) for a in answers if a['mode']==mode and a['wordNum']==wordNum]		
# 		means[mode].append(np.mean(conf_values))
# 		stdev[mode].append(np.std(conf_values))

# ind = np.arange(3)
# width=0.15

# rects1 = ax.bar(ind, means[modes[0]], width, color='red', 
# 	yerr=stdev[modes[0]], error_kw=dict(elinewidth=2,ecolor='black'))
# rects2 = ax.bar(ind+width, means[modes[1]], width, color='blue', 
# 	yerr=stdev[modes[1]], error_kw=dict(elinewidth=2,ecolor='black'))
# rects3 = ax.bar(ind+(width*2), means[modes[2]], width, color='green', 
# 	yerr=stdev[modes[2]], error_kw=dict(elinewidth=2,ecolor='black'))
# rects4 = ax.bar(ind+(width*3), means[modes[3]], width, color='gray', 
# 	yerr=stdev[modes[3]], error_kw=dict(elinewidth=2,ecolor='black'))
# ax.set_xlim(-width,len(ind)+width)
# ax.set_ylim(0,5)
# ax.set_ylabel('Confidence')
# ax.set_title('Confidence by visualization and # of words')
# xTickMarks = [wn+" words" for wn in wordNums]
# ax.set_xticks(ind+width)
# xtickNames = ax.set_xticklabels(xTickMarks)
# plt.setp(xtickNames)

# ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), modes, loc=4)



# plt.show()

# plt.figure(6,facecolor="white")
# plt.boxplot(all_values)
# plt.ylim([0,5])
# plt.show()


# ANALYSIS 4. CONTENT ANALYSIS
# topicIdx = 39

# for wordNum in wordNums:
# 	for mode in modes:
# 		print "--------%s----" % mode+" "+str(wordNum)
# 		if mode!="topic-in-a-box":
# 			values = [a['long']+"("+a['conf']+") "+a['duration'] for a in answers if a['mode']==mode and a['wordNum']==wordNum and int(a['topicIdx'])==topicIdx]
# 		else:
# 			values = [a['long']+"("+a['conf']+") "+a['duration'] for a in answers if a['mode']==mode and a['wordNum']==wordNum and int(a['topicIdx'])==topicIdx+1]
# 		pp.pprint(values[:10])



# ANALYSIS 5. SHARED TERMS
# do word size and bar length affect choice of words? 
# ? 20 word list, histogram, and topicbox tend to lose focus 
# ? Do people use bigram chains in topicbox


# ANALYSIS 5a. SHORT DESCRIPTIONS
# stemmer = snowball.EnglishStemmer()
# all_data = {}
# simplyAll = []
# shared_terms_by_wordNum = {"5":[], "10":[], "20":[]}
# for topicIdx in range(50):
# 	# print "\n\n=============== TOPIC "+str(topicIdx)+" =========================="
# 	topicTerms_raw = [terms['first'] for terms in topicJSON['topics'][str(topicIdx)]['terms']]
# 	topicTerms = [stemmer.stem(t) for t in topicTerms_raw]
# 	for mode in modes:
# 		for wordNum in wordNums:
# 			if mode not in all_data:  all_data[mode]={}
# 			if wordNum not in all_data[mode]: all_data[mode][wordNum]=[]

# 			# print "--------%s----" % mode+" "+str(wordNum)
# 			if mode!="topic-in-a-box":
# 				records = [a for a in answers if a['mode']==mode and a['wordNum']==wordNum and int(a['topicIdx'])==topicIdx]
# 			else:
# 				if topicIdx==49:
# 					records = [a for a in answers if a['mode']==mode and a['wordNum']==wordNum and int(a['topicIdx'])==0]
# 				else:
# 					records = [a for a in answers if a['mode']==mode and a['wordNum']==wordNum and int(a['topicIdx'])==topicIdx+1]
# 			# now analyze terms in the records
# 			shorts_raw = [r['short'] for r in records] 
# 			shorts = [re.sub('[^0-9a-zA-Z ]+', '', r['short']).split(" ") for r in records] 
# 			shorts = [[stemmer.stem(s.lower()) for s in sl if len(s)>0] for sl in shorts] # removing empty tokens and lowercase
# 			# print "SHORT DESCRIPTIONS: "+str(shorts_raw)
# 			# print "STEMMED DESCRIPTIONS: "+str(shorts)
# 			# print "TOPIC TERMS: "+ str(topicTerms_raw[:int(wordNum)])
# 			# print "STEMMED TOPIC TERMS: "+ str(topicTerms[:int(wordNum)])
# 			terms_taken_from_topic = [[s for s in sl if s in topicTerms[:int(wordNum)]] for sl in shorts]
# 			shared_terms_by_wordNum[wordNum].append(len(flatten(terms_taken_from_topic)))
# 			# print "SHARED: " + str(terms_taken_from_topic)
# 			if len(terms_taken_from_topic)>0:
# 				avg_num_words_from_topic = sum([len(s) for s in terms_taken_from_topic]) / (len(terms_taken_from_topic) *1.0)
# 				# print avg_num_words_from_topic
# 				all_data[mode][wordNum].append(avg_num_words_from_topic)
# 				shared_terms_by_wordNum[wordNum].append(avg_num_words_from_topic)
# 				simplyAll.append(avg_num_words_from_topic)
# 			longs = [r['long'] for r in records] 

# # data = simplyAll
# # print "# of SHARED TERMS:   Avg.:%f, STD:%f, Median:%f, Max:%f, Min:%f"%(np.mean(data), np.std(data), np.median(data), np.max(data), np.min(data))

# # pp.pprint(shared_terms_by_wordNum)

# for wordNum in wordNums:
# 	for mode in modes:
# 		data =all_data[mode][wordNum]
# 		print "%s:%s,  avg=%f, sd=%f" % (wordNum, mode, np.mean(data), np.std(data) )
# 		print len(data)


##### ANOTHER TRIAL OF SHARED WORDS : RANK CORRELATION ANALYSIS








# ##### EXPORT SHARED WORDS DATA FOR R: twoway anova repeated-measure
# with open('csv_backup_5/shared_terms_short.csv', 'w') as fp:
# 	writer = csv.writer(fp, delimiter=',')
# 	writer.writerow(['topicIdx','visualization','cardinality','avgSharedTerms'])
# 	for wordNum in wordNums:
# 		for mode in modes:
# 			data = all_data[mode][wordNum]
# 			print data
# 			for i,d in enumerate(data): 
# 				writer.writerow([i,mode,wordNum,d])
# exit()
# END OF EXPORTING

# print shared_terms_by_wordNum["5"]
# print shared_terms_by_wordNum["10"]
# print shared_terms_by_wordNum["20"]


# exit()
# # DRAW BAR CHART OF WORD->MODE
# fig = plt.figure(2,facecolor="white")
# ax = fig.add_subplot(111)
# all_data_avg = [[np.mean(vk) for kk,vk in v.iteritems()] for k,v in all_data.iteritems()]
# all_data_stdev = [[np.std(vk) for kk,vk in v.iteritems()] for k,v in all_data.iteritems()]
# print all_data_avg
# print all_data_stdev
# ind = np.arange(3)
# width=0.15
# rects1 = ax.bar(ind, all_data_avg[0], width, color='red', 
# 	yerr=all_data_stdev[0], error_kw=dict(elinewidth=2,ecolor='black'))
# rects2 = ax.bar(ind+width, all_data_avg[1], width, color='blue', 
# 	yerr=all_data_stdev[1], error_kw=dict(elinewidth=2,ecolor='black'))
# rects3 = ax.bar(ind+(width*2), all_data_avg[2], width, color='green', 
# 	yerr=all_data_stdev[2], error_kw=dict(elinewidth=2,ecolor='black'))
# rects4 = ax.bar(ind+(width*3), all_data_avg[3], width, color='gray', 
# 	yerr=all_data_stdev[3], error_kw=dict(elinewidth=2,ecolor='black'))

# ax.set_xlim(-width,len(ind)+width)
# ax.set_ylim(0,2.5)
# ax.set_ylabel('Avg. number of shared terms ')
# ax.set_title('# of Terms in Short Description taken from topics')
# xTickMarks = [wn+" words" for wn in wordNums]
# ax.set_xticks(ind+width)
# xtickNames = ax.set_xticklabels(xTickMarks)
# plt.setp(xtickNames)

# ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), modes, loc=1)
# plt.show()


# ANALYSIS 5b. LONG DESCRIPTIONS
# stemmer = snowball.EnglishStemmer()
# all_data = {}
# simplyAll = []
# shared_terms_by_wordNum = {"5":[], "10":[], "20":[]}
# for topicIdx in range(50):
# 	# print "\n\n=============== TOPIC "+str(topicIdx)+" =========================="
# 	topicTerms_raw = [terms['first'] for terms in topicJSON['topics'][str(topicIdx)]['terms']]
# 	topicTerms = [stemmer.stem(t) for t in topicTerms_raw]
# 	for mode in modes:
# 		for wordNum in wordNums:
# 			if mode not in all_data:  all_data[mode]={}
# 			if wordNum not in all_data[mode]: all_data[mode][wordNum]=[]
			
# 			if mode!="topic-in-a-box":
# 				records = [a for a in answers if a['mode']==mode and a['wordNum']==wordNum and int(a['topicIdx'])==topicIdx]
# 			else:
# 				if topicIdx==49:
# 					records = [a for a in answers if a['mode']==mode and a['wordNum']==wordNum and int(a['topicIdx'])==0]
# 				else:
# 					records = [a for a in answers if a['mode']==mode and a['wordNum']==wordNum and int(a['topicIdx'])==topicIdx+1]
# 			# now analyze terms in the records
# 			longs_raw = [r['long'] for r in records] 
# 			longs = [re.sub('[^0-9a-zA-Z ]+', '', r['long']).split(" ") for r in records] 
# 			longs = [[stemmer.stem(s.lower()) for s in sl if len(s)>0] for sl in longs] # removing empty tokens and lowercase
# 			# print "STEMMED DESCRIPTIONS: "+str(longs)
# 			# print "TOPIC TERMS: "+ str(topicTerms_raw[:int(wordNum)])
# 			# print "STEMMED TOPIC TERMS: "+ str(topicTerms[:int(wordNum)])
# 			terms_taken_from_topic = [[s for s in sl if s in topicTerms[:int(wordNum)]] for sl in longs]
# 			simplyAll += [len(t) for t in terms_taken_from_topic]
# 			if len(terms_taken_from_topic)>0:
# 				avg_num_words_from_topic = sum([len(s) for s in terms_taken_from_topic]) / (len(terms_taken_from_topic) *1.0)
				
# 				all_data[mode][wordNum].append(avg_num_words_from_topic)
# 				# portions = [len(shared_terms)/(len(longs[i])*1.0) for i,shared_terms in enumerate(terms_taken_from_topic)]
# 				# print portions
# 				# all_data[mode][wordNum].append(np.mean(portions))
# 				simplyAll.append(avg_num_words_from_topic)
# 			# longs = [r['long'] for r in records] 
# 			if avg_num_words_from_topic<4: 
# 				continue
# 			print "--------%s----" % mode+" "+str(wordNum)
# 			print "LONG DESCRIPTIONS: "+str(longs_raw)
# 			print "TOPIC TERMS: " + str(topicTerms[:int(wordNum)])
# 			print "SHARED TERMS: "+str(terms_taken_from_topic)
# 			print [len(t) for t in terms_taken_from_topic]
# 			print avg_num_words_from_topic

# data = simplyAll
# print "# of SHARED TERMS:   Avg.:%f, STD:%f, Median:%f, Max:%f, Min:%f"%(np.mean(data), np.std(data), np.median(data), np.max(data), np.min(data))

# print "AVERAGE NUMBer OF SHARED TERMS FOR SENTENCE "
# for mode in modes:
# 	for wordNum in wordNums:
# 		data =all_data[mode][wordNum]
# 		print "%s:%s,  avg=%f, sd=%f" % (wordNum, mode, np.mean(data), np.std(data) )


# # print all_data
# exit() 

# ##### EXPORT SHARED WORDS DATA FOR R: twoway anova repeated-measure
# with open('csv_backup_5/shared_terms_long.csv', 'w') as fp:
# 	writer = csv.writer(fp, delimiter=',')
# 	writer.writerow(['topicIdx','visualization','cardinality','avgSharedTerms'])
	
# 	for wordNum in wordNums:
# 		for mode in modes:
# 			data = all_data[mode][wordNum]
# 			print mode, wordNum, str(data)
# 			print len(data)
# 			print np.mean(data)
# 			for i,d in enumerate(data): 
# 				writer.writerow([i,mode,wordNum,d])
# exit()
# # END OF EXPORTING

# exit()

# # DRAW BAR CHART OF WORD->MODE
# fig = plt.figure(2,facecolor="white")
# ax = fig.add_subplot(111)
# all_data_avg = [[np.mean(vk) for kk,vk in v.iteritems()] for k,v in all_data.iteritems()]
# all_data_stdev = [[np.std(vk) for kk,vk in v.iteritems()] for k,v in all_data.iteritems()]
# print all_data_avg
# print all_data_stdev
# ind = np.arange(3)
# width=0.15
# rects1 = ax.bar(ind, all_data_avg[0], width, color='red', 
# 	yerr=all_data_stdev[0], error_kw=dict(elinewidth=2,ecolor='black'))
# rects2 = ax.bar(ind+width, all_data_avg[1], width, color='blue', 
# 	yerr=all_data_stdev[1], error_kw=dict(elinewidth=2,ecolor='black'))
# rects3 = ax.bar(ind+(width*2), all_data_avg[2], width, color='green', 
# 	yerr=all_data_stdev[2], error_kw=dict(elinewidth=2,ecolor='black'))
# rects4 = ax.bar(ind+(width*3), all_data_avg[3], width, color='gray', 
# 	yerr=all_data_stdev[3], error_kw=dict(elinewidth=2,ecolor='black'))

# ax.set_xlim(-width,len(ind)+width)
# ax.set_ylim(0,0.65)
# ax.set_ylabel('Avg. portions of shared terms ')
# ax.set_title('Portion of Terms in Long Description taken from topics')
# xTickMarks = [wn+" words" for wn in wordNums]
# ax.set_xticks(ind+width)
# xtickNames = ax.set_xticklabels(xTickMarks)
# plt.setp(xtickNames)

# ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), modes, loc=1)
# plt.show()




# ANALYSIS 5c. FURTHER ANALYSIS OF SHARED TERMS
stemmer = snowball.EnglishStemmer()

# ## EXPORTING UNIQUE WORDS USED IN LABELS
# original_words = {}   # topicIdx->mode->wordNum->cardinality->[]
# for tidx in range(50):
# 	topicIdx = str(tidx)
# 	original_words[topicIdx]={}
# 	for mode in modes:
# 		original_words[topicIdx][mode]={}
# 		for wordNum in wordNums:
# 			original_words[topicIdx][mode][wordNum]={}
# 			original_words[topicIdx][mode][wordNum]['numLabels']=0
# 			for cardi in cardinality:
# 				original_words[topicIdx][mode][wordNum][cardi]=[]

# for ans in answers:
# 	print "-----------" 
# 	topicIdx = ans['topicIdx']
# 	wordNum = ans['wordNum']
# 	# topic-in-a-box has 1-topic off the others. should fix it. 
# 	if ans['mode']=="topic-in-a-box":
# 		if topicIdx=="0":
# 			topicIdx = "49"
# 		else:
# 			topicIdx = str(int(topicIdx)-1)
# 	topicTerms = [term['first'] for term in topicJSON['topics'][topicIdx]['terms']][:int(wordNum)]
# 	topicTerms_stemmed = [stemmer.stem(t) for t in topicTerms]
# 	print topicTerms
# 	print topicTerms_stemmed
# 	# print ans
# 	short_raw = ans['short']
# 	shorts = re.sub('[^0-9a-zA-Z ]+', '', short_raw).split() 
# 	shorts_original = [s for s in shorts if stemmer.stem(s.lower()) not in topicTerms_stemmed]

# 	long_raw = ans['long']
# 	longs = re.sub('[^0-9a-zA-Z ]+', '', long_raw).split() 
# 	longs_original = [s for s in longs if stemmer.stem(s.lower()) not in topicTerms_stemmed]
	
# 	original_words[topicIdx][ans['mode']][ans['wordNum']]["numLabels"] += 1
# 	original_words[topicIdx][ans['mode']][ans['wordNum']]["short"] += shorts_original
# 	original_words[topicIdx][ans['mode']][ans['wordNum']]["long"] += longs_original
# 	original_words[topicIdx][ans['mode']][ans['wordNum']]["topic_terms"] = topicTerms
# 	original_words[topicIdx][ans['mode']][ans['wordNum']]["topic_terms_stemmed"] = topicTerms_stemmed

# 	pp.pprint(original_words[topicIdx][ans['mode']][ans['wordNum']])

# with open('csv_backup_5/original_terms.json','w') as f:
# 	f.write(json.dumps(original_words, indent=4))

# exit()


### EXPORTING TOPIC TERMS WITH THEIR FREQUENCY OF BEING SHARED
# all_data = {}
# for topicIdx in range(0,50):
# 	print "\n\n=============== TOPIC "+str(topicIdx)+" =========================="
# 	topicTerms_raw = topicJSON['topics'][str(topicIdx)]['terms']
# 	topicTerms = {}
# 	for term in topicTerms_raw:
# 		tStr = term['first']
# 		prob = term['second']
# 		if tStr not in topicTerms: 
# 			topicTerms[tStr] = prob
# 		else:
# 			topicTerms[tStr] += prob
# 	for mode in modes:
# 		for wordNum in wordNums:
# 			if topicIdx not in all_data:  all_data[topicIdx]={}
# 			if mode not in all_data[topicIdx]:  all_data[topicIdx][mode]={}
# 			# if wordNum not in all_data[topicIdx][mode]: all_data[topicIdx][mode][wordNum]={}

# 			print "--------%s----" % mode+" "+str(wordNum)
# 			if mode!="topic-in-a-box":
# 				records = [a for a in answers if a['mode']==mode and a['wordNum']==wordNum and int(a['topicIdx'])==topicIdx]
# 			else:
# 				if topicIdx<48:
# 					fixed_topicIdx = topicIdx+1 
# 				else:
# 					fixed_topicIdx = 0 
# 				records = [a for a in answers if a['mode']==mode and a['wordNum']==wordNum and int(a['topicIdx'])==fixed_topicIdx]
# 			print "STEMMED TOPIC TERMS: "+ str([term for term, data in sorted(topicTerms.iteritems(),key=lambda k: k[1], reverse=True)[:int(wordNum)]])
# 			# now analyze terms in the records
# 			shorts_raw = [r['short'] for r in records] 
# 			shorts = [re.sub('[^0-9a-zA-Z ]+', '', r['short']).split(" ") for r in records] 
# 			shorts = [[stemmer.stem(s.lower()) for s in sl if len(s)>0] for sl in shorts] # removing empty tokens and lowercase
# 			print "SHORT DESCRIPTIONS: "+str(shorts_raw)
# 			print "STEMMED DESCRIPTIONS: "+str(shorts)
# 			print "TOPIC TERMS: "+ str(topicTerms_raw[:int(wordNum)])
# 			print "STEMMED TOPIC TERMS: "+ str([term for term, data in sorted(topicTerms.iteritems(),key=lambda k: k[1], reverse=True)[:int(wordNum)]])
			
# 			longs_raw = [r['long'] for r in records] 
# 			longs = [re.sub('[^0-9a-zA-Z ]+', '', r['long']).split(" ") for r in records] 
# 			longs = [[stemmer.stem(s.lower()) for s in sl if len(s)>0] for sl in longs] # removing empty tokens and lowercase
# 			print "LONG DESCRIPTIONS: "+str(longs_raw)
# 			print "STEMMED DESCRIPTIONS: "+str(longs)
# 			print "TOPIC TERMS: "+ str(topicTerms_raw[:int(wordNum)])
# 			print "STEMMED TOPIC TERMS: "+ str([term for term, data in sorted(topicTerms.iteritems(),key=lambda k: k[1], reverse=True)[:int(wordNum)]])

# 			bag = {term:{'prob':prob, 'freq_short':0, 'freq_long':0} for term, prob in topicTerms.iteritems()} 
# 			for sl in shorts:  # FOR EACH SHORT DESCRIPTIION
# 				for s in sl:	# FOR EACH STEM IN SHORT
# 					if s in bag:   # IF THE STEM IS ONE OF THE TOPIC TERM
# 						bag[s]['freq_short'] += 1	# COUNT+1
# 			for sl in longs:
# 				for s in sl:
# 					if s in bag:   
# 						bag[s]['freq_long'] += 1
# 			# NORMALIZED FREQUENCY OF SHARED TERMS BY # OF LABELS
# 			for s in bag.keys():
# 				bag[s]['freq_short_normalized'] = float(bag[s]['freq_short']) / float(len(shorts_raw)) 
# 				bag[s]['freq_long_normalized'] = float(bag[s]['freq_long']) / float(len(longs_raw)) 
# 			pp.pprint(sorted(bag.iteritems(), key=lambda k:k[1]['prob'], reverse=True))
# 			all_data[topicIdx][mode][wordNum]=bag
# 			# terms_taken_from_topic = [[s for s in sl if s in topicTerms[:int(wordNum)]] for sl in shorts]
# 			# print terms_taken_from_topic
# 			# if len(terms_taken_from_topic)>0:
# 			# 	avg_num_words_from_topic = sum([len(s) for s in terms_taken_from_topic]) / (len(terms_taken_from_topic) *1.0)
# 			# 	print avg_num_words_from_topic
# 			# 	all_data[mode][wordNum].append(avg_num_words_from_topic)
# 			# longs = [r['long'] for r in records] 

# # pp.pprint(all_data)
# exit()

# with open('csv_backup_5/topic_terms_with_frequency_normalized.json','w') as f:
# 	f.write(json.dumps(all_data, indent=4))

# exit()


# # with open('csv_backup_5/topic_terms_with_frequency.json','w') as f:
# # 	f.write(json.dumps(all_data, indent=4))


### BASIC STATISTICS OF SHARED TERM FREQUENCY
# f = open("csv_backup_5/topic_terms_with_frequency.json","r")
# shared_term_dict = json.loads(f.read())

# freq_short_list = []
# freq_long_list = []

# for topicIdx in range(50):
# 	for mode in modes:
# 		for wordNum in wordNums:
# 			terms = shared_term_dict[str(topicIdx)][mode][wordNum]
# 			for word,freq_info in terms.iteritems():
# 				freq_short_list.append(freq_info['freq_short'])
# 				freq_long_list.append(freq_info['freq_long'])

# data = freq_short_list
# print "SHORT SHARE TERMS:   Avg.:%f, STD:%f, Median:%f, Max:%f, Min:%f"%(np.mean(data), np.std(data), np.median(data), np.max(data), np.min(data))
# data = freq_long_list
# print "LONG SHARE TERMS:   Avg.:%f, STD:%f, Median:%f, Max:%f, Min:%f"%(np.mean(data), np.std(data), np.median(data), np.max(data), np.min(data))


# pp.pprint(all_data)
# fig = plt.figure(2,facecolor="white")
# ax = fig.add_subplot(111)
# prob = []
# freq = []
# for mode in modes:
# 	for wordNum in wordNums:		
# 		for topicIdx in range(50):
# 			bag = all_data[topicIdx][mode][wordNum]
# 			for term, data in bag.iteritems():
# 				prob.append(data['prob'])
# 				freq.append(data['freq']+random.uniform(-0.15,0.15))


# ax.scatter(prob,freq, s=2, marker="+")
# ax.set_title("Terms probability and Frequency of Being Used in Short Descriptions ")
# ax.set_ylabel("Frequency")   
# ax.set_xlabel("Term Probability")   
# ax.text(0.95, 0.95, "correlation = "+ str(np.corrcoef(prob,freq)[0][1]), ha='right', va='top', transform=ax.transAxes)
# plt.show()
# # DRAW PROB,FREQ scatterplot for different settings
# fig = plt.figure(3,facecolor="white")
# count = 1
# for mode in modes:
# 	for wordNum in wordNums:
# 		# COLLECT PROB,FREQ INFO FROM ALL TOPICS
# 		prob = []
# 		freq = []
# 		for topicIdx in range(50):
# 			bag = all_data[topicIdx][mode][wordNum]
# 			for term, data in bag.iteritems():
# 				prob.append(data['prob'])
# 				freq.append(data['freq']+random.uniform(-0.15,0.15))
# 		# LETS DRAW SCATTERPLOT
# 		ax = fig.add_subplot(len(modes),len(wordNums),count)
# 		ax.tick_params(axis='both',which='major', labelsize=8)
# 		ax.tick_params(axis='both',which='minor', labelsize=7)
# 		ax.set_ylim(0,8)
# 		ax.scatter(prob,freq, s=3, marker='+')
# 		ax.text(0.98, 0.98, str(np.corrcoef(prob,freq)[0][1])[:8], ha='right', va='top', transform=ax.transAxes)
# 		if count==1 or count==4 or count==7 or count==10:
# 			ax.set_ylabel(mode)   
# 		if count<10:
# 			ax.get_xaxis().set_visible(False)
# 		else:
# 			ax.set_xlabel(wordNum + " words")   
# 		count += 1
# plt.show()
 

# DRAW BAR CHART OF WORD->MODE
# fig = plt.figure(2,facecolor="white")
# ax = fig.add_subplot(111)
# all_data_avg = [[np.mean(vk) for kk,vk in v.iteritems()] for k,v in all_data.iteritems()]
# all_data_stdev = [[np.std(vk) for kk,vk in v.iteritems()] for k,v in all_data.iteritems()]
# print all_data_avg
# print all_data_stdev
# ind = np.arange(3)
# width=0.15
# rects1 = ax.bar(ind, all_data_avg[0], width, color='red', 
# 	yerr=all_data_stdev[0], error_kw=dict(elinewidth=2,ecolor='black'))
# rects2 = ax.bar(ind+width, all_data_avg[1], width, color='blue', 
# 	yerr=all_data_stdev[1], error_kw=dict(elinewidth=2,ecolor='black'))
# rects3 = ax.bar(ind+(width*2), all_data_avg[2], width, color='green', 
# 	yerr=all_data_stdev[2], error_kw=dict(elinewidth=2,ecolor='black'))
# rects4 = ax.bar(ind+(width*3), all_data_avg[3], width, color='gray', 
# 	yerr=all_data_stdev[3], error_kw=dict(elinewidth=2,ecolor='black'))

# ax.set_xlim(-width,len(ind)+width)
# ax.set_ylim(0,2.5)
# ax.set_ylabel('Avg. number of shared terms ')
# ax.set_title('# of Terms in Short Description taken from topics')
# xTickMarks = [wn+" words" for wn in wordNums]
# ax.set_xticks(ind+width)
# xtickNames = ax.set_xticklabels(xTickMarks)
# plt.setp(xtickNames)

# ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), modes, loc=1)
# plt.show()

##### TOPIC INCLUSION:  DRAW LINE GRAPH BY RANKING ~ FREQUENCY #####  
f = open("csv_backup_5/topic_terms_with_frequency_normalized.json","r")
shared_term_dict = json.loads(f.read())

all = [["topicIdx","mode","wordNum","rank","freq_normalized","shortOrLong"]]
for topicIdx, d in shared_term_dict.iteritems():
	for mode, dd in d.iteritems():
		for wordNum, ddd in dd.iteritems():
			# print "---------"
			sorted_by_prob = sorted(ddd.items(), key=lambda (k,v): v['prob'], reverse=True)[:int(wordNum)]
			for i, s in enumerate(sorted_by_prob):  
				all.append([topicIdx, mode, wordNum, i, s[0], s[1]['prob'], s[1]['freq_short_normalized'], "short"])
				all.append([topicIdx, mode, wordNum, i, s[0], s[1]['prob'], s[1]['freq_long_normalized'], "long"])
# pp.pprint([a for a in all if a[0]=="0" and a[1]=="word" and a[2]=="5"])
# exit()

### Exporting intermediate data for Leah's sanity check
# with open('csv_backup_5/rank_inclusion.csv','w') as f:
# 	writer = csv.writer(f, delimiter=',')
# 	for line in all:
# 		writer.writerow([unicode(s).encode("utf-8") for s in line])

list_raw = []
all_freq_list = []
for shortOrLong in ["short","long"]:
	# print shortOrLong + "-----"
	for mode in modes:
		# print "----"+ mode + "-----"
		for wordNum in wordNums:
			freq_list = []
			# EXTRACTING FREQUENCY INFORMATION THAT MATCHES THE CURRENT SETTING
			data = [a for a in all if a[2]==wordNum and a[7]==shortOrLong and a[1]==mode]		
			# pp.pprint(data)
			# exit()
			for rank in range(int(wordNum)):
				list_raw.append([d[6] for d in data if d[3]==rank])
				freq_list.append(np.mean([d[6] for d in data if d[3]==rank]))
			# print freq_list
			all_freq_list.append({'shortOrLong':shortOrLong, "mode":mode, "wordNum":wordNum, 'freq':freq_list})

pp.pprint(all_freq_list)


# ###  REPORTING SIMPLE COMPARISON OF 5-word and 20-word setting 
# for shortOrLong in ["short","long"]:
# 	for mode in modes:
# 		freq5=[a['freq'][0] for a in all_freq_list if (a['mode']==mode and a['wordNum']=='5' and a['shortOrLong']==shortOrLong)][0]
# 		freq20=[a['freq'][0] for a in all_freq_list if (a['mode']==mode and a['wordNum']=='20' and a['shortOrLong']==shortOrLong)][0]
# 		print mode, shortOrLong, freq5, freq20, freq20/freq5


### DRAW LINE CHARTS OF AVERAGE PROBABILITY OF WORDS FOR EACH RANK BEING USED IN SHORT/LONG LABELS 
fig = plt.figure(1,facecolor="white")
count = 0
color={"5":"r","10":"g","20":"b"}

for mode in modes:
	for shortOrLong in ["short","long"]:
		count+=1
		ax = plt.subplot(4,2,count)
		for wordNum in wordNums:
			fl = [flist for flist in all_freq_list if flist['shortOrLong']==shortOrLong and flist['mode']==mode and flist["wordNum"]==wordNum][0]
			plt.plot(range(len(fl['freq'])),fl['freq'], color[wordNum], linewidth=2.0)
			if count%2==0:   
				ax.set_ylim(0,0.35)
				ax.yaxis.set_ticks(np.arange(0,0.35,0.05))
				if count==8:
					ax.set_xlabel("Rank of Topic Words", fontsize=18)
			else:
				ax.set_ylim(0,0.35)
				ax.yaxis.set_ticks(np.arange(0,0.35,0.05))
				ax.set_ylabel(mode, fontsize=20)
				if count==7:
					ax.set_xlabel("Rank of Topic Words", fontsize=18)
				
			
fig.set_size_inches(8, 10.5)			
fig.savefig('linechart_wordrank_prob.png',dpi=250)		
#plt.show()

		

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
# ANALYSIS 6. DOCUMENT ANALYSIS
# topicIdx = 3
# for topicIdx in range(50):
# 	topicTerms = [term['first'].encode("utf8") for term in topicJSON['topics'][str(topicIdx)]['terms']]
# 	doc_id_list = [(doc['first'],doc['second']) for doc in topicJSON['topics'][str(topicIdx)]['docs']]
# 	doc_xml_list = [open(path_to_documents+df[0]).read() for df in doc_id_list]
# 	doc_soup_list = [BeautifulSoup(doc_xml) for doc_xml in doc_xml_list]
# 	titles = [ds.hedline.text.strip().replace("\n","") for ds in doc_soup_list]
# 	leads = [ds.find_all("block",class_="full_text") for ds in doc_soup_list]

# 	print "================================================================="
# 	print "TOPIC "+str(topicIdx) + " --- " + str(topicTerms) + ""
# 	print "================================================================="
# 	# print "TOP-DOCS"+str(doc_id_list)
# 	for i in range(len(doc_xml_list)):
# 		print "["+str(titles[i].encode("utf8")) + "]\t\t(" + str(doc_id_list[i][1]) + ")"
# 		# print "["+str(titles[i].encode("utf8")) + "]"
# 		print "\t"+(leads[i][0].text.encode("utf8").strip().replace("\n","  ")[:500]+"..." if len(leads[i])>0 else "EMPTY")
# 		print ""
# # leads = [lead[0].text for lead in leads]


# for wordNum in wordNums:
# 	for mode in modes:
# 		print "--------%s----" % mode+" "+str(wordNum)
# 		if mode!="topic-in-a-box":
# 			values = [a['long']+"("+a['conf']+") "+a['duration'] for a in answers if a['mode']==mode and a['wordNum']==wordNum and int(a['topicIdx'])==topicIdx]
# 		else:
# 			values = [a['long']+"("+a['conf']+") "+a['duration'] for a in answers if a['mode']==mode and a['wordNum']==wordNum and int(a['topicIdx'])==topicIdx+1]
# 		# pp.pprint(values[:10])




# #######################################################################################################################
# #######################################################################################################################
# #######################################################################################################################
# # ANALYSIS 7. LABELING HIT ANALYSIS

# with open('csv_backup_3/LabelingHit.csv', mode='r') as infile:
#	 reader = csv.DictReader(infile)
#     hits = [rows for rows in reader]

# dict_usercode = {}
# for hit in hits:
# 	uc = hit['usercode']
# 	if uc in dict_usercode:
# 		dict_usercode[uc] += 1
# 	else:
# 		dict_usercode[uc] = 1

# pp.pprint(dict_usercode)
# print len(dict_usercode.keys())
# print len(dict_usercode.keys()) * 5
# print len(answers)













#plt.savefig("path.png")