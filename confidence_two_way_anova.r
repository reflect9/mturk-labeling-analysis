setwd("~/Documents/study/ITM/MTurk-findingTopic/analysis")
conf_table = read.csv("csv_backup_5/confidence_table.csv")
int <- aov(conf ~ mode*wordNum)
summary(int)



%%%%%   friedman test
ct_5 = read.csv("csv_backup_5/confidence_table_5.csv")
friedman.test(conf ~ index | mode, data=ct_5)

