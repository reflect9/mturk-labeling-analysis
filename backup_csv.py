from subprocess import call

path_csv = "csv_backup_5"
tables_to_backup = ["Evaluation", "LazyTurker"]



loader = "bulkloader.py"
template = " --download --url http://findingtopic.appspot.com/_ah/remote_api --config_file generated_bulkloader3.yaml --kind %s --filename %s/%s.csv"

for table in tables_to_backup:
	options = (template % (table,path_csv,table))
	print options
	# call("bulkloader.py")
	call(loader + " "+ options, shell=True)
