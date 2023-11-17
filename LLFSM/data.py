# Prepare data for training and testing, if run as main, obtain stats about the inputted
# data file, else this should be used purely for library functions

import sys


file = ""
opt_debug = 0


def DisplayUsage():
	print("python data.py <options> <file>")
	print("options:")
	print("-h | --help			display a verbose help menu")
	exit(1)

def ProcessCommandLine():
	i = 0
	global file
	for arg in sys.argv[1:]:
		if(arg[0] == '-'):
			if arg == '-h' or arg == '--help':
				print("help")
			else:
				print(f"Error: {arg} not recognised as argument")
				DisplayUsage()
		elif i == 0:
			file = arg
			i += 1
		else:
			print(f"Error: unknown input {arg}")
			DisplayUsage()

	if file == "":
		print("Error: no file has been inputted!")
		DisplayUsage()

	return;


"""
Load and check the file is a sensible data pool, only expected single column
wln data
"""
class FileParser:
	filename = ""
	def __init__(self,file):
		self.filename = file 

"""
Transform sequence data and perform the one hot encode as preperation
assumes that the data has been prepared for a given task
"""
class DataLoader:
	sequences = []
	chardict = {}
	symbols = 0
	max_len = 0

	def __init__(self,string_sequences):
		self.sequences = string_sequences






def main():
	fl = FileParser(file)
	print(fl.filename)
	return 0

if __name__ == "__main__":
	ProcessCommandLine()
	main()