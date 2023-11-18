# Prepare data for training and testing, if run as main, obtain stats about the inputted
# data file, else this should be used purely for library functions

import sys
import numpy as np
from wlnparser import WLNParser

file = ""
parser_path = ""
opt_debug = 0


def DisplayUsage():
	sys.stderr.write("python data.py <options> <file> <parser bin>\n")
	sys.stderr.write("options:\n")
	sys.stderr.write("-d | --debug		display debugging to stderr\n")
	sys.stderr.write("-h | --help			display a verbose help menu\n")
	exit(1)

def ProcessCommandLine():
	i = 0
	global file
	global parser_path
	global opt_debug

	for arg in sys.argv[1:]:
		if(arg[0] == '-'):
			if arg == '-h' or arg == '--help':
				sys.stderr.write("help")
			elif arg == '-d' or arg == '--debug':
				opt_debug = 1
			else:
				sys.stderr.write(f"Error: {arg} not recognised as argument\n")
				DisplayUsage()
		elif i == 0:
			file = arg
			i += 1
		elif i == 1:
			parser_path = arg
			i += 1
		else:
			sys.stderr.write(f"Error: unknown input {arg}\n")
			DisplayUsage()

	if file == "":
		sys.stderr.write("Error: no file has been inputted!\n")
		DisplayUsage()

	if parser_path == "":
		sys.stderr.write("Error: parser path has not been inputted!\n")
		DisplayUsage()

	return;


"""
Transform sequence data and perform the one hot encode as preperation
assumes that the data has been prepared for a given task

https://medium.com/analytics-vidhya/one-hot-encoding-of-text-data-in-natural-language-processing-2242fefb2148
"""
class DataLoader:
	filename = ""
	chardict = {}
	max_len = 0

	def __init__(self,file):
		self.filename = file 

	def read_sequences(self,parser):
		open_file = open(self.filename,"r")
		if open_file:
			sequences = parser.filter_sequences(self.filename)
			i = 0

			for line in sequences:
				if len(line.strip()) > self.max_len:
					self.max_len = len(line.strip())

				for ch in line.strip():
					if ch not in self.chardict:
						self.chardict[ch] = i
						i+=1

			if(opt_debug):
				sys.stderr.write(f"data loader: {len(sequences)} lines read\n")
				sys.stderr.write(f"data loader: {self.max_len} max length\n")
				sys.stderr.write(f"data loader: {len(self.chardict)} chars in character set\n")

			open_file.close()
			return sequences
		else: 
			return []

	def direct_encode(self,sequences):
		results  = np.zeros(shape = (len(sequences),
                  			self.max_len,
                         max(self.chardict.values()) + 1))

		for i,wln in enumerate(sequences):
			for j, char in enumerate(wln):

				index = self.chardict[char]
				results[i,j,index] = 1

		sys.stderr.write(f"data loader: one hot encoded shape {results.shape}\n")
		return results


def main():
	dl = DataLoader(file)
	parser = WLNParser(parser_path)

	sequences = dl.read_sequences(parser)
	dl.direct_encode(sequences)

	return 0

if __name__ == "__main__":
	ProcessCommandLine()
	main()