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


class DataLoader:
	"""
	Transform sequence data and perform the one hot encode as preperation
	assumes that the data has been prepared for a given task

	https://medium.com/analytics-vidhya/one-hot-encoding-of-text-data-in-natural-language-processing-2242fefb2148
	"""

	filename = ""
	chardict = {}
	inv_chardict = {}
	max_len = 0

	def __init__(self,file):
		self.filename = file 

	def read_sequences(self,parser, filter_rings=False):
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
				sys.stderr.write(f"{len(sequences)} lines read, {self.max_len} max length\n")
				sys.stderr.write(f"{len(self.chardict)} chars in character set\n")

			# create an inverse map for predicition reading 
			self.inv_chardict = {v: k for k,v in self.chardict.items()}
			
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

		if(opt_debug):
			sys.stderr.write(f"one hot encoded shape {results.shape}\n")
		return results



	def split_wln_sequences(self,sequences):
		'''
		takes the wln sequences, splits them into predicting the next char
		'''
		x_sequences = []
		y_sequences = []
		for wln_string in sequences:
			for i in range(len(wln_string),0,-1):
				x_sequences.append(wln_string[:i])

				if wln_string[i:i+1] == "":
					y_sequences.append("nill")
				else:
					y_sequences.append(wln_string[i:i+1])

		if(opt_debug):
			sys.stderr.write(f"{len(x_sequences)} training sequences\n")

		return (x_sequences,y_sequences)


	def prepare_training_data(self,x_sequences,y_sequences):
		y_data  = np.zeros(shape = (len(y_sequences),
                        max(self.chardict.values()) + 1))
	
		for i,char in enumerate(y_sequences):
			if(char != "nill"):
				index = self.chardict[char]
				y_data[i,index] = 1
				enc_char = self.decode_character(y_data[i])
				
				if char != enc_char:
					sys.stderr.write(f"Error: {char} != {enc_char}\n")
			
		x_data = self.direct_encode(x_sequences)
		return (x_data,y_data)


	def decode_character(self,oh_array):
		for i,val in enumerate(oh_array):
			if(val > 0):
				return self.inv_chardict[i]
		return "nill"


def main():
	dl = DataLoader(file)
	parser = WLNParser(parser_path)

	sequences = dl.read_sequences(parser)
	x_sequences,y_sequences = dl.split_wln_sequences(sequences)
	dl.prepare_training_data(x_sequences,y_sequences)
	return 0

if __name__ == "__main__":
	ProcessCommandLine()
	main()