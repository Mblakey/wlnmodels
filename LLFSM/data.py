# Prepare data for training and testing, if run as main, obtain stats about the inputted
# data file, else this should be used purely for library functions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


import sys
import numpy as np
import tensorflow as tf

from wlnparser import WLNParser

file = ""
parser_path = ""
opt_debug = 0
opt_remove_rings = 0


def DisplayUsage():
	sys.stderr.write("python data.py <options> <file> <parser bin>\n")
	sys.stderr.write("options:\n")
	sys.stderr.write("-r | --remove-rings		remove cycles from the wln data\n")
	sys.stderr.write("-d | --debug		display debugging to stderr\n")
	sys.stderr.write("-h | --help			display a verbose help menu\n")
	exit(1)

def ProcessCommandLine():
	i = 0
	global file
	global parser_path
	global opt_debug
	global opt_remove_rings

	for arg in sys.argv[1:]:
		if(arg[0] == '-'):
			if arg == '-h' or arg == '--help':
				sys.stderr.write("help")
			elif arg == '-d' or arg == '--debug':
				opt_debug = 1
			elif arg == '-r' or arg == '--remove-rings':
				opt_remove_rings = 1
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
	vocab_size = 0

	def __init__(self,file):
		self.filename = file 

	def read_sequences(self,parser, filter_rings=False):
		
		if(filter_rings):
			sys.stderr.write("removing rings from the wln data\n")

		open_file = open(self.filename,"r")
		if open_file:
			sequences = parser.filter_sequences(self.filename,filter_rings)
			i = 1

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
			self.vocab_size = len(self.chardict)+1
			
			open_file.close()
			return sequences
		else: 
			return []


	def split_wln_sequences(self,sequences):
		'''
		takes the wln sequences, splits them into predicting the next char
		'''
		x_sequences = []
		y_sequences = []
		for wln_string in sequences:
			for i in range(len(wln_string)-1,0,-1): # ensures we always predict a character
				x_sequences.append(wln_string[:i])
				y_sequences.append(wln_string[i:i+1])

		if(opt_debug):
			sys.stderr.write(f"{len(x_sequences)} sequences\n")

		return (x_sequences,y_sequences)


	def encode_sequences(self,sequence_list):	
		encoded_sequence_list = []
		for string in sequence_list:
			encoded = []
			for ch in string:
				encoded.append(self.chardict[ch])
			
			while(len(encoded) < self.max_len):
				encoded.append(0)

			encoded_sequence_list.append(encoded)
	
		return np.array(encoded_sequence_list)


	def encode_categorical(self,sequence_list):
		categorical = []
		for string in sequence_list:
			encoded = [];
			while(len(encoded) < self.vocab_size):
				encoded.append(0)

			if string != "nill":
				encoded[self.chardict[string[0]]] = 1
			
			categorical.append(encoded)

		return np.array(categorical)


def main():
	dl = DataLoader(file)
	parser = WLNParser(parser_path)

	sequences = dl.read_sequences(parser,opt_remove_rings)
	x_sequences,y_sequences = dl.split_wln_sequences(sequences)
	
	x = dl.encode_sequences(x_sequences)
	y = dl.encode_categorical(y_sequences)

	print(x.shape)
	print(y.shape)
	
	return 0

if __name__ == "__main__":
	ProcessCommandLine()
	main()