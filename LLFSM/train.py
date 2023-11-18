
import sys
import numpy as np
import wlnparser
import data

import tensorflow as tf
import sklearn

from wlnparser import WLNParser
from data import DataLoader

file = ""
parser_path = ""
epochs = 5

def DisplayUsage():
	sys.stderr.write("python train.py <options> <file> <parser bin>\n")
	sys.stderr.write("options:\n")
	sys.stderr.write("-d | --debug	display debugging to stderr\n")
	sys.stderr.write("-h | --help	display a verbose help menu\n")
	sys.stderr.write("-e=<int> | --epochs=<int>	set number of epochs, default is 5\n")
	exit(1)


def ProcessCommandLine():
	i = 0
	global file
	global parser_path
	global epochs

	global opt_debug

	for arg in sys.argv[1:]:
		if(arg[0] == '-'):
			if arg == '-h' or arg == '--help':
				sys.stderr.write("help")
			elif arg == '-d' or arg == '--debug':
				opt_debug = 1
				data.opt_debug = 1
			elif arg[0:3] == "-e=":
				epochs = int(arg[3:])
			elif arg[0:9] == "--epochs=":
				epochs = int(arg[9:])
			else:
				sys.stderr.write(f"Error: {arg} not recognised as arguement\n")
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


if __name__ == "__main__":
	ProcessCommandLine()
	
	parser = WLNParser(parser_path)
	loader = DataLoader(file)

	sequences = loader.read_sequences(parser)
	x_sequences,y_sequences = loader.split_wln_sequences(sequences)
	x_array,y_array = loader.prepare_training_data(x_sequences,y_sequences)

