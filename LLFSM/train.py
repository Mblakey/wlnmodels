
import sys

def DisplayUsage():
	print("python train.py <options> <file>")
	print("options:")
	print("-h | --help			display a verbose help menu")
	exit(1)

def ProcessCommandLine():
	for arg in sys.argv[1:]:
		if arg == '-h' or arg == '--help':
			print("help")
		else:
			print(f"Error: {arg} not recognised as argument")
			DisplayUsage()






if __name__ == "__main__":
	ProcessCommandLine()
	
