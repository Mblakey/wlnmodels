
import re
import sys
import subprocess
from subprocess import Popen, PIPE


class WLNParser:
	path = ""

	def __init__(self,parser_path):
		self.path = parser_path

	def filter_sequences(self,inp_file):
		if(self.path == ""):
			sys.stderr.write("path to wln parser <bin> not set\n")
			return 0

		ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
		p = subprocess.run([f"{self.path}/wlngrep","-x",inp_file], capture_output=True, text=True)
		
		sequences = []
		lines = p.stdout.splitlines()
		for line in lines:
			result = ansi_escape.sub('', line)
			sequences.append(result)

		return sequences


	def WLNToSmiles(self):
		return True

	def SmilesToWLN(self):
		return True

