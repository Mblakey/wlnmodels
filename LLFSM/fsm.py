# Finite state python loader in python, places inside the prediciton loop for any model
# generating WLN notatation. We can recalculate or stop on a valid point if nothing valid
# is possible. 


class FSMNode:
	id = 0
	jumptable = {}

	def __init__(self,id):
		self.id = id 


class WLNDFA:
	nodes = []

	def __init__(self):
		return True

	def read_dot(self,filename):
		'''
		reads the dumped dot file from wlngrep finite state machine,
		builds the python version with a jump table rather than pointers
		'''
