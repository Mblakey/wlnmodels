# classes to hold model architecture, if run as main, obtain stats about the models
# shape, dimension and parameters 

import sys
import tensorflow as tf

class RNNModel:
	
	input_shape = ()
	def __init__(self, xshape,yshape):
		self.input_shape = (xshape,yshape)


	def build():
		return True



def main():
	print(tf.config.list_physical_devices('GPU'))
	model = RNNModel(10,20)
	return 0

if __name__ == "__main__":
	main() 