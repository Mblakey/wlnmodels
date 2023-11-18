# classes to hold model architecture, if run as main, obtain stats about the models
# shape, dimension and parameters 

import sys
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GRU,Embedding,Input,Flatten

class RNNModel:
	
	model = 0
	vocab_size = 0
	embed_dim = 64
	max_len = 0

	def __init__(self, vocab_size,max_len):
		self.vocab_size = vocab_size
		self.max_len = max_len

	def create_model(self):
		self.model = Sequential()
		self.model.add(Embedding(self.vocab_size,self.embed_dim,input_length=self.max_len))
		self.model.add(Flatten())
		#self.model.add(GRU(128)) 
		self.model.add(Dense(self.vocab_size,activation="softmax")) 
		self.model.compile(optimizer='adam',loss = "categorical_crossentropy", metrics=['accuracy'])
		return self.model

