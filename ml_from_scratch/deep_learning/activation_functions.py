import numpy as np

class Sigmoid():
	def __init__(self):
		return

	def __call__(self, x):
		return 1 / (1 + np.exp(-x))

	def gradient(self, x):
		p = self.__call__(x)
		return p * (1 - p)

class Softmax():
	def __call__(self, x):
		e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
		return e_x / np.sum(e_x, axis=-1, keepdims=True)
		#x_norm = x - np.max(x)
		#return np.exp(x_norm) / np.sum(np.exp(x_norm))

	def gradient(self, x):
		p = self.__call__(x)
		return p * (1 - p)

class ReLU():
	def __init__(self):
		return
	
	def __call__(self, x):
		return np.where(x >= 0, x, 0)
	
	def gradient(self, x):
		return np.where(x >= 0, 1, 0)
