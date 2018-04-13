import numpy as np
import scipy.special
import scipy.misc

class NeuralNetwork:

	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, activate, batch_size, optimize_algorithm):
		self.inodes = input_nodes
		self.onodes = output_nodes
		self.hidden_nodes = hidden_nodes
		self.lr = learning_rate
		self.batch_size = batch_size
		self.theta = []
		self.b = []
		self.optimize_algorithm = optimize_algorithm

		for i in range(0,len(hidden_nodes)+1):
			if i == 0:
				self.theta.append(np.random.randn(self.hidden_nodes[i], self.inodes).astype(np.float32) * np.sqrt(2.0/(self.inodes)))
				self.b.append(np.random.randn(self.hidden_nodes[i], self.batch_size).astype(np.float32))
			elif i == len(hidden_nodes):
				self.theta.append(np.random.randn(self.onodes, self.hidden_nodes[i-1]).astype(np.float32) * np.sqrt(2.0/(self.hidden_nodes[i-1])))
				self.b.append(np.random.randn(self.onodes, self.batch_size).astype(np.float32))
			else:
				self.theta.append(np.random.randn(self.hidden_nodes[i], self.hidden_nodes[i-1]).astype(np.float32) * np.sqrt(2.0/(self.hidden_nodes[i])))
				self.b.append(np.random.randn(self.hidden_nodes[i], self.batch_size).astype(np.float32))
		
		self.num_layers = len(hidden_nodes) + 1
		self.activate = activate
		self.Beta1 = 0.9
		self.Beta2 = 0.999
		self.mt = []
		self.v = []
		self.gamma = 0.9
		self.cache = []
		for i in range(0,2*self.num_layers):
			self.mt.append(0.0)
			self.cache.append(0.0)
			self.v.append(0.0)
		self.ephsilon = 10 ** -8
		self.t = 0
		self.alpha = 0.001

	def sigmoid_activation(self, x):
		return scipy.special.expit(x)

	def tanh_activation(self, x):
		return np.tanh(x)

	def relu_activation(self, x):
		result = x * (x>0)
		return result	

	def softmax_activation(self, x):
		e_x = np.exp(x - np.max(x))
		return e_x/e_x.sum(axis=0)

	def relu_backprop(self,x):
		return 1.0 * (x > 0)

	def sigmoid_backprop(self, x):
		return x * (1.0 - x)

	def tanh_backprop(self, x):
		return (1 - x**2)

	def activation(self, x):
		if self.activate == 'relu':
			return self.relu_activation(x)
		elif self.activate == 'sigmoid':
			return self.sigmoid_activation(x)
		else:
			return self.tanh_activation(x)
	
	def backprop(self, x):
		if self.activate == 'relu':
			return self.relu_backprop(x)
		elif self.activate == 'sigmoid':
			return self.sigmoid_backprop(x)
		else:
			return self.tanh_backprop(x)

	def adam_optimization(self, theta, derivative, i):
		self.mt[i] = self.Beta1 * self.mt[i] + (1.0 - self.Beta1) * derivative
		self.v[i] = self.Beta2 * self.v[i] + (1.0 - self.Beta2) * (derivative**2)
		new_mt = self.mt[i]/(1.0 - self.Beta1 ** self.t)
		new_v = self.v[i]/(1.0 - self.Beta2 ** self.t)
		new_theta = theta + self.alpha * new_mt/(np.sqrt(new_v) + self.ephsilon)
		return new_theta

	def sgd_optimization(self, theta, derivative):
		new_theta = theta + derivative
		return new_theta

	def rmsprop_optimization(self, theta, derivative, i):
		self.cache[i] = self.gamma * self.cache[i] + (1.0 - self.gamma) * (derivative ** 2)
		theta = theta - self.alpha * derivative / (np.sqrt(self.cache[i]) + self.ephsilon)
		return theta 

	def forward_prop(self, x):
		z = [0 for _ in range(self.num_layers)]
		a = [0 for _ in range(self.num_layers)]

		for i in range(0, self.num_layers):
			if i == 0:
				z[i] = np.dot(self.theta[i], x) + self.b[i]
			else:
				z[i] = np.dot(self.theta[i], a[i-1]) + self.b[i]
			a[i] = self.activation(z[i])

		return a

	def optimizer(self, theta, derivative, i):
		if self.optimize_algorithm == 'sgd':
			return self.sgd_optimization(theta, derivative)
		elif self.optimize_algorithm == 'adam':
			return self.adam_optimization(theta, derivative, i)
		else:
			return self.rmsprop_optimization(theta, derivative, i)

	def train(self, x_list, target_list):
		x = np.array(x_list,ndmin=2).T
		y = np.array(target_list,ndmin=2).T

		z = [0 for _ in range(self.num_layers)]
		a = [0 for _ in range(self.num_layers)]

		for i in range(0, self.num_layers):
			if i == 0:
				z[i] = np.dot(self.theta[i], x) + self.b[i]
			else:
				z[i] = np.dot(self.theta[i], a[i-1]) + self.b[i]
			a[i] = self.activation(z[i])

		e = [0 for _ in range(self.num_layers)]
		loss = sum((y - a[self.num_layers-1])**2)

		for j in range(self.num_layers-1, -1, -1):
			if j == self.num_layers - 1:
				e[j] = y - a[j]
			else:
				e[j] = np.dot(self.theta[j+1].T, e[j+1]) * self.backprop(a[j])

		for i in range(self.num_layers-1, -1, -1):
			if i == 0:
				self.theta[i] = self.optimizer(self.theta[i], np.dot(e[i], x.T), i)
			else:
				self.theta[i] = self.optimizer(self.theta[i], np.dot(e[i], a[i-1].T), i)
			self.b[i] = self.optimizer(self.b[i],e[i],i+self.num_layers)

		return sum(loss)
		

	def classify(self, x_list):
		x = np.array(x_list,ndmin=2).T
		z = [0 for _ in range(self.num_layers)]
		a = [0 for _ in range(self.num_layers)]
		for i in range(0, self.num_layers):
			if i == 0:
				z[i] = np.dot(self.theta[i], x) + (np.sum(self.b[i],axis=1)/self.batch_size).reshape(self.b[i].shape[0],1)
			else:
				z[i] = np.dot(self.theta[i], a[i-1]) + (np.sum(self.b[i],axis=1)/self.batch_size).reshape(self.b[i].shape[0],1)
			a[i] = self.activation(z[i])		
		return a[self.num_layers-1]