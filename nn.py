import sys
import os
import numpy as np
import pandas as pd

np.random.seed(42)

NUM_FEATS = 90

class Net(object):
	'''
	'''

	def __init__(self, num_layers, num_units):
		'''
		Initialize the neural network.
		Create weights and biases.

		Here, we have provided an example structure for the weights and biases.
		It is a list of weight and bias matrices, in which, the
		dimensions of weights and biases are (assuming 1 input layer, 2 hidden layers, and 1 output layer):
		weights: [(NUM_FEATS, num_units), (num_units, num_units), (num_units, num_units), (num_units, 1)]
		biases: [(num_units, 1), (num_units, 1), (num_units, 1), (num_units, 1)]

		Please note that this is just an example.
		You are free to modify or entirely ignore this initialization as per your need.
		Also you can add more state-tracking variables that might be useful to compute
		the gradients efficiently.


		Parameters
		----------
			num_layers : Number of HIDDEN layers.
			num_units : Number of units in each Hidden layer.
		'''
		self.num_layers = num_layers
		self.num_units = num_units

		self.biases = []
		self.weights = []
		for i in range(num_layers):

			if i==0:
				# Input layer
				self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units)))
			else:
				# Hidden layer
				self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, self.num_units)))

			self.biases.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

		# Output layer
		self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))
		self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))
	
	
	def relu(self,z):
		return np.maximum(0,z)

	def __call__(self, X):
		'''
		Forward propagate the input X through the network,
		and return the output.
		
		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
		Returns
		----------
			y : Output of the network, numpy array of shape m x 1
		'''

		
		# print(X.shape)
		for i,(b,w) in enumerate (zip(self.biases,self.weights)):
			if i < len(self.weights)-1:
				X = self.relu(np.dot(X,w)+b.T)
			else:
				X = np.dot(X,w)+b.T
		return X
		# raise NotImplementedError

	def backward(self, X, y, lamda):
		'''
		Compute and return gradients loss with respect to weights and biases.
		(dL/dW and dL/db)

		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
			y : Output of the network, numpy array of shape m x 1
			lamda : Regularization parameter.

		Returns
		----------
			del_W : derivative of loss w.r.t. all weight values (a list of matrices).
			del_b : derivative of loss w.r.t. all bias values (a list of vectors).

		Hint: You need to do a forward pass before performing bacward pass.
		'''
		
		
		
		del_b = [np.zeros(b.shape) for b in self.biases]
		del_w = [np.zeros(w.shape) for w in self.weights]

		
		# print("biases")
		# for b in del_b:
			# print(b.shape)
		# print("weights")
		# for w in del_w:
			# print(w.shape)
		
		#forward passs
		res = []
		for x,y in zip(X,y):
			nabla_b = [np.zeros(b.shape) for b in self.biases]
			nabla_w = [np.zeros(w.shape) for w in self.weights]

			activation = x.reshape(len(x),1)
			activations = [activation]
			zs = []

			for i,(b,w) in enumerate(zip(self.biases,self.weights)):
				z = np.dot(w.T,activation) + b
				zs.append(z)
				if i < len(self.weights) - 1:
					activation = self.relu(z)
				else:
					activation = z
				activations.append(activation)
			
			delta = (activations[-1]-y)
			nabla_b[-1] = delta
			nabla_w[-1] = np.dot(delta,activations[-2].T)	
			# print("nabla_b[-1] shape ",nabla_b[-1].shape)
			# print("nabla_w[-1] shape ",nabla_w[-1].shape)


			for l in range(2,self.num_layers+2):
				z = zs[-l]
				rp = relu_prime(z)
				delta = np.dot(self.weights[-l+1],delta)*rp
				nabla_b[-l] = delta
				nabla_w[-l] = np.dot(delta,activations[-l-1].T)
				# print("nabla_b[-l] shape ",nabla_b[-l].shape)
				# print("nabla_w[-l] shape ",nabla_w[-l].shape)
			del_b = [db+dnb/(X.shape[0]) for db,dnb in zip(del_b,nabla_b)]
			del_w = [dw+dnw.T/(X.shape[0]) for dw,dnw in zip(del_w,nabla_w)]
			# print("here")
			# print(del_w)
		
		# print("Return shape of db")
		# for db in del_b:
			# print(db.shape)
		# print("Return shape of dw")			
		# for db in del_w:
			# print(db.shape)			
		# print(del_w)
		return del_w,del_b
		


def relu_prime(x):
	return np.where(x < 0, 0, 1)

class Optimizer(object):
	'''
	'''

	def __init__(self, learning_rate):
		'''
		Create a Gradient Descent based optimizer with given
		learning rate.

		Other parameters can also be passed to create different types of
		optimizers.

		Hint: You can use the class members to track various states of the
		optimizer.
		'''
		self.lr = learning_rate

	def step(self, weights, biases, delta_weights, delta_biases):
		'''
		Parameters
		----------
			weights: Current weights of the network.
			biases: Current biases of the network.
			delta_weights: Gradients of weights with respect to loss.
			delta_biases: Gradients of biases with respect to loss.
		'''
		updated_w = [w - (self.lr)*nw 
						for w,nw in zip(weights,delta_weights)]
		updated_b = [b - (self.lr)*nb
						for b,nb in zip(biases,delta_biases)]
		# print("updated w shapes")
		# for uw in updated_w:
			# print(uw.shape)
		# print("updated b shape")
		# for ub in updated_b:
			# print(ub.shape)
		return [updated_w, updated_b]


def loss_mse(y, y_hat):
	'''
	Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		MSE loss between y and y_hat.
	'''
	mse = (np.square(y-y_hat)).mean(axis=None)
	return mse

def loss_regularization(weights, biases):
	'''
	Compute l2 regularization loss.

	Parameters
	----------
		weights and biases of the network.

	Returns
	----------
		l2 regularization loss 
	'''
	reg = 0
	for w in weights:
		reg += np.sum(np.square(w))
	return reg

def loss_fn(y, y_hat, weights, biases, lamda):
	'''
	Compute loss =  loss_mse(..) + lamda * loss_regularization(..)

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1
		weights and biases of the network
		lamda: Regularization parameter

	Returns
	----------
		l2 regularization loss 
	'''
	return loss_mse(y,y_hat) + lamda*loss_regularization(weights,biases)

def rmse(y, y_hat):
	'''
	Compute Root Mean Squared Error (RMSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		RMSE between y and y_hat.
	'''
	return (loss_mse(y,y_hat))**0.5

# From -> https://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function
def early_stopping(dev_loss):
    last_k_losses.append(dev_loss)
    if len(last_k_losses) < k:
        return False
    last_k_losses.pop(0)
    return (max(last_k_losses) - min(last_k_losses)) < min_diff

k = 3 #for early stopping # check last k error values
last_k_losses = []
min_diff = 0.09


def train(
	net, optimizer, lamda, batch_size, max_epochs,
	train_input, train_target,
	dev_input, dev_target
):
	'''
	In this function, you will perform following steps:
		1. Run gradient descent algorithm for `max_epochs` epochs.
		2. For each bach of the training data
			1.1 Compute gradients
			1.2 Update weights and biases using step() of optimizer.
		3. Compute RMSE on dev data after running `max_epochs` epochs.

	Here we have added the code to loop over batches and perform backward pass
	for each batch in the loop.
	For this code also, you are free to heavily modify it.
	'''

	m = train_input.shape[0]

	for e in range(max_epochs):
		epoch_loss = 0.
		for i in range(0, m, batch_size):
			batch_input = train_input[i:i+batch_size]
			batch_target = train_target[i:i+batch_size]
			pred = net(batch_input)
			# Compute gradients of loss w.r.t. weights and biases
			dW, db = net.backward(batch_input, batch_target, lamda)
			# Get updated weights based on current weights and gradients
			weights_updated, biases_updated = optimizer.step(net.weights, net.biases, dW, db)

			# Update model's weights and biases
			net.weights = weights_updated	
			net.biases = biases_updated
			# Compute loss for the batch
			# print(weights_updated)

			batch_loss = loss_fn(batch_target, pred, net.weights, net.biases, lamda)
			epoch_loss += batch_loss

			print(e, i, rmse(batch_target, pred), batch_loss)

		print(e, epoch_loss)
		if(early_stopping(batch_loss)):
			break
		# Write any early stopping conditions required (only for Part 2)
		# Hint: You can also compute dev_rmse here and use it in the early
		# 		stopping condition.

	# After running `max_epochs` (for Part 1) epochs OR early stopping (for Part 2), compute the RMSE on dev data.
	dev_pred = net(dev_input)
	dev_rmse = rmse(dev_target, dev_pred)

	print('RMSE on dev data: {:.5f}'.format(dev_rmse))


def get_test_data_predictions(net, inputs):
	'''
	Perform forward pass on test data and get the final predictions that can
	be submitted on Kaggle.
	Write the final predictions to the part2.csv file.

	Parameters
	----------
		net : trained neural network
		inputs : test input, numpy array of shape m x d

	Returns
	----------
		predictions (optional): Predictions obtained from forward pass
								on test data, numpy array of shape m x 1
	'''
	results = net(inputs)
	return results
	# raise NotImplementedError

def get_data(filename, is_test = False):
	data_frame = pd.read_csv(filename)
	x = data_frame.values
	if is_test:
		return x
	else:
		inputs = x[:,1:]
		targets = x[:,:1]
		return inputs,targets
	

def read_data():
	'''
	Read the train, dev, and test datasets
	'''
	train_input, train_target = get_data('data/train.csv')
	dev_input,dev_target = get_data('data/dev.csv')
	test_input = get_data('data/test.csv',True)

	return train_input, train_target, dev_input, dev_target, test_input


def main():

	# These parameters should be fixed for Part 1
	max_epochs = 50
	batch_size = 256
	learning_rate = 0.0001
	num_layers = 1
	num_units = 64
	lamda = 0.1 # Regularization Parameter
	
	train_input, train_target, dev_input, dev_target, test_input = read_data()
	net = Net(num_layers, num_units)
	optimizer = Optimizer(learning_rate)
	train(
		net, optimizer, lamda, batch_size, max_epochs,
		train_input, train_target,
		dev_input, dev_target
	)
	predictions = get_test_data_predictions(net, test_input)
	# print(predictions.shape)
	pred_idx = np.insert(predictions, 0, range(1,predictions.size+1), axis=1)
	np.savetxt('pred.csv', pred_idx, delimiter=',', header='Id,Predicted',fmt='%0.1f,%d',comments="")
if __name__ == '__main__':
	main()
