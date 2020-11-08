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
        for i,(b,w) in enumerate (zip(self.biases,self.weights)): #going through biases and weights for forward pass
            if i < len(self.weights)-1:
                X = np.maximum(0,(np.dot(X,w)+b.T)) #applying relu to get activation function
            else:
                X = np.dot(X,w)+b.T #no relu for output layer
        return X #returning output of forward pass


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


        del_W = [np.zeros(w.shape) for w in self.weights]
        del_b = [np.zeros(b.shape) for b in self.biases]




        #forward pass

        for x,y in zip(X,y):
            partial_b = [np.zeros(b.shape) for b in self.biases]

            partial_w = [np.zeros(w.shape) for w in self.weights]

            activation = x.reshape(len(x),1) # reshaping it to take one example each
            activations = [activation] #to store the activations
            Z = [] #to store Z(Xw+b)

            #applying forward pass in order to save activations and Z required during backward pass
            for i,(b,w) in enumerate(zip(self.biases,self.weights)):
                z = np.dot(w.T,activation) + b
                Z.append(z)
                if i < len(self.weights) - 1:
                    activation = np.maximum(0,z)
                else:
                    activation = z
                activations.append(activation)

            diff = (activations[-1]-y) #taking derivative of loss function with respect to b
            partial_b[-1] = diff
            partial_w[-1] = np.dot(diff,activations[-2].T) + lamda*(self.weights[-1].T)	#taking derivative of loss function with respect to w



            for l in range(2,self.num_layers+2):
                z = Z[-l]
                relu_prime = np.where(z < 0, 0, 1) #taking derivative of relu
                diff = np.dot(self.weights[-l+1],diff)*relu_prime
                partial_b[-l] = diff
                partial_w[-l] = np.dot(diff,activations[-l-1].T) + lamda*(self.weights[-l].T)
            del_W = [dw+dpw.T/X.shape[0] for dw,dpw in zip(del_W,partial_w)]
            del_b = [db+dpb/X.shape[0] for db,dpb in zip(del_b,partial_b)]

        #print(del_W)
        return del_W,del_b



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
        self.learning_rate = learning_rate

    def step(self, weights, biases, delta_weights, delta_biases):
        '''
        Parameters
        ----------
            weights: Current weights of the network.
            biases: Current biases of the network.
            delta_weights: Gradients of weights with respect to loss.
            delta_biases: Gradients of biases with respect to loss.
        '''
        updated_weights = [weights - (self.learning_rate/delta_weights.shape[0])*delta_weights
                        for weights,delta_weights in zip(weights,delta_weights)]
        updated_biases = [biases - (self.learning_rate/delta_biases.shape[0])*delta_biases
                        for biases,delta_biases in zip(biases,delta_biases)]

        return [updated_weights, updated_biases]


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
    return (np.square(y-y_hat)).mean(axis=None)


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
    regularization = 0
    for weights in weights:
        regularization += np.linalg.norm(weights)
    return regularization

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
            batch_loss = loss_fn(batch_target, pred, net.weights, net.biases, lamda)
            epoch_loss += batch_loss
            # print(net.weights)
            # print(e, i, rmse(batch_target, pred), batch_loss) # comment after checking

        print(e, epoch_loss) # comment after checking
        print("train rmse : ", rmse(net(train_input),train_target))
        # Write any early stopping conditions required (only for Part 2)
        # Hint: You can also compute dev_rmse here and use it in the early
        # 		stopping condition.

    # After running `max_epochs` (for Part 1) epochs OR early stopping (for Part 2), compute the RMSE on dev data.
    train_pred = net(train_input)
    train_rmse = rmse(train_target,train_pred)
    dev_pred = net(dev_input)
    dev_rmse = rmse(dev_target, dev_pred)
    

    print('RMSE on train data: {:.5f}'.format(train_rmse))
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
    return net(inputs)

def read_data():
    '''
    Read the train, dev, and test datasets
    '''

    train = pd.read_csv('dataset/train.csv')
    dev = pd.read_csv('dataset/dev.csv')
    test = pd.read_csv('dataset/test.csv')

    t = train.values
    train_input = t[:,1:]
    train_target = t[:,:1]
    # print(train_target)

    d = dev.values
    dev_input = d[:,1:]
    dev_target = d[:,:1]
    # print(dev_target)
    test_input = test.values

    return train_input, train_target, dev_input, dev_target, test_input

def main():

    # These parameters should be fixed for Part 1
    max_epochs = 50
    batch_size = 128


    learning_rate = 0.01
    num_layers = 2
    num_units = 128
    lamda = 5 # Regularization Parameter

    train_input, train_target, dev_input, dev_target, test_input = read_data()
    net = Net(num_layers, num_units)
    optimizer = Optimizer(learning_rate)
    train(
        net, optimizer, lamda, batch_size, max_epochs,
        train_input, train_target,
        dev_input, dev_target
    )
    get_test_data_predictions(net, test_input)
    #pred_idx = np.insert(predictions, 0, range(1,predictions.size+1), axis=1)
    #np.savetxt('203050058_2.csv', pred_idx, delimiter=',', header='Id,Predicted',fmt='%0.1f,%d',comments="")

if __name__ == '__main__':
    main()

#http://neuralnetworksanddeeplearning.com/chap1.html : reference
