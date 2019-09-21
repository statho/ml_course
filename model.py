
import numpy as np
from losses import softmax


class FCLayer:
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size, use_bias):
        self.use_bias = use_bias
        if use_bias:
            self.bias = np.zeros((1, output_size))    
        self.weights  = np.random.normal(loc=0.0, scale=(1.0/input_size), size=(input_size, output_size))

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights)
        if self.use_bias:
            self.output += self.bias
        return self.output

    # computes dL/dW, dL/dB for a given output_error=dL/dY. Returns input_error=dL/dX.
    def backward_propagation(self, output_error, learning_rate):
    	input_error   = np.dot(output_error, self.weights.T)
    	weights_error = np.dot(self.input.T, output_error)
    	# dBias = output_error

    	# update parameters
    	self.weights -= learning_rate * weights_error
    	if self.use_bias:
            self.bias -= learning_rate * output_error

    	return input_error


class ActivationLayer:

    def __init__(self, activation, activation_derivative):
    	self.activation = activation
    	self.activation_derivative = activation_derivative

    # returns the activated input
    def forward_propagation(self, input_data):
    	self.input = input_data
    	self.output = self.activation(self.input)
    	return self.output

    # Returns input_error=dL/dX for a given output_error=dL/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
    	return self.activation_derivative(self.input) * output_error



class NeuralNet:

    def __init__(self):
    	self.layers = []
    	self.loss = None
    	self.loss_derivative = None

    # add layer to network
    def add(self, layer):
    	self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_derivative):
    	self.loss = loss
    	self.loss_derivative = loss_derivative


    def calculate_accuracy(self, X, Y):
        c   = 0
        acc = 0
        for i, _ in enumerate(X):
            if np.any(Y[i]):
                c += 1
                p = softmax(X[i])
                m = np.argmax(p)
                if Y[i,m] == 1:
                    acc += 1  
        acc /= c
        return acc



    def fit(self, x_train, y_train, x_val, y_val, feat_dict, epochs, learning_rate, filename, early_stopping, input_drop):
    	
        feat_num = x_train.shape[2]
        samples_train = len(x_train)
        samples_val   = len(x_val)

        with open(filename, 'a+') as fname:
            fname.write("Epoch,Training-Loss,Validation-Loss,Accuracy-training,Accuracy-validation\n")


        # drop randomly some input features (can help to learn more general representations)
        if input_drop > 0:      
            for j in range(samples_train):    
                for _ in range(input_drop):
                    r = np.random.randint(0, feat_num)
                    while x_train[j,0,r]==0:
                        r = np.random.randint(0, feat_num)
                    x_train[j,0,r] = 0

        for i in range(epochs):
            err_val   = 0
            err_train = 0
            acc_val   = 0
            acc_train = 0 
            for j in range(samples_train):
                # Training phase -- forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                output  = np.reshape(output, (feat_num, 2))
                err_train += self.loss(output, y_train[j], feat_dict)
                acc_train += self.calculate_accuracy(output, y_train[j])

                # Training phase -- backprobagation 
                error = self.loss_derivative(output, y_train[j])
                for layer in reversed(self.layers):
                	error = layer.backward_propagation(error, learning_rate)


            for j in range(samples_val):
                # Validation phase -- only perform forward propagation to calculate the loss
                output = x_val[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                output  = np.reshape(output, (feat_num, 2))
                err_val += self.loss(output, y_train[j], feat_dict)
                acc_val += self.calculate_accuracy(output, y_train[j])


            # calculate average training and validation error on all samples
            err_train /= samples_train
            err_val   /= samples_val
            acc_train /= samples_train
            acc_val   /= samples_val
            print("Epoch: {} / {}\nTraining Loss : {} --- Training Accuracy : {}\nValidation Loss : {} --- Validation Accuracy : {}".format(i+1, epochs, err_train, acc_train, err_val, acc_val))
            with open(filename, 'a+') as fname:
                fname.write(str(i) + ',' + str(err_train) + ',' + str(err_val) + ',' + str(acc_train) + ',' + str(acc_val)+'\n')

            # add early stopping to avoid overfitting
            if early_stopping:
                if i == 0:
                    prev_acc_val = acc_val
                    patience = -1

                if acc_val <= (prev_acc_val+1e-7):
                    patience += 1
                    prev_acc_val = acc_val
                else:
                	patience = 0

                if patience > 1:
                    print("Early Stopping in epoch : {}!\nTraining Error: {} --- Validation Error: {}".format(i, err_train, err_val))
                    print("Training Accuracy : {} --- Validation Accuracy : {}".format(acc_train, acc_val))
                    return



    def evaluate(self, X, Y, feat_dict):

        samples  = len(X)
        err      = 0
        acc      = 0
        feat_num = X.shape[2]
        for j in range(samples):
            # Forward propagation
            output = X[j]
            for layer in self.layers:
                output = layer.forward_propagation(output)

            output = np.reshape(output, (feat_num, 2))
            err += self.loss(output, Y[j], feat_dict)
            acc += self.calculate_accuracy(output, Y[j])

        err /= samples
        acc /= samples
        return err, acc 


    # predict output for given input
    def predict(self, X):
        result   = np.zeros((X.shape[0], X.shape[2]))
        # sample dimension first
        samples  = len(X)
        feat_num = X.shape[2]
        # run network over all samples
        res = []
        for i in range(samples):
            # forward propagation
            output = X[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)

            output = np.reshape(output, (feat_num, 2))
            p = np.zeros(output.shape)
            for j in range(len(output)):
                p[j]  = softmax(output[j])
                result[i,j] = np.argmax(p[j])
            
        return result
