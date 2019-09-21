
import sys
import numpy as np
import pandas as pd
# import joblib just to save our model
from sklearn.externals import joblib

# import functions that we implemented in relevant python source files
from model import FCLayer, ActivationLayer, NeuralNet
from losses import softmax, cross_entropy, delta_cross_entropy
from activations import sigmoid, sigmoid_derivative, tanh, tanh_derivative, leaky_relu, leaky_relu_derivative


def create_dict(dfr):
	# create a dictionary to save for each inital feature, its corresponding indices after the one-hot encoding
	prev = '' 
	feat_dict = {}
	for i, key in enumerate(dfr.keys()):
		curr = key.split("__")[0]
		if curr != prev:
			feat_dict[curr] = [i]
		else:
			feat_dict[curr].append(i)
		prev = curr
	return feat_dict


def transform_data(x):
	'''
	transform the input tensor into the appropriate form
	'''
	if np.isnan(x):
		return 0
	elif x>0:
		return 1
	else:
		return -1


def create_labels(X):
	n, m = X.shape
	Y = np.zeros((n, m, 2), dtype=int)
	for i in range(n):
		for j in range(m):
			if X[i,j] == -1:
				Y[i,j,0] = 1
			elif X[i,j] == 1:
				Y[i,j,1] = 1
	return Y



if __name__ == '__main__':

	
	# read train/val/test set from csv files
	df_train = pd.read_csv("../datasets/train_set.csv", sep=',', encoding='latin-1', dtype=str).drop(columns='session_id')
	df_val   = pd.read_csv("../datasets/val_set.csv", sep=',', encoding='latin-1', dtype=str).drop(columns='session_id')
	df_test  = pd.read_csv("../datasets/test_set.csv", sep=',', encoding='latin-1', dtype=str).drop(columns='session_id')	
	print("Shape of training set: {} --- Shape of validation set: {} --- Shape of test set: {}".format(df_train.shape, df_val.shape, df_test.shape))

	# use numpy matrices instead of pandas dataframes
	X_train = df_train.as_matrix().astype(float)
	X_val   = df_val.as_matrix().astype(float)
	X_test  = df_test.as_matrix().astype(float)
	
	Y_train = create_labels(X_train)
	Y_val   = create_labels(X_val)
	Y_test  = create_labels(X_test)

	feat_num = X_train.shape[1]
	X_train = np.reshape(X_train, (X_train.shape[0], 1, feat_num))
	X_val   = np.reshape(X_val, (X_val.shape[0], 1, feat_num))
	X_test   = np.reshape(X_test, (X_test.shape[0], 1, feat_num))
	
	X_train = np.vectorize(transform_data)(X_train)
	X_val   = np.vectorize(transform_data)(X_val)
	X_test  = np.vectorize(transform_data)(X_test)
	print("Shape of training set: {} --- Shape of validation set: {}".format(X_train.shape, X_val.shape))
	print("Shape of training labels: {} --- Shape of validation labels: {}".format(Y_train.shape, Y_val.shape))
	feat_dict   = create_dict(df_val)

	l     			= [25, 10, 25]
	early_stopping  = True
	use_bias_list   = [False, True]
	input_drop_list = [0, 10, 100]
	
	for input_drop in input_drop_list:
		for use_bias in use_bias_list:	
			
			if use_bias:
				filename = "train_drop"+str(input_drop)+"-bias"
			else:
				filename = "train_drop"+str(input_drop)
			
			net = NeuralNet()
			net.add(FCLayer(feat_num, l[0], use_bias))
			net.add(ActivationLayer(tanh, tanh_derivative))
			net.add(FCLayer(l[0], l[1], use_bias))
			net.add(ActivationLayer(tanh, tanh_derivative))
			net.add(FCLayer(l[1], l[2], use_bias))
			net.add(ActivationLayer(tanh, tanh_derivative))	
			net.add(FCLayer(l[2], 2*feat_num, use_bias))
			net.add(ActivationLayer(tanh, tanh_derivative))
			net.use(cross_entropy, delta_cross_entropy)
			net.fit(X_train, Y_train, X_val, Y_val, feat_dict, epochs=15, learning_rate=1e-4, 
					filename=filename+".csv", early_stopping=early_stopping, input_drop=input_drop)
			joblib.dump(net, filename+'.pkl')

			# evaluate model after training in test set
			test_error, test_acc = net.evaluate(X_test, Y_test, feat_dict)
			with open("testing_results.txt", "a+") as f:
				f.write("Average Testing Error: {} --- Average Testing Accuracy: {}\n".format(test_error, test_acc))
		
