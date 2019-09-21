
import numpy as np


def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def sigmoid_derivative(x):
	# input x -> sigmoid(y)
	return x * (1.0 - x)


def tanh(x):
    return np.tanh(x);

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2


def leaky_relu(x, alpha=0.01):
	return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
	dx = np.ones_like(x)
	dx[x < 0] = alpha
	return dx

