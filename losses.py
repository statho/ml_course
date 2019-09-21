
import numpy as np


def softmax(x):
	x -= np.max(x) 					   # subtract x for numeric stability reasons
	return np.exp(x) / np.sum(np.exp(x))


def cross_entropy(X, Y, feat_dict):
    '''
    X : output of neural network [features x 2]
    Y : labels [features x 2]
    feat_dict: dictionary that correlated the inital and the one-hot encoded features
    @ return loss: scalar value -- one-hot features are averaged to have balance in contribution of the initial features to the cost
    '''
    loss = 0
    for f in feat_dict.values():
        l = 0
        c = 0
        for v in f:
            if np.any(Y[v]):
                c += 1
                l -= np.log(sum(Y[v] * softmax(X[v])))
        if c > 0:
            loss += l / c

    return loss


def delta_cross_entropy(X, Y):
    '''
    X : output of neural network [features x 2]
    Y : labels [features x 2]
    @ return grad: dL/dY [1 X 2*features]
    '''
    grad = np.zeros(X.shape)
    for i, _ in enumerate(X):
        if np.any(Y[i]):
            p = softmax(X[i])
            grad[i] = p - Y[i]

    grad = np.reshape(grad, (1, 2*X.shape[0]))
    return grad

