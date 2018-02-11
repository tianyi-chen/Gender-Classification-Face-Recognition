import numpy as np

def cost(X, y, theta):
    '''
    X: n*m ndarray
    y: k*m ndarray
    theta: n*k ndarray
    '''
    X = np.vstack( (np.ones((1, X.shape[1])), X))
    return np.sum((np.dot(theta.T, X) - y) ** 2)

def gradient(X, y, theta):
    '''
    X: n*m ndarray
    y: k*m ndarray
    theta: n*k ndarray
    '''
    X = np.vstack( (np.ones((1, X.shape[1])), X))
    inner_product = np.dot(theta.T, X) - y
    return 2 * np.dot(X, inner_product.T)

def finite_difference(X, y, theta, p, q, h=1e-8):
    '''
    p, q: index of the element to approximate the gradient
    '''
    theta0 = np.copy(theta)
    theta[p,q] = theta[p,q] + h
    return (cost(X, y, theta) - cost(X, y, theta0))/h


