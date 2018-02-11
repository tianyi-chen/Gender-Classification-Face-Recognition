import numpy as np
import grad_descent
import prepare_x_y

# Quadratic cost function
def f(x, y, theta):
    x = np.vstack( (np.ones((1, x.shape[1])), x))
    return np.sum( (y - np.dot(theta.T, x)) ** 2)

# Gradient
def df(x, y, theta):
    x = np.vstack( (np.ones((1, x.shape[1])), x))
    return -2*np.sum((y-np.dot(theta.T, x))*x, 1)

def prepare(X1, X2):
    # Get training data and labels of Baldwin
    x_baldwin, y_baldwin = prepare_x_y.prepare_x_y(X1, 1)
    # Get training data and labels of Carell
    x_carell, y_carell = prepare_x_y.prepare_x_y(X2, 0)
    # Combine Baldwins and Carell to get the full training set
    x = np.vstack((x_baldwin, x_carell)) / 255.0
    y = np.concatenate((y_baldwin, y_carell), axis=0)
    return x, y

def fit_model(x, y, theta0, alpha, EPS, max_iter):
    '''
    Use linear regression to build a classifer to distinguish pictures of Alec Baldwin from pictures of Steve Carell.
    :param x: features
    :param y: targets
    :param alpha: learning rate
    :param EPS:
    :param max_iter:
    :return: theta, cost, iters
    '''
    # Run gradient descent
    theta, cost, iters = grad_descent.grad_descent(f, df, x.T, y.T, theta0, alpha, EPS, max_iter)

    return theta, cost, iters

def classify(X, theta):
    '''
    Classify images into Baldwin or Carell with linear regression model parameters theta.
    :param X: ndarray, input features
    :param theta: ndarray, model parameters
    :return: y_pred
    '''
    X = np.vstack((np.ones((1, X.shape[1])), X))
    y_pred = np.dot(theta.T, X)
    for i in range(y_pred.shape[0]):
        if y_pred[i] > 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred