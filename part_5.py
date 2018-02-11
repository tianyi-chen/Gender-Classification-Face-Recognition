import numpy as np
import prepare_x_y
import grad_descent
import accuracy

# Quadratic cost function
def f(x, y, theta):
    x = np.vstack( (np.ones((1, x.shape[1])), x))
    return np.sum( (y - np.dot(theta.T, x)) ** 2)

# Gradient
def df(x, y, theta):
    x = np.vstack( (np.ones((1, x.shape[1])), x))
    return -2*np.sum((y-np.dot(theta.T, x))*x, 1)


def prepare(X_train, X_val, X_train_others, train_size):
    '''
    Pepare the training, validation sets and targets of an actor for gender classification.
    '''
    # Female
    x_train_f, y_train_f = np.zeros((3 * train_size, 1024)), np.zeros(3 * train_size)
    x_val_f, y_val_f = np.zeros((3 * 10, 1024)), np.zeros(3 * 10)
    x_others_m, y_others_m = np.zeros((3 * 10, 1024)), np.zeros(3 * 10)

    for i in range(3):
        # first three in X_train & X_val are female
        x_train_f[train_size * i:train_size * (i + 1)], y_train_f[train_size * i:train_size * (i + 1)]\
            = prepare_x_y.prepare_x_y(X_train[i], y_val=0)
        x_val_f[10 * i:10 * (i + 1)], y_val_f[10 * i:10 * (i + 1)] \
            = prepare_x_y.prepare_x_y(X_val[i], y_val=0)
        # first three in X_train_others are male
        x_others_m[10 * i:10 * (i + 1)], y_others_m[10 * i:10 * (i + 1)] \
            = prepare_x_y.prepare_x_y(X_train_others[i], y_val=1)
    # Male
    x_train_m, y_train_m = np.zeros((3 * train_size, 1024)), np.zeros(3 * train_size)
    x_val_m, y_val_m = np.zeros((3 * 10, 1024)), np.zeros(3 * 10)
    x_others_f, y_others_f = np.zeros((3 * 10, 1024)), np.zeros(3 * 10)

    for i in range(3, len(X_train)):
        x_train_m[train_size * (i-3):train_size * (i-2)], y_train_m[train_size * (i-3):train_size * (i-2)] \
            = prepare_x_y.prepare_x_y(X_train[i], y_val=1)
        x_val_m[10 * (i-3):10 * (i-2)], y_val_m[10 * (i-3):10 * (i-2)] \
            = prepare_x_y.prepare_x_y(X_val[i], y_val=1)
        x_others_f[10 * (i-3):10 * (i-2)], y_others_f[10 * (i-3):10 * (i-2)] \
            = prepare_x_y.prepare_x_y(X_train_others[i], y_val=0)

    x_train = np.vstack((x_train_f, x_train_m)) / 255.0
    y_train = np.concatenate((y_train_f, y_train_m), axis=0)

    x_val = np.vstack((x_val_f, x_val_m)) / 255.0
    y_val = np.concatenate((y_val_f, y_val_m), axis=0)

    x_others = np.vstack((x_others_f, x_others_m)) / 255.0
    y_others = np.concatenate((y_others_f, y_others_m), axis=0)

    return x_train, y_train, x_val, y_val, x_others, y_others

def classify(X, theta):
    '''
    Classify images into male or female with linear regression model parameters theta.
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

def fit_predict(x_train, y_train, alpha, EPS, max_iter, x_val, y_val, x_others, y_others):
    '''
    Fit a linear regression model and classify the training and the validation set.
    '''
    # Train the model
    theta0 = np.ones(1025) * 0.01
    theta, cost, iters = grad_descent.grad_descent(f, df, x_train.T, y_train.T, theta0, alpha, EPS, max_iter)
    # Performance of the model on the training set
    y_pred = classify(x_train.T, theta.T)
    acc_train = accuracy.accuracy(y_train.T, y_pred.T)
    # Performance of the model on the validation set
    y_pred = classify(x_val.T, theta.T)
    acc_val = accuracy.accuracy(y_val.T, y_pred.T)
    # Performance of the model on the other 6 actors
    y_pred = classify(x_others.T, theta.T)
    acc_others = accuracy.accuracy(y_others.T, y_pred.T)

    return acc_train, acc_val, acc_others

