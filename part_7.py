import numpy as np
import part_6
import grad_descent
import prepare_x_y

def classify_face(x, theta):
    x = np.vstack( (np.ones((1, x.shape[1])), x))
    y_pred = np.argmax(np.dot(theta.T, x), axis=0)
    return y_pred

def prepare(X_train, X_val):
    x_train, y_train = prepare_x_y.prepare_x_y_2(X_train[0], 0)
    x_val, y_val = prepare_x_y.prepare_x_y_2(X_val[0], 0)
    for i in range(1, len(X_train)):
        x_train_i, y_train_i = prepare_x_y.prepare_x_y_2(X_train[i], i)
        x_train = np.vstack((x_train, x_train_i))
        y_train = np.vstack((y_train, y_train_i))

        x_val_i, y_val_i = prepare_x_y.prepare_x_y_2(X_val[i], i)
        x_val = np.vstack((x_val, x_val_i))
        y_val = np.vstack((y_val, y_val_i))

    x_train = x_train / 255.0
    x_val = x_val / 255.0

    return x_train, x_val, y_train, y_val

def fit_model(x, y, alpha, EPS, max_iter):
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
    theta0 = np.ones((1025, 6)) * 0.01
    theta, cost, iters = grad_descent.grad_descent(part_6.cost, part_6.gradient, x.T, y.T, theta0, alpha, EPS, max_iter)

    return theta, cost, iters

def classify(x, theta):
    x = np.vstack( (np.ones((1, x.shape[1])), x))
    y_pred = np.argmax(np.dot(theta.T, x), axis=0)
    return y_pred