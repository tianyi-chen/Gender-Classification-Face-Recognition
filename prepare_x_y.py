import numpy as np

def prepare_x_y(X, y_val):
    '''
    Pepare the features and targets of an actor.
    :param X: list, a list of images of a particular actor
    :param y_val: int, the value of target, either 0 or 1
    :return: x, y
    '''
    x = np.array(X)
    y = np.ones(x.shape[0]) * y_val
    return x, y

def prepare_x_y_2(X, k):
    x = np.array(X)
    y = np.zeros((len(X), 6)) # 6 categories
    y[:,k] = 1
    return x, y

