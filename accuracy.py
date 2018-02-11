import numpy as np

def accuracy(y, y_pred):
    count = np.sum(y == y_pred)
    return count*1.0 / y.shape[0]

