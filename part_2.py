from pylab import *
import numpy as np

np.random.seed(0)

def split(X, k_train=70, k_val=10, k_test=10):
    '''
    Split the feature data into non-overlapping training, validation and test sets.
    :param X: list, of which each element corresponding to a particualr actor
    :param k_train: int, the size of the training set
    :param k_val: int, the size of the validation set
    :param k_test: int, the size of the test set
    :return: X_train, X_val, X_test
    '''
    X_train = []
    X_val = []
    X_test = []
    for i in range(len(X)):
        # Training set
        ix = list(range(len(X[i])))  # all possible indices of the images for the i-th actor
        ix_train = np.random.choice(ix, k_train, replace=False)
        X_train_i = np.zeros((len(ix_train), 1024))  # training images of the i-th actor
        for j in range(len(ix_train)):
            X_train_i[j] = X[i][ix_train[j]]
        X_train.append(X_train_i)

        # Validation set
        ix_rest = [n for n in ix if n not in ix_train] # remove indices for training set
        ix_val = np.random.choice(ix_rest, k_val, replace=False)
        X_val_i = np.zeros((len(ix_val), 1024))  # validation images of the i-th actor
        for j in range(len(ix_val)):
            X_val_i[j] = X[i][ix_val[j]]
        X_val.append(X_val_i)

        # Test set
        ix_rest = [n for n in ix_rest if n not in ix_val] # remove indices for validation set
        ix_test = np.random.choice(ix_rest, k_test, replace=False)
        X_test_i = np.zeros((len(ix_test), 1024))  # testing images of the i-th actor
        for j in range(len(ix_test)):
            X_test_i[j] = X[i][ix_test[j]]
        X_test.append(X_test_i)
    return X_train, X_val, X_test

