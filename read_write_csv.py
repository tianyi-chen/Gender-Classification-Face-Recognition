import numpy as np
import csv

def read(act, file=''):
    X = []
    for i in range(len(act)):
        im_list = np.zeros(1024)  # list of images of a particular actor
        with open(file + str(i) + '.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=';')
            for row in readCSV:
                im_list = np.vstack((im_list, np.array(list(map(float, row[:-1])))))
        X.append(im_list[1:, ])

    return X

def write(X, file=''):
    for i in range(len(X)):
        out = open(file + str(i) + '.csv', 'w')
        for row in X[i]:
            for col in row:
                out.write('%1f;' % col)
            out.write('\n')
        out.close()