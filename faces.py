import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.misc import imshow
import part_0
import part_2
import part_3
import part_5
import part_6
import part_7
import grad_descent
import prepare_x_y
import read_write_csv
import accuracy

np.random.seed(0)


if __name__ == '__main__':

    #### Part 0: data preparation ####
    # To run the file, you need an empty uncropped and an empty cropped folder
    # List of actor/ress names
    act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    X = part_0.download(act, "uncropped/", "cropped/")
    # Write to csv
    # read_write_csv.write(X)

    # Should not be uncommented unless csv files are provided
    # X = read_write_csv.read(act)

    #### Part 1: data exploration ####
    # Display cropped and resized images of actors
    X_train, X_val, X_test = part_2.split(X, 1, 1, 1)
    fig = plt.figure()
    for i in range(len(X_train)):
        a = fig.add_subplot(2, 3, i+1)
        im = X_train[i].reshape((32,32))
        imgplot = plt.imshow(im, cmap=cm.gray)
    plt.show()

    #### Part 2: datasets ####
    # Split the data into training, validation and test sets
    X_train, X_val, X_test = part_2.split(X, 70, 10, 10)

    #### Part 3: Baldwin vs. Carell classification ####
    # Build a classifier to distinguish pictures of Alec Baldwin from pictures of Steve Carell
    print("Running Baldwin vs. Carell classification...")
    x_train, y_train = part_3.prepare(X_train[3], X_train[5]) # X_train[3] contains the photos of Baldwin
    theta0 = np.ones(1025) * 0.01
    theta, cost, iters = part_3.fit_model(x_train, y_train, theta0, 1e-5, 1e-6, 5000)
    print("Value of the cost function on the training set:", part_3.f(x_train.T, y_train.T, theta))

    x_val, y_val = part_3.prepare(X_val[3], X_val[5])
    print("Value of the cost function on the validation set:", part_3.f(x_val.T, y_val.T, theta))

    # Performance of the model on the training set
    y_pred_train = part_3.classify(x_train.T, theta)
    acc_train = accuracy.accuracy(y_train, y_pred_train)
    print("Accuracy on the training set:", acc_train)

    # Performance of the model on the validation set
    y_pred_val = part_3.classify(x_val.T, theta)
    acc_val = accuracy.accuracy(y_val, y_pred_val)
    print("Accuracy on the validation set:", acc_val)

    # Plot the value of the cost function against the # of iterations
    plt.plot(iters, cost)
    plt.xlabel('Number of iterations')
    plt.ylabel('f(x)')
    plt.show()

    #### Part 4: training size and early stopping ####
    print("Plotting images of thetas...")
    fig = plt.figure()
    theta, cost, iters = part_3.fit_model(x_train, y_train, theta0, alpha=1e-5, EPS=1e-6, max_iter=5000)
    a = fig.add_subplot(2, 2, 1)
    im = theta[1:, ].reshape((32, 32))  # remove the parameters for bias
    imgplot = plt.imshow(im, cmap="RdBu", interpolation='spline16')
    a.set_title('training_size: 70 max_iter = 5,000')

    X_train_2, X_val_2, X_test_2 = part_2.split(X, 2, 2, 2)
    x_train_2, y_train_2 = part_3.prepare(X_train_2[3], X_train_2[5])
    theta, cost, iters = part_3.fit_model(x_train_2, y_train_2, theta0, alpha=1e-5, EPS=1e-6, max_iter=5000)
    a = fig.add_subplot(2, 2, 2)
    im = theta[1:, ].reshape((32, 32))  # remove the parameters for bias
    imgplot = plt.imshow(im, cmap="RdBu", interpolation='spline16')
    a.set_title('training_size: 2 max_iter = 5,000')

    theta, cost, iters = part_3.fit_model(x_train, y_train, theta0, alpha=1e-5, EPS=1e-6, max_iter=500)
    a = fig.add_subplot(2, 2, 3)
    im = theta[1:, ].reshape((32, 32))  # remove the parameters for bias
    imgplot = plt.imshow(im, cmap="RdBu", interpolation='spline16')
    a.set_title('training size: 70 max_iter = 500')

    theta, cost, iters = part_3.fit_model(x_train, y_train, theta0, alpha=1e-5, EPS=1e-6, max_iter=50000)
    a = fig.add_subplot(2, 2, 4)
    im = theta[1:, ].reshape((32, 32))  # remove the parameters for bias
    imgplot = plt.imshow(im, cmap="RdBu", interpolation='spline16')
    a.set_title('training size: 70 max_iter = 50,000')

    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    theta, cost, iters = part_3.fit_model(x_train, y_train, theta0, alpha=1e-5, EPS=1e-6, max_iter=5000)
    a = fig.add_subplot(1, 2, 1)
    im = theta[1:, ].reshape((32, 32))  # remove the parameters for bias
    imgplot = plt.imshow(im, cmap="RdBu", interpolation='spline16')
    a.set_title('theta = [0.01 0.01 ... 0.01]')

    theta0 = np.random.rand(1025)
    theta, cost, iters = part_3.fit_model(x_train, y_train, theta0, alpha=1e-5, EPS=1e-6, max_iter=5000)
    a = fig.add_subplot(1, 2, 2)
    im = theta[1:, ].reshape((32, 32))  # remove the parameters for bias
    imgplot = plt.imshow(im, cmap="RdBu", interpolation='spline16')
    a.set_title('random theta in [0, 1)')

    plt.show()

    #### Part 5: gender classification ####
    # Build classifiers that classify the actors as male or female
    print("Running gender classification...")
    act_2 = ['Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan', 'Kristin Chenoweth', 'Fran Drescher',
           'America Ferrera']
    X_others = part_0.download(act_2, "uncropped_others/", "cropped_others/")
    # read_write_csv.write(X_others, file = 'others')
    # Should not be uncommented unless csv files are provided
    # X_others = read_write_csv.read(act, file='others')

    accs_train = []  # accuracies with different training sizes
    accs_val = []
    accs_others = []
    # Randomly select 10 images per actor from the other six actors as the test set
    X_others_test, _, _ = part_2.split(X_others, k_train=10, k_val=10, k_test=10)
    l = list(range(1, 101, 5))
    # Run gradient descent with different training sizes
    for i in l:
        X_train, _, _ = part_2.split(X, k_train=i, k_val=10, k_test=10)
        x_train, y_train, x_val, y_val, x_others, y_others = part_5.prepare(X_train, X_val, X_others_test, i)
        acc_train, acc_val, acc_others = part_5.fit_predict(x_train, y_train, 1e-6, 1e-6, 30000, x_val, y_val, x_others, y_others)
        accs_train.append(acc_train)
        accs_val.append(acc_val)
        accs_others.append(acc_others)
    print("Accuracies on training, validation and test sets with training size of", "per actor:", accs_train[-1], accs_val[-1], accs_others[-1])

    plt.plot(l, accs_train, 'r-', label='training')
    plt.plot(l, accs_val, 'b-', label='validation')
    plt.plot(l, accs_others, 'c-', label='other actors')
    plt.xlabel('Training size (# of images per actor)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    #### Part 6: gradient and finite difference ####
    # Compare gradient function and finite difference approximation
    print("Comparing gradient function and finite difference approximation...")
    X_train, X_val, X_test = part_2.split(X, k_train=70, k_val=10, k_test=10)
    x_train, x_val, y_train, y_val = part_7.prepare(X_train, X_val)
    # Randomly initialize theta with values in [0,1)
    theta = np.random.rand(1025, 6)
    print("h=1e-6/abs difference", "h=1e-7/abs difference", "gradient")
    # Randomly select 5 coordinates
    for i in range(5):
        p = np.random.randint(0, 1025)
        q = np.random.randint(0, 6)
        g1_6 = part_6.finite_difference(x_train.T, y_train.T, theta, p, q, 1e-6)
        g1_7 = part_6.finite_difference(x_train.T, y_train.T, theta, p, q, 1e-7)
        g2 = part_6.gradient(x_train.T, y_train.T, theta)
        print(g1_6, abs(g1_6 - g2[p, q]), g1_7, abs(g1_7 - g2[p, q]), g2[p, q])

    #### Part 7: face recognition ####
    # Run gradient descent on the set of six actors in act in order to perform face recognition
    print("Running face recognition...")
    X_train, X_val, X_test = part_2.split(X, k_train=70, k_val=10, k_test=10)
    x_train, x_val, y_train, y_val = part_7.prepare(X_train, X_val)
    theta, cost, iters = part_7.fit_model(x_train, y_train, 1e-6, 1e-6, 30000)

    # Performance of the model on the training set
    y_pred = part_7.classify(x_train.T, theta)
    y_true = np.argmax(y_train, axis=1)
    acc_train = accuracy.accuracy(y_true.T, y_pred.T)
    print("Accuracy on the training set:", acc_train)

    # Performance of the model on the validation set
    y_pred = part_7.classify(x_val.T, theta)
    y_true = np.argmax(y_val, axis=1)
    acc_val = accuracy.accuracy(y_true.T, y_pred.T)
    print("Accuracy on the validation set:", acc_val)

    #### Part 8: visualization of thetas ####
    print("Plotting images of thetas...")
    fig = plt.figure()
    for i in range(theta[1:,].shape[1]):
        a = fig.add_subplot(3, 2, i+1)
        im = theta[1:,][:,i].reshape((32, 32))  # remove the parameters for bias
        imgplot = plt.imshow(im, cmap="RdBu", interpolation='spline16')
        a.set_title(act[i])

    plt.tight_layout()
    plt.show()

