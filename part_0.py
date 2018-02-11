from pylab import *
import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
import os
# import urllib
import urllib.request

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray / 255.


def download(act, uncropped, cropped):
    '''
    Downlad images from urls in faces_subset.txt. Crop and resize the images.
    :param act: list, actor names
    :param uncropped: string, uncropped image folder directory
    :param cropped: string, cropped image folder directory
    :return: X
    '''
    # Create folders
    if not os.path.exists(uncropped):
        os.makedirs(uncropped)
    if not os.path.exists(cropped):
        os.makedirs(cropped)
    # image data
    ## X is the list of images of all six actors. Each element of X is a list of images of an actor.
    ## Each image is an 1-D array of length 1024
    X = []
    # Download and crop images
    for a in act:
        name = a.split()[1].lower()  # surname
        i = 0  # id
        im_list = np.zeros(1024)  # list of images of a particular actor
        # For each image
        for line in open("faces_subset.txt"):
            if a in line:
                filename = name + str(i) + '.' + line.split()[4].split('.')[-1]  # filename = surname + id
                # A version without timeout (uncomment in case you need to unsupress exceptions, which timeout() does)
                # urllib.request.urlretrieve(line.split()[4], "uncropped/" + filename)
                timeout(urllib.request.urlretrieve, (line.split()[4], uncropped + filename), {}, 30)
                if not os.path.isfile(uncropped + filename):
                    continue
                i += 1
                try:
                    im = imread(uncropped + filename)
                    # Convert the image to gray scale
                    im = rgb2gray(im)
                    # Crop downloaded images
                    [x1, y1, x2, y2] = np.array(line.split()[5].split(',')).astype(int)
                    im = im[y1:y2, x1:x2]
                    # Resize it
                    im = imresize(im, (32, 32))
                    imsave(name=cropped + filename, arr=im)
                    im_list = vstack((im_list, im.reshape(1024)))
                except:
                    print("Error: couldn't open file " + filename)
                    continue

        X.append(im_list[1:, ])  # remove the first row of zeros

    return X
