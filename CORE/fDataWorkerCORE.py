# ---------------------------------------------------------------------# IMPORTS


import numpy as np
from numpy import *
import theano
import theano.tensor as T
import csv as csv
import cPickle
import time
from scipy.cluster.vq import *
import matplotlib.pylab as plt


#---------------------------------------------------------------------# ROLLOUT


def rollOut(l):
    numClasses = 4
    n = l.shape[0]
    l = l.reshape((1, -1))
    l = np.tile(l, (numClasses, 1))
    g = np.array(range(1, numClasses + 1)).reshape((-1, 1))
    g = np.tile(g, (1, n))
    res = l == g * 1.0
    return res


#---------------------------------------------------------------------# DATA GENERATOR


def noisedSinGen(number=10000, phase=0):
    # Number of points
    time = np.linspace(-np.pi * 10, np.pi * 10, number)
    # y=(    sin(x - pi /2) + cos(x * 2 * pi)         ) / 10 + 0.5
    #series = (np.sin((time+phase)-np.pi/2) + np.cos((time+phase)*2.0*np.pi))/10.0+0.5
    series = np.sin(time + phase) / 2 + 0.5
    noise = np.random.uniform(0.0, 0.01, number)
    data = series + noise  # Input data
    #fig = plt.figure(figsize=(200, 9))
    #ax = fig.add_subplot(1, 1, 1)
    #ax.scatter(time, data, s=5, alpha=0.5, color="blue")
    return (time, data)


#---------------------------------------------------------------------#
#---------------------------------------------------------------------#