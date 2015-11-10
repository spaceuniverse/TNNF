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

# ---------------------------------------------------------------------# ROLLOUT
def rollOut(l):
    numClasses = 4
    n = l.shape[0]
    l = l.reshape((1, -1))
    l = np.tile(l, (numClasses, 1))
    g = np.array(range(1, numClasses + 1)).reshape((-1, 1))
    g = np.tile(g, (1, n))
    res = l == g * 1.0
    return res

# ---------------------------------------------------------------------# DATA GENERATOR
number = 10000  # Number of points
time = np.linspace(-np.pi * 100, np.pi * 100, number)
series = np.sin(time) + 2
noise = np.random.uniform(0.0, 0.05, number)

data = series + noise  # Input data

fig = plt.figure(figsize=(200, 9))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(time, data, s=5, alpha=0.5, color="blue")
# ---------------------------------------------------------------------# RNN BUILD
n = 1  # Number of LSTM blocks
w_size = 5  # Window size
learnStep = 0.01
batchSize = 1

seq = np.array(data).reshape((1, -1))

L1 = w_size + n
L2 = n * 4

# ---------------------------------------------------------------------# PEEHOLES
mask = np.zeros((n, L2))
for i in range(n):
    mask[i, i * 4 + 1: i * 4 + 4] = 1
mask = np.float32(mask)
print mask
# ---------------------------------------------------------------------#
B = [0, 0, -2, 2]  # Some German magic for bias numbers
B = np.array(B)

# random = np.sqrt(6) / np.sqrt(L1 + L2)
random = np.sqrt(6) / (L1 + L2)  # For [-0.1; 0.1]

X = T.matrix('X')
Y = T.vector('Y')

w = theano.shared((np.random.randn(L2, L1) * 2 * random - random).astype(theano.config.floatX), name='w')
# b = theano.shared(np.tile(0.0, (L2,)).astype(theano.config.floatX), name = 'b')
b = theano.shared(B.astype(theano.config.floatX), name='b')
A = theano.shared(np.tile(0.0, (n, 1)).astype(theano.config.floatX), name='A')

# ---------------------------------------------------------------------# MASK USAGE
extX = T.zeros((L1, batchSize))
extX = T.set_subtensor(extX[: L1 - n, :], X)
extX = T.set_subtensor(extX[L1 - n:, :], A)
maskedW = T.set_subtensor(w.T[- n:, :], (w.T[- n:, :] * mask).astype(theano.config.floatX)).T
# ---------------------------------------------------------------------#

s = T.dot(maskedW, extX) + b.dimshuffle(0, 'x')
s = T.nnet.sigmoid(s)

num = T.shape(X)[1]

s = s.reshape((n, 4, num))

Pi = s[:, 0, :] * s[:, 1, :]
Pr = s[:, 2, :] * A
Au = Pi + Pr
Po = s[:, 3, :] * Au

cost = T.sqrt(T.sum(T.sqr(Po - Y)) / num)

gw, gb = T.grad(cost, [w, b])

ltsm = theano.function(inputs=[X, Y], outputs=[Po, cost],
                       updates=((A, Au), (w, w - learnStep * gw), (b, b - learnStep * gb)), allow_input_downcast=True)
# ---------------------------------------------------------------------# RNN CALC
predictions = []
markers = []
errors = []

for i in range(number - w_size):
    # RANDOM WINDOWS
    # idx = np.random.randint(0, seq.shape[1] - w_size, 1)
    # feed = seq[:, idx : idx + w_size].T
    # Y = seq[:, idx + w_size].reshape((-1))
    # END
    feed = seq[:, i: i + w_size].T
    Y = seq[:, i + w_size].reshape((-1))
    p, e = ltsm(feed, Y)
    predictions.append(p)
    markers.append(time[i + w_size])
    errors.append(e)
    print i, i + w_size
    print "P: ", p, "	", "Y: ", Y, "	", "E: ", e
# ---------------------------------------------------------------------# FINAL PICTURE
ax.scatter(markers, predictions, s=5, alpha=0.5, color="red")
ax.scatter(markers, errors, s=1, alpha=1, color="orange")
fig.savefig("fPictureFF.png")

# ---------------------------------------------------------------------# FEEDFORWARD
print "G"
ltsmFF = theano.function(inputs=[X], outputs=[Po], updates=[(A, Au)], allow_input_downcast=True)

gd = seq[:, number - w_size: number]
ga = []

for i in range(number):
    feed = gd[:, i: i + w_size].T
    out = ltsmFF(feed)
    print i, i + w_size, out
    ga.append(out)
    gd = np.append(gd, out).reshape((1, -1))
    print len(ga), gd.shape
print seq[:, number - w_size: number]
print gd[:, 0: 10]
# ---------------------------------------------------------------------# FINAL PICTURE 2
ax.scatter(time, ga, s=5, alpha=0.5, color="violet")
fig.savefig("fPictureFF2.png")
# ---------------------------------------------------------------------#