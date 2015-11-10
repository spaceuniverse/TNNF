# ---------------------------------------------------------------------# imports


import random
import numpy as np
from numpy import *
import matplotlib.pylab as plt

from sinG import *
from sinG2 import *

from TNNF.fTheanoNNclassCORE import *
from TNNF.fGraphBuilderCORE import Graph


# ---------------------------------------------------------------------# data

data1 = []
data2 = []

for i in xrange(2000):
    data1.append(noisedSin())
    data2.append(noisedSin2())

print len(data1), len(data2)
# ---------------------------------------------------------------------# debug

fig = plt.figure(figsize=(200, 9))
ax = fig.add_subplot(1, 1, 1)

'''
fig = plt.figure(figsize=(200, 9))
ax = fig.add_subplot(1, 1, 1)

zipped = zip(data1, data2)

for sq in zipped:
    d1, d2 = sq
    y1, x1 = d1
    y2, x2 = d2
    colors = np.random.rand(3, 1)
    # print colors.shape
    ax.scatter(x1, y1, s=5, alpha=0.5, color=colors)
    ax.scatter(x2, y2, s=5, alpha=0.5, color=colors)
    fig.savefig("main.png")
'''

# ---------------------------------------------------------------------# LSTM

EPOCHS = 3
WINDOW = 5

print "Options:"

OPTIONS = OptionsStore(learnStep=0.0001,   #00
                       #rmsProp=0.9,
                       rmsProp=False,
                       #mmsmin=1e-6,
                       minibatch_size=1,
                       CV_size=1)

OPTIONS.Printer()

'''
L1 = LayerRNN(size_in=10,   # WINDOW
              size_out=3,
              blocks=3,
              peeholes=True,
              activation=FunctionModel.Sigmoid)

L2 = LayerNN(size_in=3,
             size_out=1,
             activation=FunctionModel.Linear)

RNN = TheanoNNclass(OPTIONS, (L1, L2))
'''

L1 = LayerRNN(size_in=5,   # WINDOW
              size_out=1,
              blocks=1,
              peeholes=True,
              activation=FunctionModel.Sigmoid)

RNN = TheanoNNclass(OPTIONS, (L1,))

RNN.trainCompile()
RNN.predictCompile()

originals = []
predicted = []
accumulate = []

for iteration in xrange(EPOCHS):
    start = time.time()
    #
    print "Data"
    data = data2
    label = np.array([1, 0]).reshape((2, -1))
    #
    rnd = random.randint(0, len(data) - 1)
    data = data[rnd][0]
    #
    # print data.shape, data, data.shape[0], label
    #
    # startN = random.randint(0, data.shape[0] - 1)
    # endN = random.randint(startN, data.shape[0] - 1)
    startN = random.randint(0, 1000)
    endN = random.randint(startN + 1000, data.shape[0] - 1)
    print "Start point:", startN, "End point:", endN
    #
    data = data[startN: endN]
    #data = data
    print data.shape
    #
    for i in xrange(data.shape[0] - WINDOW - 1):
        accumulate.append(RNN.architecture[0].A.get_value()[0])
        feed = np.transpose(data[i:i + WINDOW].reshape((1, -1)))
        label = np.array(data[i + WINDOW]).reshape((1, -1))
        #label = np.array(data[i + WINDOW + 1]).reshape((1, -1))
        #
        # print label, label.shape
        #
        # print feed, feed.shape
        #
        RNN.trainCalc(feed, label, iteration=1, debug=True, errorCollect=True)
        out = RNN.predictCalc(feed, debug=False).out
        print label, out
        originals.append(label)
        predicted.append(out)
    ###
    ax.scatter(range(len(originals)), originals, s=5, alpha=0.5, color="blue")
    ax.scatter(range(len(predicted)), predicted, s=5, alpha=0.5, color="orange")
    fig.savefig("~sinonlytrain.png")
    ###
    stop = time.time()
    print str(iteration) + "\titeration time: " + str(stop - start)
    Graph.Builder(name="~error.png", error=RNN.errorArray, legend_on=True)
    Graph.Builder(name="~accumulate.png", accuracy=accumulate, legend_on=True)
    Graph.Builder(name="~error2.png", error=RNN.errorArray, legend_on=True)
    print "null A"
    # DEBUG
    '''
    for k in xrange(RNN.lastArrayNum):
        if isinstance(RNN.architecture[k], LayerRNN):
            print k, RNN.architecture[k].A.get_value()
    w0 = RNN.paramGetter()[0]['w']
    w1 = RNN.paramGetter()[1]['w']
    print np.sum(w0) + np.sum(w1)
    '''
    #
    for k in xrange(RNN.lastArrayNum):
        if isinstance(RNN.architecture[k], LayerRNN):
            RNN.architecture[k].A.set_value(np.float32(RNN.architecture[k].A.get_value() * 0).astype(theano.config.floatX))
    #
    for k in xrange(RNN.lastArrayNum):
        if isinstance(RNN.architecture[k], LayerRNN):
            RNN.architecture[k].A_predict.set_value(np.float32(RNN.architecture[k].A_predict.get_value() * 0).astype(theano.config.floatX))
    print "------"
RNN.modelSaver("~model")
print "DONE"
# ---------------------------------------------------------------------#