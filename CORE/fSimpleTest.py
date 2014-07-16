# ---------------------------------------------------------------------#
# External libraries
# ---------------------------------------------------------------------#


from matplotlib.pylab import *
from fTheanoNNclassCORE import *
import time
import random
import numpy as np
from fDataWorkerCORE import *
from fGraphBuilder import *


# ---------------------------------------------------------------------#

number = 1000
n = 1
w_size = 10
learnStep = 0.01
batchSize = 1

numofseq = 50   # number ob seq

print "Options:"

OPTIONS = newOptionsStore(learnStep=learnStep,
                          rmsProp=0.9,
                          rProp=False,
                          minibatch_size=batchSize)
OPTIONS.Printer()

L2 = LayerRNN(size_in=w_size,
              blocks=n,
              size_out=1,
              activation=FunctionModel.Sigmoid,
              peeholes=True)

# ---------------------------------------------------------------------#

newNN = newTheanoNNclass(OPTIONS, (L2, ))

newNN.trainCompile()
newNN.predictCompile()

for j in xrange(numofseq):

    TIME, DATA = noisedSinGen(number=number, phase=(20.0 * random.random()))
    print 'data: ' + str(DATA.shape)
    seq = np.array(DATA).reshape((1, -1))

    markers = []
    pred = []

    # null A
    for k in xrange(newNN.lastArrayNum):
        newNN.architecture[k].A.set_value(np.tile(0.0, (n, batchSize)).astype(theano.config.floatX))

    for i in range(number - w_size):
        feed = seq[:, i: i + w_size].T
        Y = seq[:, i + w_size].reshape((1, -1))
        pred.append(newNN.predictCalc(feed, debug=False).out)
        print i, newNN.out
        newNN.trainCalc(feed, Y, iteration=1, debug=True, errorCollect=True)
        markers.append(TIME[i + w_size])

fig = plt.figure(figsize=(200, 9))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(TIME, DATA, s=5, alpha=0.6, color="blue")
ax.scatter(markers, pred, s=5, alpha=0.6, color="red")
fig.savefig("./aaaaaaV.png")

fig = plt.figure(figsize=(200, 9))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(range(len(newNN.errorArray)), newNN.errorArray, s=5, alpha=0.9, color="orange")
fig.savefig("./aaaaaaG.png")


print "GB test"
Graph.Builder(test=pred)

print "SAVE"
newNN.modelSaver("./model.txt")
print "LOAD"
newNN.modelLoader("./model.txt")

print "AUTO"

TIME, DATA = noisedSinGen(number=number, phase=(20.0 * random.random()))
print 'data: ' + str(DATA.shape)
seq = np.array(DATA).reshape((1, -1))

gd = seq[:, number - w_size: number]
ga = []

for i in range(number):
    feed = gd[:, i: i + w_size].T
    out = newNN.predictCalc(feed, debug=False).out
    print i, i + w_size, out
    ga.append(out)
    gd = np.append(gd, out).reshape((1, -1))
    #print len(ga), gd.shape

fig = plt.figure(figsize=(200, 9))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(TIME, ga, s=5, alpha=0.9, color="violet")
fig.savefig("./aaaaaaAUTO.png")

print "FINISH"