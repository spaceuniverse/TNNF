# ---------------------------------------------------------------------#
# External libraries
# ---------------------------------------------------------------------#


from matplotlib.pylab import *
from fTheanoNNclassCORE import *
import time
import random
import numpy as np
from fDataWorkerCORE import *


# ---------------------------------------------------------------------#

number = 10000
n = 5
w_size = 10
learnStep = 0.01
batchSize = 1

TIME, DATA = noisedSinGen(number=number)

print "Options:"

OPTIONS = newOptionsStore(learnStep=learnStep,
                          rmsProp=False,
                          rProp=False,
                          minibatch_size=batchSize)
OPTIONS.Printer()

L2 = LayerRNN(size_in=w_size,
              blocks=n,
              size_out=5,
              activation=FunctionModel.Sigmoid,
              peeholes=True)

print 'data: ' + str(DATA.shape)

seq = np.array(DATA).reshape((1, -1))

# ---------------------------------------------------------------------#

newNN = newTheanoNNclass(OPTIONS, (L2, ))

newNN.trainCompile()
newNN.predictCompile()

markers = []
pred = []

for i in range(number - w_size):
    feed = seq[:, i: i + w_size].T
    Y = seq[:, i + w_size].reshape((1, -1))
    pred.append(newNN.predictCalc(feed, debug=False).out)
    print i, newNN.out
    newNN.trainCalc(feed, Y, iteration=1, debug=False, errorCollect=True)
    markers.append(TIME[i + w_size])

fig = plt.figure(figsize=(200, 9))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(TIME, DATA, s=5, alpha=0.5, color="blue")
ax.scatter(markers, pred, s=5, alpha=0.5, color="red")
ax.scatter(markers, newNN.errorArray, s=5, alpha=0.5, color="orange")

fig.savefig("./aaaaaa.png")
"""
gd = seq[:, number - w_size: number]
ga = []

for i in range(number):
    feed = gd[:, i: i + w_size].T
    out = newNN.predictCalc(feed, debug=False).out
    print i, i + w_size, out
    ga.append(out)
    gd = np.append(gd, out).reshape((1, -1))
    print len(ga), gd.shape

ax.scatter(TIME, ga, s=5, alpha=0.5, color="violet")
fig.savefig("./aaaaaa2.png")
"""
print 'FINISH'
