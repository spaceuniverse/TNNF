__author__ = 'rhrub'

#---------------------------------------------------------------------#
# External libraries
#---------------------------------------------------------------------#
from matplotlib.pylab import *
from fTheanoNNclassCORE import *                # Some cool NN builder here
import time                                     # What time is it? Adven...
import random
import h5py
import numpy as np
#---------------------------------------------------------------------#
import Data.DataWorker
#---------------------------------------------------------------------#

#----------#
batchSize = 400
crossSize = 400
epochs = 100000
#----------#

#DATA
D = Data.DataWorker.DataWorker(batchSize)
#----------#
print "Options:"
OPTIONS = newOptionsStore(learnStep=0.01,
                          rProp=False,
                          rmsProp=0.9,
                          mmsmin=1e-15,
                          minibatch_size=batchSize,
                          CV_size=batchSize)

OPTIONS.Printer()
#----------#
L1 = LayerRNN(size_in=57,
              size_out=400,
              blocks=400,
              peeholes=True,
              activation=FunctionModel.Sigmoid)

L2 = LayerRNN(size_in=400,
              size_out=17,
              blocks=17,
              peeholes=True,
              activation=FunctionModel.Sigmoid)

#L1 = LayerNN(size_in=57,
#             size_out=19,
#             activation=FunctionModel.Sigmoid)

#----------#

RNN = newTheanoNNclass(OPTIONS, (L1, L2))

#----------#
RNN.trainCompile()
RNN.predictCompile()

for i in xrange(epochs):
    start = time.time()

    feed = D.getBatch()
    feed = Data.DataWorker.binarizer(feed, (8, 7, 10, 16, 16))
    print 'feed.shape:', feed.shape

    Y = D.getLabels().reshape((2, -1))
    Y = Data.DataWorker.binarizer(Y, (7, 10))
    print 'Y.shape:', Y.shape

    RNN.trainCalc(feed, Y, iteration=1, debug=True, errorCollect=True)
    RNN.predictCalc(feed, debug=False)

    stop = time.time()

    print str(i) + '\tIteration time: ' + str(stop - start)

    #Handle sequence change and clear A
    if D.A_mask.sum() != batchSize:

        print '### A DEBUG ###'
        #print L1.A.get_value().shape
        #print L1.A.get_value()
        print RNN.varArrayW[-1].get_value().shape
        print RNN.varArrayW[-1].get_value()
        print RNN.out.shape
        print RNN.out
        print '::::: ------ ::::::'

        #L1.A.set_value(np.float32(L1.A.get_value() * D.A_mask))

        # null A
        for k in xrange(RNN.lastArrayNum):
            RNN.architecture[k].A.set_value(np.float32(RNN.architecture[k].A.get_value() * D.A_mask).astype(theano.config.floatX))