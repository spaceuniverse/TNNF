import numpy as np
import unittest
import os
import sys
import h5py
import random
sys.path.append('../../../CORE')
from fTheanoNNclassCORE import OptionsStore, TheanoNNclass, NNsupport, FunctionModel, LayerNN
from fGraphBuilderCORE import Graph

def getBatch(d, n, i):
    res = []
    idx = random.sample(i, n)
    idx = np.sort(idx)
    res = d[idx, 1:]
    return res.T / 255.0

#We use HDF because of its speed and convenience
#Set data's file names and path
srcFolder = './Data/src/'
hdf_type = '.hdf5'
train_set = 'mnist_train'
test_set = 'mnist_test'

#Read train data
f_train = h5py.File(srcFolder + train_set + hdf_type, 'r+')
DATA = f_train['/hdfDataSet']

#Read CV data
f_test = h5py.File(srcFolder + test_set + hdf_type, 'r+')
DATA_CV = f_test['/hdfDataSet']

#Print out shapes of loaded data
print 'Data shape:', DATA.shape, '\n', 'CV shape:', DATA_CV.shape

#Extract some useful data
dataSize = DATA.shape[0]
cvSize = DATA_CV.shape[0]
validDataIndexes = xrange(0, dataSize)

# As we have all data we need for Auto Encoder (AE),
# let's create an appropriate NN

# Set few additional options
numberOfFeatures = 196
batchSize = 200
inputSize = DATA.shape[1] - 1   # Subtract label
iterations = 10000
checkCvEvery = 500

#Common options for whole NN
options = OptionsStore(learnStep=0.005,
                       rmsProp=0.9,
                       mmsmin=1e-20,
                       minibatch_size=batchSize,
                       CV_size=cvSize)

#First layer
L1 = LayerNN(size_in=inputSize,
             size_out=numberOfFeatures,
             sparsity=0.1,
             beta=3,
             weightDecay=3e-3,
             activation=FunctionModel.Sigmoid)

#Second layer
L2 = LayerNN(size_in=numberOfFeatures,
             size_out=inputSize,
             weightDecay=3e-3,
             activation=FunctionModel.Sigmoid)

#Compile all together
AE = TheanoNNclass(options, (L1, L2))

AE.trainCompile()
AE.predictCompile()

X_CV = DATA_CV[:, 1:].T / 255.0
#X = DATA[:, 1:].T / 255.0

CV_error = []

for i in xrange(iterations):

    X = getBatch(DATA, batchSize, validDataIndexes)

    AE.trainCalc(X, X, iteration=1, debug=True, errorCollect=True)

    if i % checkCvEvery == 0:

        CV_error.append(NNsupport.crossV(X_CV, X_CV, AE))

        print 'CV error: ', CV_error[-1]

        #Draw how error and accuracy evolves vs iterations
        Graph.Builder(name='AE_error.png', error=AE.errorArray, cv=CV_error, legend_on=True)

        AE.weightsVisualizer(folder='weights', size=(28, 28))