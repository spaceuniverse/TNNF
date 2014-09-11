__author__ = 'rhrub'

### IMPORTS ###
from matplotlib.pylab import *
import theano.tensor as T
from theano.tensor.signal import downsample
import theano
import time					                    # What time is it? Adven...
import random
import h5py
import numpy as np
import cPickle
from PIL import Image, ImageOps, ImageFilter
from AE.fTheanoNNclassCORE import *
#------#


def rollOut(l):
    numClasses = 10
    n = l.shape[0]
    l = l.reshape((1, -1))
    l = np.tile(l, (numClasses, 1))
    g = np.array(range(numClasses)).reshape((-1, 1))
    g = np.tile(g, (1, n))
    res = l == g * 1.0
    return res


def getBatch(d, n):
    res = []
    lbl = []
    size = np.sqrt(d.shape[1] - 1)
    idx = np.random.randint(0, d.shape[0], n)
    for i in range(n):
        p = d[idx[i], 1:]
        res.append(p)
        lbl.append(d[idx[i], 0])
    res = np.array(res)
    lbl = np.array(lbl)
    return res / 256.0, lbl


def localModelLoader(folder, layers):
    f = file(folder, "rb")
    loadedObjects = []
    for i in xrange(2 * layers):
        loadedObjects.append(cPickle.load(f))
    f.close()
    return loadedObjects
#------#

### DATA ###
srcFolder = './Data/src/'
hdf_type = '.hdf5'
train_set = 'mnist_train'
test_set = 'mnist_test'

#DATA
f_train = h5py.File(srcFolder + train_set + hdf_type, 'r+')
DATA = f_train['/hdfDataSet']

#CV
f_test = h5py.File(srcFolder + test_set + hdf_type, 'r+')
DATA_CV = f_test['/hdfDataSet']

print 'Data shape:', DATA.shape, 'CV shape:', DATA_CV.shape

#------#
### WEIGHTS ###
kernelModelName = './AE/AE_for_conv.txt'
kernelW = localModelLoader(kernelModelName, 1)

kernelB = kernelW[1]
kernelW = kernelW[0]
#------#
### CONVOLUTION ###


X = T.matrix('X')
W = T.matrix('W')
b = T.matrix('b')

sX = T.cast(T.sqrt(T.shape(X)[1]), 'int16')
Xr = T.reshape(X, (T.shape(X)[0], 1, sX, sX))

sW = T.cast(T.sqrt(T.shape(W)[1]), 'int16')
Wr = T.reshape(W, (T.shape(W)[0], 1, sW, sW))

#Convolve
res = T.nnet.conv2d(Xr, Wr, border_mode='valid')

#Add bias
res = res + b.reshape((T.shape(b)[0],)).dimshuffle('x', 0, 'x', 'x')

#Sigmoid
res = 1 / (1 + T.exp(-res))

#Pooling
pool_shape = (3, 3)
res = downsample.max_pool_2d(res, pool_shape, ignore_border=True)


cnn = theano.function(inputs=[X, W, b],
                      outputs=res,
                      allow_input_downcast=True)

#v = cnn(D, kernelW, kernelB)

#------#

CV_size = 6000
dataSize = DATA.shape[0]
batchSize = 300

#------#
print "Options:"
OPTIONS = OptionsStore(learnStep=0.0005,
                       regularization=False,
                       sparsity=False,
                       sparsityParam=0.1,
                       beta=3,
                       lamda=1e-5,
                       rmsProp=True,
                       rProp=False,
                       decay=0.9,
                       dropout=True,
                       dropOutParam=(0.75, 0.5),
                       mmsmin=1e-15)
OPTIONS.Printer()

modelName = 'SM_autosave.txt'
print 'data: ' + str(DATA.shape)
print 'CV: ' + str(CV_size)
print 'Model name: ' + modelName

#------#

NN = TheanoNNclass((2401, 400, 10), OPTIONS, modelFunction=(FunctionModel.Sigmoid,
                                                            FunctionModel.SoftMax))

NN.trainCompile(batchSize)
NN.predictCompile(CV_size)

#------#

CVstep = 100
cv_err = []
acc = []

#------#
CV = DATA_CV[:CV_size, 1:]
CV /= 256.
CV = cnn(CV, kernelW, kernelB).reshape((CV_size, -1)).T

CV_Y = DATA_CV[:CV_size, 0]
labels = CV_Y
CV_Y = rollOut(CV_Y)

#------#
for i in xrange(2000000):
    start = time.time()

    feed, Y = getBatch(DATA, batchSize)

    #------#
    start_cnn = time.time()
    feed = cnn(feed, kernelW, kernelB).reshape((batchSize, -1)).T
    stop_cnn = time.time()
    print '\tCNN time: ' + str(round(stop_cnn - start_cnn, 3)) + ' sec'
    #------#

    Y = rollOut(Y)

    #print feed.shape, Y.shape

    NN.trainCalc(feed, Y, iteration=1, debug=True, errorCollect=True)

    stop = time.time()

    print str(i) + '\tIteration time: ' + str(stop - start)

    if i % CVstep == 0:

        E = NNsupport.crossV(CV_size, CV_Y, CV, NN)

        accuracy = np.true_divide((np.argmax(NN.out, axis=0) == labels).sum(), CV_size)

        print '\n *** CV ERROR: ' + str(E) + ' ***\n'
        print 'Accuracy: ' + str(round(accuracy * 100, 3)) + ' %'

        cv_err.append(E)
        acc.append(accuracy)

        #Plot errors
        train_axes = range(0, len(NN.errorArray))
        cv_axes = range(0, len(NN.errorArray), CVstep)

        plot(train_axes, NN.errorArray, 'r,', markeredgewidth=0, label='train')
        plot(cv_axes, cv_err, 'g.', markeredgewidth=0, label='CV')
        plot(cv_axes, acc, 'b.', markeredgewidth=0, label='Accuracy')

        title('Error vs epochs', fontsize=12)
        xlabel('epochs', fontsize=10)
        ylabel('Error', fontsize=10)
        legend(loc='upper right', fontsize=10, numpoints=3, shadow=True, fancybox=True)

        grid()
        margins(0.04)
        savefig('train_cv_error.png', dpi=120)
        close()

        #Save the model
        NN.modelSaver('./' + modelName)
print 'FINISH'















