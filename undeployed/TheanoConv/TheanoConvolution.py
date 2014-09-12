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
#------#


def getBatch(d, n):
    res = []
    size = np.sqrt(d.shape[1] - 1)
    idx = np.random.randint(0, d.shape[0], n)
    for i in range(n):
        p = d[idx[i], 1:].reshape((size, size))
        res.append(p.reshape((-1)))
    res = np.array(res)
    return res


def modelLoader(folder, layers):
    f = file(folder, "rb")
    loadedObjects = []
    for i in xrange(2 * layers):
        loadedObjects.append(cPickle.load(f))
    f.close()
    return loadedObjects
#------#

### DATA ###
srcFolder = '/home/rhrub/PycharmProjects/TheanoConv/Data/src/'
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

D = getBatch(DATA, 100)

#------#
### WEIGHTS ###
kernelModelName = './AE/AE_for_conv.txt'
kernelW = modelLoader(kernelModelName, 1)

kernelB = kernelW[1]
kernelW = kernelW[0]
#------#
### CONVOLUTION ###


X = T.matrix('X')
#W = T.matrix('W')
#b = T.matrix('b')
Wcnn = theano.shared(kernelW.astype(theano.config.floatX), name='Wcnn')
Bcnn = theano.shared(kernelB.astype(theano.config.floatX), name='Bcnn')

sX = T.cast(T.sqrt(T.shape(X)[1]), 'int16')
Xr = T.reshape(X, (T.shape(X)[0], 1, sX, sX))

sW = T.cast(T.sqrt(T.shape(Wcnn)[1]), 'int16')
Wr = T.reshape(Wcnn, (T.shape(Wcnn)[0], 1, sW, sW))

#Convolve
res = T.nnet.conv2d(Xr, Wr, border_mode='valid')

#Add bias
res = res + Bcnn.reshape((T.shape(Bcnn)[0],)).dimshuffle('x', 0, 'x', 'x')

#Sigmoid
res = 1 / (1 + T.exp(-res))

#Pooling
pool_shape = (3, 3)
res = downsample.max_pool_2d(res, pool_shape, ignore_border=True)

w1 = theano.shared(np.zeros((400, 100)).astype(theano.config.floatX), name = 'w1')

nShape = T.shape(res)[1] * T.shape(res)[2] * T.shape(res)[3]

res = T.dot(w1, res.reshape((T.shape(X)[0], nShape)))


cnn = theano.function(inputs=[X],
                      outputs=res,
                      allow_input_downcast=True)


print D.shape

#v = cnn(D, kernelW, kernelB)
v = cnn(D)

print v.shape


































'''
#Test filtered image

print 'Test img!'
print v[0, 0, :, :].shape

imsave = Image.fromarray(v[0, 0, :, :] * 256.0)
imsave = imsave.convert('L')
imsave.save('./AE/img/f.jpg', 'JPEG', quality=100)


imsave = Image.fromarray(D[0, :].reshape((28, 28)))
imsave = imsave.convert('L')
imsave.save('./AE/img/o.jpg', 'JPEG', quality=100)

'''
