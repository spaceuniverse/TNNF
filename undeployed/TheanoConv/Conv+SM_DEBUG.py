### IMPORTS ###
from matplotlib.pylab import *
import pylearn2
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import time					                    # What time is it? Adven...
import random
import h5py
import numpy as np
import cPickle
from PIL import Image, ImageOps, ImageFilter
#------#



#Saves list of objects in cPickle format
def localModelSaver(folder, model):
    f = file(folder, "wb")
    for obj in model:
        cPickle.dump(obj, f, protocol = cPickle.HIGHEST_PROTOCOL)
    f.close()
    return


#Convert matrix with classes in sparse matrix. Say class "3" => "0010"; in case we have only 4 classes.
def rollOut(l):
    numClasses = 10
    n = l.shape[0]
    l = l.reshape((1, -1))
    l = np.tile(l, (numClasses, 1))
    g = np.array(range(numClasses)).reshape((-1, 1))
    g = np.tile(g, (1, n))
    res = l == g * 1.0
    return res


#custom getBatch function
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


#loads objects from cPickle in list
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
#pre-trained AE's weights
kernelModelName = './AE/AE_for_conv.txt'
kernelW = localModelLoader(kernelModelName, 1)

kernelB = kernelW[1].reshape((-1,))
kernelW = kernelW[0]
#------#
#Common configuration
learnStep = 0.001
regularization = False
sparsity = True
sparsityParam = 0.1
beta = 3
lamda = 1e-5
rmsProp = True
rProp = False
decay = 0.9
dropout = True
dropOutParam = (0.5, 0.5)
dropoutCnnParam = 0.75
mmsmin = 1e-15
pool_shape = (2, 2)

#------#
### CONVOLUTION ###


X = T.matrix('X')
Y = T.matrix('Y')

#Convolution part architecture. L1cnn = 36 and L2cnn = 25 - means we use 6x6 window and it results in 25-length vector
L1cnn = 49
L2cnn = 36


#------#
#Calc some shapes
#Calc shape of internal convolutional matrix. Needed for proper reshape
dataShape = (28, 28)
convSize = int(np.sqrt(L1cnn))
sizeBeforePooling = 28 - convSize + 1
sizeAfterPooling = sizeBeforePooling / pool_shape[0]
#------#

#Standard NN's architecture. This NN classifies convolution output
L1 = sizeAfterPooling ** 2 * L2cnn
L2 = 400
L3 = 10

#------#
print 'Kernel shape: (%s, %s), (%s,)' % (L2cnn, L1cnn, L2cnn)
print 'Network arch: \nL1 = %s\nL2 = %s\nL3 = %s\n' % (L1, L2, L3)
#------#

#values to init random weights. Different for NN and its cnn parts
random_cnn = sqrt(6) / sqrt(L1cnn + L2cnn)
random = sqrt(6) / sqrt(L1 + L3)


#Loads pre-trained AE's weights
#Wcnn = theano.shared(kernelW.astype(theano.config.floatX), name='Wcnn')
#Bcnn = theano.shared(kernelB.astype(theano.config.floatX), name='Bcnn')

#Init with random weights for CNN
Wcnn = theano.shared((np.random.randn(L2cnn, L1cnn) * 2 * random_cnn - random_cnn).astype(theano.config.floatX), name='Wcnn')
Bcnn = theano.shared(np.tile(0.0, (L2cnn,)).astype(theano.config.floatX), name='Bcnn')

#Calc shapes for reshape function on-the-fly.
sX = T.cast(T.sqrt(T.shape(X)[1]), 'int16')

#Converts input from 2 to 4 dimensions
Xr = T.reshape(X, (T.shape(X)[0], 1, sX, sX))

#Same reshape magic for weights
sW = T.cast(T.sqrt(T.shape(Wcnn)[1]), 'int16')
Wr = T.reshape(Wcnn, (T.shape(Wcnn)[0], 1, sW, sW))


#Dropout mask for CNN kernel
mcnn = 1
if dropout:
    srng = RandomStreams()
    mcnn = srng.binomial(p=dropoutCnnParam, size=(convSize, convSize)).astype(theano.config.floatX)


#Convolve
#Name convention:
#res - train
#res_p - predict

#Mul weights with dropout matrix. Dirty hack needs to be rechecked
res = T.nnet.conv2d(Xr, Wr * mcnn.dimshuffle('x', 'x', 0, 1), border_mode='valid')
res_p = T.nnet.conv2d(Xr, Wr * dropoutCnnParam, border_mode='valid')

#Add bias
res = res + Bcnn.dimshuffle('x', 0, 'x', 'x')
res_p = res_p + Bcnn.dimshuffle('x', 0, 'x', 'x')

#Sigmoid
res = T.nnet.sigmoid(res)
res_p = T.nnet.sigmoid(res_p)


#Calc avg activation for convolution results
sprs = 0
if sparsity:
    sparse = T.mean(res, axis=(0, 2, 3))
    epsilon = 1e-20
    sparse = T.clip(sparse, epsilon, 1 - epsilon)
    KL = T.sum(sparsityParam * T.log(sparsityParam / sparse) +
               (1 - sparsityParam) * T.log((1 - sparsityParam) / (1 - sparse)))
    sprs = KL * beta

#Pooling
res = downsample.max_pool_2d(res, pool_shape, ignore_border=True)
res_p = downsample.max_pool_2d(res_p, pool_shape, ignore_border=True)

#Separate function if U want to estimate output just after pooling
#cnn = theano.function(inputs=[X], outputs=res, allow_input_downcast=True)

#------#

CV_size = 6000
dataSize = DATA.shape[0]
batchSize = 512

#------#

modelName = 'Conv+SM_autosave.txt'
print 'data: ' + str(DATA.shape)
print 'CV: ' + str(CV_size)
print 'Model name: ' + modelName

#------#

#Standrd NN's weights init
w1 = theano.shared((np.random.randn(L2, L1) * 2 * random - random).astype(theano.config.floatX), name = 'w1')
b1 = theano.shared(np.tile(0.0, (L2,)).astype(theano.config.floatX), name = 'b1')

w2 = theano.shared((np.random.randn(L3, L2) * 2 * random - random).astype(theano.config.floatX), name = 'w2')
b2 = theano.shared(np.tile(0.0, (L3,)).astype(theano.config.floatX), name = 'b2')

# --- dropout --- #
m1 = 1
m1_p = 1
m2 = 1
m2_p = 1
if dropout:
    m1 = srng.binomial(p=dropOutParam[0], size=(L1,)).astype(theano.config.floatX)
    m1_p = dropOutParam[0]
    m2 = srng.binomial(p=dropOutParam[1], size=(L2,)).astype(theano.config.floatX)
    m2_p = dropOutParam[1]

# --- train ---
nShape = T.shape(res)[1] * T.shape(res)[2] * T.shape(res)[3]
numClasses = T.shape(w2)[0]

z1 = T.dot(w1, res.reshape((T.shape(X)[0], nShape)).T * m1.dimshuffle(0, 'x')) + b1.dimshuffle(0, 'x')
a1 = T.nnet.sigmoid(z1)

z2 = T.dot(w2, a1 * m2.dimshuffle(0, 'x')) + b2.dimshuffle(0, 'x')
z_max = T.max(z2, axis=0)
a2 = T.exp(z2 - T.log(T.dot(T.alloc(1.0, numClasses, 1), [T.sum(T.exp(z2 - z_max), axis=0)])) - z_max)

# --- predict ---
z1_p = T.dot(w1 * m1_p, res_p.reshape((T.shape(X)[0], nShape)).T) + b1.dimshuffle(0, 'x')
a1_p = T.nnet.sigmoid(z1_p)

z2_p = T.dot(w2 * m2_p, a1_p) + b2.dimshuffle(0, 'x')
z_max = T.max(z2_p, axis=0)
a2_p = T.exp(z2_p - T.log(T.dot(T.alloc(1.0, numClasses, 1), [T.sum(T.exp(z2_p - z_max), axis=0)])) - z_max)


XENT = 1.0 / batchSize * T.sum((Y - a2) ** 2 * 0.5)

cost = XENT + sprs

gw1, gb1, gw2, gb2, gWcnn, gBcnn = T.grad(cost, [w1, b1, w2, b2, Wcnn, Bcnn])


# --- RMS variable ---
w1_mmsp = theano.shared(np.tile(0.0, (L2, L1)).astype(theano.config.floatX), name='w1_mmsp')
b1_mmsp = theano.shared(np.tile(0.0, (L2,)).astype(theano.config.floatX), name='b1_mmsp')

w2_mmsp = theano.shared(np.tile(0.0, (L3, L2)).astype(theano.config.floatX), name='w2_mmsp')
b2_mmsp = theano.shared(np.tile(0.0, (L3,)).astype(theano.config.floatX), name='b2_mmsp')

Wcnn_mmsp = theano.shared(np.tile(0.0, (L2cnn, L1cnn)).astype(theano.config.floatX), name='Wcnn_mmsp')
Bcnn_mmsp = theano.shared(np.tile(0.0, (L2cnn,)).astype(theano.config.floatX), name='Bcnn_mmsp')


# --- RMS calc ---
w1_mmsn = decay * w1_mmsp + (1 - decay) * gw1 ** 2
w1_mmsn = T.clip(w1_mmsn, mmsmin, 1e+20)

b1_mmsn = decay * b1_mmsp + (1 - decay) * gb1 ** 2
b1_mmsn = T.clip(b1_mmsn, mmsmin, 1e+20)

w2_mmsn = decay * w2_mmsp + (1 - decay) * gw2 ** 2
w2_mmsn = T.clip(w2_mmsn, mmsmin, 1e+20)

b2_mmsn = decay * b2_mmsp + (1 - decay) * gb2 ** 2
b2_mmsn = T.clip(b2_mmsn, mmsmin, 1e+20)

Wcnn_mmsn = decay * Wcnn_mmsp + (1 - decay) * gWcnn ** 2
Wcnn_mmsn = T.clip(Wcnn_mmsn, mmsmin, 1e+20)

Bcnn_mmsn = decay * Bcnn_mmsp + (1 - decay) * gBcnn ** 2
Bcnn_mmsn = T.clip(Bcnn_mmsn, mmsmin, 1e+20)

# --- w,b update ---
w1_u = learnStep * gw1 / w1_mmsn ** 0.5
b1_u = learnStep * gb1 / b1_mmsn ** 0.5
w2_u = learnStep * gw2 / w2_mmsn ** 0.5
b2_u = learnStep * gb2 / b2_mmsn ** 0.5
Wcnn_u = learnStep * gWcnn / Wcnn_mmsn ** 0.5
Bcnn_u = learnStep * gBcnn / Bcnn_mmsn ** 0.5

train = theano.function(inputs=[X, Y],
                        outputs=[XENT, cost, a2],
                        updates=((w1, w1 - w1_u),
                                 (b1, b1 - b1_u),
                                 (w2, w2 - w2_u),
                                 (b2, b2 - b2_u),
                                 (Wcnn, Wcnn - Wcnn_u),
                                 (Bcnn, Bcnn - Bcnn_u),
                                 (w1_mmsp, w1_mmsn),
                                 (b1_mmsp, b1_mmsn),
                                 (w2_mmsp, w2_mmsn),
                                 (b2_mmsp, b2_mmsn),
                                 (Wcnn_mmsp, Wcnn_mmsn),
                                 (Bcnn_mmsp, Bcnn_mmsn),
                                 ),
                         allow_input_downcast=True
                        )

predict = theano.function(inputs=[X],
                          outputs=a2_p,
                          allow_input_downcast=True)

#------#
#Load custom model
'''
model = localModelLoader('./' + modelName, 3)

Wcnn.set_value(model[0])
Bcnn.set_value(model[1])
w1.set_value(model[2])
b1.set_value(model[3])
w2.set_value(model[4])
b2.set_value(model[5])
print 'Custom model loaded!'
'''
#Model loaded
#------#

CVstep = 100
cv_err = []
acc = []
errorArray = []

#------#
#------#
CV = DATA_CV[:CV_size, 1:]
CV /= 256.


CV_Y = DATA_CV[:CV_size, 0]
labels = CV_Y
CV_Y = rollOut(CV_Y)
#------#



#------#
for i in xrange(2000000):
    start = time.time()

    feed, Y = getBatch(DATA, batchSize)

    Y = rollOut(Y)

    #print feed.shape, Y.shape

    err, cost, out = train(feed, Y)
    errorArray.append(err)

    stop = time.time()
    print 'Cost:', cost
    print str(i) + '\tIteration time: ' + str(stop - start)

    if i % CVstep == 0:

        P = predict(CV)
        E = 1.0 / CV_size * np.sum((CV_Y - P) ** 2 * 0.5)

        accuracy = np.true_divide((np.argmax(P, axis=0) == labels).sum(), CV_size)

        print '\n *** CV ERROR: ' + str(E) + ' ***\n'
        print 'Accuracy: ' + str(round(accuracy * 100, 3)) + ' %'

        cv_err.append(E)
        acc.append(accuracy)

        #Plot errors
        train_axes = range(0, len(errorArray))
        cv_axes = range(0, len(errorArray), CVstep)

        plot(train_axes, errorArray, 'r,', markeredgewidth=0, label='train')
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
        model = []
        model.append(Wcnn.get_value())
        model.append(Bcnn.get_value())
        model.append(w1.get_value())
        model.append(b1.get_value())
        model.append(w2.get_value())
        model.append(b2.get_value())
        localModelSaver('./' + modelName, model)
print 'FINISH'
