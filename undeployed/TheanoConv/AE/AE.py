__author__ = 'rhrub'


#---------------------------------------------------------------------#
# External libraries
#---------------------------------------------------------------------#
from matplotlib.pylab import *
from fTheanoNNclassCORE import *				# Some cool NN builder here
import time					                    # What time is it? Adven...
import random
import h5py
import numpy as np
#---------------------------------------------------------------------#


def getBatch(d, n, s=(8, 8)):
    res = []
    size = np.sqrt(d.shape[1] - 1)
    idx = np.random.randint(0, d.shape[0], n)
    s_point = np.random.randint(0, size - s[1], 2 * n)
    for i in range(n):
        p = d[idx[i], 1:].reshape((size, size))
        r = p[s_point[i]:s_point[i] + s[0], s_point[n + i]:s_point[n + i] + s[1]]
        res.append(r.reshape((-1)))
    res = np.array(res)
    return res


### DATA ###
srcFolder = '../Data/src/'
hdf_type = '.hdf5'
train_set = 'mnist_train'
test_set = 'mnist_test'

#DATA
f_train = h5py.File(srcFolder + train_set + hdf_type, 'r+')
DATA = f_train['/hdfDataSet']

#CV
f_test = h5py.File(srcFolder + test_set + hdf_type, 'r+')
DATA_CV = f_test['/hdfDataSet']


CV_size = 5000
dataSize = DATA.shape[0]
batchSize = 300
w_size = (6, 6)

CV = getBatch(DATA_CV, CV_size, w_size).T / 256.0

#------#
print "Options:"
OPTIONS = OptionsStore(learnStep=0.01,
                       regularization=False,
                       sparsity=True,
                       sparsityParam=0.1,
                       beta=3,
                       lamda=1e-5,
                       rmsProp=True,
                       rProp=False,
                       decay=0.9,
                       dropout=False,
                       dropOutParam=(0.75, 0.5),
                       mmsmin=1e-15)
OPTIONS.Printer()

modelName = 'AE_autosave_6x6.txt'
print 'data: ' + str(DATA.shape)
print 'CV: ' + str(CV_size)
print 'Model name: ' + modelName

#---------------------------------------------------------------------#
NN = TheanoNNclass((36, 32, 36), OPTIONS, modelFunction=(FunctionModel.Sigmoid,
                                                         FunctionModel.Sigmoid))


#NN.localModelLoader('./' + modelName)
NN.trainCompile(batchSize)
NN.predictCompile(CV_size)
#------#

CVstep = 500
cv_err = []

#---------------------------------------------------------------------#

for i in xrange(2000000):
    start = time.time()

    feed = getBatch(DATA, batchSize, w_size).T / 256.0
    NN.trainCalc(feed, feed, iteration=1, debug=True, errorCollect=True)

    stop = time.time()

    print str(i) + '\tIteration time: ' + str(stop - start)

    if i % CVstep == 0:

        E = NNsupport.crossV(CV_size, CV, CV, NN)

        print '\n *** CV ERROR: ' + str(E) + ' ***\n'
        cv_err.append(E)

        #Plot errors
        train_axes = range(0, len(NN.errorArray))
        cv_axes = range(0, len(NN.errorArray), CVstep)

        plot(train_axes, NN.errorArray, 'r,', markeredgewidth=0, label='train')
        plot(cv_axes, cv_err, 'g.', markeredgewidth=0, label='CV')

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
        NN.weightsVisualizer('./weights/', size=(6, 6))

print 'FINISH'