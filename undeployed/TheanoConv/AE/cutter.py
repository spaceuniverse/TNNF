from atk import Image
#---------------------------------------------------------------------#
# External libraries
#---------------------------------------------------------------------#
import time					                    # What time is it? Adven...
from PIL import Image, ImageOps, ImageFilter
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

#------#
batchSize = 100

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

d = getBatch(DATA, batchSize)
print d.shape


#imsave = Image.fromarray(DATA[5, 1:].reshape((28, 28)))
imsave = Image.fromarray(d[5, :].reshape((8, 8)))
imsave = imsave.convert('L')
imsave.save('./img/t.jpg', 'JPEG', quality=100)

