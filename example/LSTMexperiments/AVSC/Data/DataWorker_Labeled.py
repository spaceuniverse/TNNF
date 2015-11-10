__author__ = 'rhrub'

import numpy as np
import h5py

class DataWorker(object):
    def __init__(self, size):
        #Indexes
        self.idx = None
        self.idx_old = None

        #batchsize
        self.size = size

        #A mask
        self.A_mask = None

        #data
        srcFolder = '/mnt/DATA/repo/AVSC/Data/'
        hdf_type = '.hdf5'
        target = 'testSequenceLabeled'

        f_train = h5py.File(srcFolder + target + hdf_type, 'r+')
        self.DATA = f_train['/hdfDataSet']

        self.DATA_shape = self.DATA.shape

        #init idx
        self.idx = np.random.randint(0, self.DATA_shape[0], (self.size,))
        self.idx_old = np.zeros_like(self.idx)

        #in sequence number
        self.sequence = np.zeros_like(self.idx)

    def getBatch(self):
        #A_mask
        self.A_mask = np.ones((self.size,))

        #batch
        batch = []
        for i in xrange(self.size):
            n = self.idx[i]
            s = self.sequence[i]
            b = self.DATA[n, :, s]
            batch.append(b)

        #Update idx_old
        self.idx_old[:] = self.idx[:]

        #After batch updates
        self.sequence += 1

        #Check whether sequence is still last
        last = []
        for i in xrange(self.size):
            n = self.idx[i]
            #add '1' to stop one value before end (last one will be a last label)
            s = self.sequence[i] + 1
            l = self.DATA[n, 0, s] == 0
            last.append(l)

        #If last.sum() != 0 = some sequence ended
        if np.sum(last) != 0:
            changed_idx = np.nonzero(last)
            for i in changed_idx:
                self.sequence[i] = 0
                self.idx[i] = np.random.randint(0, self.DATA_shape[0])

        #Compare whether idx changed
        if not np.array_equal(self.idx, self.idx_old):
            changed_idx = self.idx[:] != self.idx_old[:]
            self.A_mask[changed_idx] = 0

        return np.array(batch).T

    def getLabels(self):
        '''
        Method should be called AFTER getBatch().
        :return:
        labels for already returned batch
        '''

        labels = []
        for i in xrange(self.size):
            n = self.idx[i]
            s = self.sequence[i]
            l = self.DATA[n, -1, s]
            labels.append(l)

        return np.array(labels).T


def binarizer(arr, base):
    res = np.zeros((arr.shape[1], 1))
    
    count = 0
    for b in base:
        if b == 'skip':
            count += 1
        elif b:
            a = np.int64(arr[count, :])
            a = 1 & (a[:, np.newaxis] / 2 ** np.arange(b - 1, -1, -1))
            count += 1
            res = np.concatenate((res, a), axis=1)
        else:
            a = np.int64(arr[count, :]).reshape((-1, 1))
            res = np.concatenate((res, a), axis=1)
            count += 1
    
    res = res[:, 1:]

    return res.T


'''
iterations = 5
D = DataWorker(10)

for i in xrange(iterations):
    d = D.getBatch()
    l = D.getLabels()
    print i, d.shape, l.shape
    print D.A_mask
    c1 = binarizer(l)
    c2 = advanced_binarizer(l, (7, 10))
    
    print np.array_equal(c1, c2)
    
    
    #if D.A_mask.sum() != 10:
    #    break

'''







