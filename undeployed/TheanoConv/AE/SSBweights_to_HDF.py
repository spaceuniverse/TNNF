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


def modelSaver(folder, model):
    f = file(folder, "wb")
    for obj in model:
        cPickle.dump(obj, f, protocol = cPickle.HIGHEST_PROTOCOL)
    f.close()
    return


def modelLoader(folder, layers):
    f = file(folder, "rb")
    loadedObjects = []
    for i in xrange(2 * layers):
        loadedObjects.append(cPickle.load(f))
    f.close()
    return loadedObjects


#------#
OPTIONS = OptionsStore(learnStep=0.01,
                       mmsmin=1e-15)

NN = TheanoNNclass((64, 49, 64), OPTIONS, modelFunction=(FunctionModel.Sigmoid,
                                                         FunctionModel.Sigmoid))

modelName = 'AE_autosave.txt'


NN.modelLoader('./' + modelName)
NNL = NN.paramGetter()
NNL = NNL[:2]
#------#

for L in NNL:
    print L.shape

modelSaver('./AE_for_conv.txt', NNL)
