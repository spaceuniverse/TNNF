# ---------------------------------------------------------------------#
# External libraries
# ---------------------------------------------------------------------#


import numpy as np
import os
import sys
sys.path.append('./CORE')
sys.path.append('../CORE')
from fTheanoNNclassCORE import OptionsStore, TheanoNNclass, NNsupport, FunctionModel, LayerNN
from fDataWorkerCORE import csvDataLoader, DataMutate


# ---------------------------------------------------------------------#

batchSize = 200

options = OptionsStore(learnStep=0.001,
                       rmsProp=0.9,
                       mmsmin=1e-20,
                       minibatch_size=batchSize)
options.Printer()

iterations = 5000  # 900000 for awesome features
numberOfFeatures = 512
imageSize = (28, 28)

datasetFolder = os.path.join(os.path.dirname(__file__), "../mnist/")

trainData = csvDataLoader(datasetFolder + "train.csv", startColumn=1)
trainData.X = DataMutate.deNormalizer(trainData.X, afterzero=8)
print trainData.X.shape, trainData.Y.shape

L1 = LayerNN(size_in=trainData.input,
             size_out=numberOfFeatures,
             sparsity=0.05,
             beta=7,
             weightDecay=3e-7,
             activation=FunctionModel.Sigmoid)

L2 = LayerNN(size_in=numberOfFeatures,
             size_out=trainData.input,
             weightDecay=3e-7,
             activation=FunctionModel.Sigmoid)

AE = TheanoNNclass(options, (L1, L2))

AE.trainCompile()

for i in xrange(iterations):
    X, index = trainData.miniBatch(batchSize)
    AE.trainCalc(X, X, iteration=1, debug=True)
    print i

if os.path.exists(datasetFolder + "out/"):
    shutil.rmtree(datasetFolder + "out/")

os.makedirs(datasetFolder + "out/")
os.makedirs(datasetFolder + "out/w/")

AE.modelSaver(datasetFolder + "out/" + "/AE_NEW_VERSION_MNIST.txt")
AE.modelLoader(datasetFolder + "out/" + "/AE_NEW_VERSION_MNIST.txt")
AE.weightsVisualizer(datasetFolder + "out/w/", size=imageSize, color="L", second=False)

print "FINISH"


# ---------------------------------------------------------------------#