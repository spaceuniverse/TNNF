#---------------------------------------------------------------------#
# External libraries
#---------------------------------------------------------------------#
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from numpy import *
from numpy import dot, sqrt, diag
from numpy.linalg import eigh
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import theano.tensor as T
import cPickle
import time
import matplotlib.pyplot as plt
from fCutClassCORE import *
#---------------------------------------------------------------------#
# Data mutation and calculation functions
#---------------------------------------------------------------------#
class DataMutate(object):
    @staticmethod
    def deNormalizer(ia, afterzero = 20):
        ia = np.array(ia)
        ia = np.around(ia / 255.0, decimals = afterzero)
        return ia
    @staticmethod
    def Normalizer(ia):
        min = np.min(ia)
        max = np.max(ia)
        koeff = 255 / (max - min)
        ia = (ia - min) * koeff
        return ia
    @staticmethod
    def PCAW(X, epsilon = 0.01):
        M = X.mean(axis = 0)
        X = X - M
        C = dot(X, X.T)
        U, S, V = linalg.svd(C)
        ex1 = diag(1.0 / sqrt(S + epsilon))
        ex2 = dot(U, ex1)
        ex3 = dot(ex2, U.T)
        xPCAWhite = dot(ex3, X)
        return xPCAWhite
#---------------------------------------------------------------------#
# Support functions
#---------------------------------------------------------------------#
class Graphic(object):
    @staticmethod
    def PicSaver(img, folder, name, color = "L"):
        imsave = Image.fromarray(DataMutate.Normalizer(img))
        imsave = imsave.convert(color)
        imsave.save(folder + name + ".jpg", "JPEG", quality = 100)
#---------------------------------------------------------------------#
# Data workers
#---------------------------------------------------------------------#
class BatchMixin(object):
    REPORT = "OK"
    def miniBatch(self, number):
        minIndex = np.random.randint(0, self.number, number)
        self.miniX = self.X[:, minIndex]
        return self.miniX, minIndex
#---------------------------------------------------------------------#
class cPicleDataLoader(BatchMixin):			
    def __init__(self, folder):
        dataload = open(folder, "rb")
        data = cPickle.load(dataload)
        dataload.close()
        data = np.array(data)
        data = data.astype('float')
        self.X = data.T
        self.number = len(data)
        self.input = len(self.X)
#---------------------------------------------------------------------#
class csvDataLoader(BatchMixin):				
    def __init__(self, folder, startColumn = 1, skip = 1):
        data = np.loadtxt(open(folder, "rb"), delimiter = ",", skiprows = skip)
        data = data.astype('float')
        if len(data.shape) == 1:
            data = np.reshape(data, (data.shape[0], 1))
        self.X = data[:, startColumn:].T
        self.Y = data[:, 0:startColumn].T
        self.number = len(data)
        self.input = len(self.X)
#---------------------------------------------------------------------#
class multiData(BatchMixin):				
    def __init__(self, *objs):
        xtuple = ()
        ytuple = ()
        for obj in objs:
            xtuple += (obj.X,)
            ytuple += (obj.Y,)
        self.X = np.concatenate(xtuple, axis = 1)
        self.Y = np.concatenate(ytuple, axis = 1)
        self.number = self.X.shape[1]
        self.input = self.X.shape[0]
#---------------------------------------------------------------------#
# Activation functions
#---------------------------------------------------------------------#
class FunctionModel(object):
    @staticmethod
    def Sigmoid(W, X, B, E, *args):
        z = T.dot(W, X) + T.dot(B, E)
        a = 1 / (1 + T.exp(-z))
        return a
    @staticmethod
    def Tanh(W, X, B, E, *args):
        z = T.dot(W, X) + T.dot(B, E)
        a = (T.exp(z) - T.exp(-z)) / (T.exp(z) + T.exp(-z))
        return a
    @staticmethod
    def SoftMax(W, X, B, E, *args):
        z = T.dot(W, X) + T.dot(B, E)
        numClasses = T.shape(W)[0]
        z_max = T.max(z, axis = 0)
        a = T.exp(z - T.log(T.dot(T.alloc(1.0, numClasses, 1), [T.sum(T.exp(z - z_max), axis = 0)])) - z_max)
        return a
    @staticmethod
    def MaxOut(W, X, B, E, *args):
        numPatches = E.get_value().shape[1]
        A = np.zeros((1, numPatches)).astype(theano.config.floatX)
        neurons = T.arange(T.shape(W)[0])
        components, updates = theano.scan(fn = lambda n, weights, bias, data, activation: T.max(T.concatenate([data.T * T.dot(T.shape_padleft(weights[n, :]).T, E).T, (bias[n, 0] * E).T], axis = 1), axis = 1), sequences = neurons, non_sequences = [W, B, X, A], outputs_info = None)
        return components
#---------------------------------------------------------------------#
# Options instance
#---------------------------------------------------------------------#
class OptionsStore(object):
    def __init__(self, learnStep = 0.01, regularization = False, lamda = 0.001, sparsity = False, sparsityParam = 0.01, beta = 0.01, dropout = False, dropOutParam = (1, 1), rmsProp = False, decay = 0.9, mmsmin = 1e-10, rProp = False):
        self.learnStep = learnStep		# Learning step for gradient descent
        self.regularization = regularization	# Weight decay on|off
        self.lamda = lamda			# Weight decay coef
        self.sparsity = sparsity		# Sparsity on|off
        self.sparsityParam = sparsityParam
        self.beta = beta
        self.dropout = dropout		# Dropout on|off
        self.dropOutParam = dropOutParam		# dropOutParam = (1, 0.7, 0,5) etc.
        self.rmsProp = rmsProp		# rmsProp on|off
        self.decay = decay
        self.mmsmin = mmsmin			# Min mms value
        self.rProp = rProp			# For full batch only
    def Printer(self):
        print self.__dict__
#---------------------------------------------------------------------#
# Basic neuralnet class
#---------------------------------------------------------------------#
class TheanoNNclass(object):				
    REPORT = "OK"
    def __init__(self, architecture, options, modelFunction = (FunctionModel.Sigmoid, FunctionModel.Sigmoid)):
        self.architecture = architecture
        self.options = options
        self.modelFunction = modelFunction
        self.lastArrayNum = len(architecture) - 1
        random = sqrt(6) / sqrt(architecture[0] + architecture[self.lastArrayNum])
        self.varArrayW = []
        self.varArrayB = []
        for i in xrange(self.lastArrayNum):
            w = theano.shared((np.random.randn(architecture[i + 1], architecture[i]) * 2 * random - random).astype(theano.config.floatX), name = "w%s" % (i + 1))
            self.varArrayW.append(w)
            b = theano.shared(np.tile(0.0, (architecture[i + 1], 1)).astype(theano.config.floatX), name = "b%s" % (i + 1))
            self.varArrayB.append(b)
        self.gradArray = []
        for i in xrange(self.lastArrayNum):
            self.gradArray.append(self.varArrayW[i])
            self.gradArray.append(self.varArrayB[i])
    def trainCompile(self, numpatches):
        self.ein = theano.shared(np.tile(1.0, (1, numpatches)).astype(theano.config.floatX), name = "ein")
        self.x = T.matrix("x")
        self.y = T.matrix("y")
        if self.options.dropout:
            srng = RandomStreams()
            self.dropOutVectors = []
            for i in xrange(self.lastArrayNum):
                self.dropOutVectors.append(srng.binomial(p = self.options.dropOutParam[i], size = (self.architecture[i], 1)).astype(theano.config.floatX))
        self.varArrayA = []
        for i in xrange(self.lastArrayNum):
            variable2 = T.dot(self.dropOutVectors[i], self.ein) if self.options.dropout else 1.0
            variable = self.x if i == 0 else self.varArrayA[i - 1]
            a = self.modelFunction[i](self.varArrayW[i], variable * variable2, self.varArrayB[i], self.ein)
            self.varArrayA.append(a)
        self.sparse = 0
        self.regularize = 0
        if self.options.sparsity and self.lastArrayNum <= 2:
            sprs = T.sum(self.varArrayA[0], axis = 1) / (numpatches + 0.0)
            epsilon = 1e-20
            sprs = T.clip(sprs, epsilon, 1 - epsilon)
            KL = T.sum(self.options.sparsityParam * T.log(self.options.sparsityParam / sprs) + (1 - self.options.sparsityParam) * T.log((1 - self.options.sparsityParam) / (1 - sprs)))
            self.sparse = self.options.beta * KL
        if self.options.regularization:
            wsum = 0
            for w in self.varArrayW:
                wsum += T.sum(w ** 2)
            self.regularize = self.options.lamda / 2 * wsum
        XENT = 1.0 / numpatches * T.sum((self.y - self.varArrayA[-1]) ** 2 * 0.5)
        self.errorArray = []
        self.cost = XENT + self.sparse + self.regularize
        self.derivativesArray = []
        self.derivativesArray = T.grad(self.cost, self.gradArray)
        if self.options.rmsProp:
            self.MMSprev = []
            self.MMSnew = []
            for i in xrange(len(self.derivativesArray)):
                mmsp = theano.shared(np.tile(0.0, self.gradArray[i].get_value().shape).astype(theano.config.floatX), name = "mmsp%s" % (i + 1))
                self.MMSprev.append(mmsp)
                mmsn = self.options.decay * mmsp + (1 - self.options.decay) * self.derivativesArray[i] ** 2
                mmsn = T.clip(mmsn, self.options.mmsmin, 1e+20)
                self.MMSnew.append(mmsn)
        if self.options.rProp:
            baseRpropStep = 0.001
            decreaseCoef = 0.1
            increaseCoef = 1.3
            minRpropStep = 1e-6
            maxRpropStep = 50
            signedChanged = 0
            numWeights = 0
            for i in xrange(self.lastArrayNum):
                numWeights += self.architecture[i] * self.architecture[i + 1]
            self.prevGW = []
            self.deltaW = []
            for i in xrange(len(self.derivativesArray)):
                prevGW = theano.shared(np.tile(1.0, self.gradArray[i].get_value().shape).astype(theano.config.floatX), name = "prewGW%s" % (i + 1))
                deltaW = theano.shared(np.tile(np.float32(baseRpropStep), self.gradArray[i].get_value().shape).astype(theano.config.floatX), name = "deltaW%s" % (i + 1))
                self.prevGW.append(prevGW)
                self.deltaW.append(deltaW)
        self.updatesArray = []
        for i in xrange(len(self.derivativesArray)):
            if self.options.rmsProp:
                updateVar = self.options.learnStep * self.derivativesArray[i] / self.MMSnew[i] ** 0.5
                self.updatesArray.append((self.MMSprev[i], self.MMSnew[i]))
            elif self.options.rProp:
                updateVar = self.deltaW[i] * decreaseCoef * T.lt(self.prevGW[i] * self.derivativesArray[i], 0) + self.deltaW[i] * increaseCoef * T.gt(self.prevGW[i] * self.derivativesArray[i], 0)
                updateVar = T.clip(updateVar, minRpropStep, maxRpropStep)
                updateVar = updateVar * T.sgn(self.derivativesArray[i])
                self.updatesArray.append((self.prevGW[i], self.derivativesArray[i]))
                self.updatesArray.append((self.deltaW[i], T.abs_(updateVar)))
                signedChanged += T.sum(T.lt(self.prevGW[i] * self.derivativesArray[i], 0))
            else:
                updateVar = self.options.learnStep * self.derivativesArray[i]
            self.updatesArray.append((self.gradArray[i], self.gradArray[i] - updateVar))
        self.train = theano.function(inputs = [self.x, self.y], outputs = [self.cost, XENT], updates = self.updatesArray, allow_input_downcast = True)
        return self
    def predictCompile(self, numpatches, layerNum = -1):
        self.ein2 = theano.shared(np.tile(1.0, (1, numpatches)).astype(theano.config.floatX), name = "ein2")
        self.data = T.matrix("data")
        self.varArrayAc = []
        for i in xrange(self.lastArrayNum):
            variable2 = self.options.dropOutParam[i] if self.options.dropout else 1.0
            variable = self.data if i == 0 else self.varArrayAc[i - 1]
            a = self.modelFunction[i](self.varArrayW[i] * variable2, variable, self.varArrayB[i], self.ein2)
            self.varArrayAc.append(a)
        self.predict = theano.function(inputs = [self.data], outputs = self.varArrayAc[layerNum], allow_input_downcast = True)
        return self
    def trainCalc(self, X, Y, iteration = 10, debug = False, errorCollect = False):
        for i in xrange(iteration):
            timeStart = time.time()
            error, ent = self.train(X, Y)
            if errorCollect:
                self.errorArray.append(ent)
            timeStop = time.time()
            if debug:
                print error
        return self
    def predictCalc(self, X, debug = False):
        self.out = self.predict(X)
        if debug: print self.out.shape
        return self
    def getStatus(self):
        print self.REPORT
        return self
    def paramGetter(self):
        self.model = []
        for obj in self.gradArray:
            self.model.append(obj.get_value())
        return self.model
    def paramSetter(self, array):
        for obj in self.gradArray:
            obj.set_value(array[self.gradArray.index(obj)].astype(theano.config.floatX))
    def modelSaver(self, folder):
        f = file(folder, "wb")
        for obj in self.paramGetter():
            cPickle.dump(obj, f, protocol = cPickle.HIGHEST_PROTOCOL)
        f.close()
        self.getStatus()
        return self
    def modelLoader(self, folder):
        f = file(folder, "rb")
        loadedObjects = []
        for i in xrange(len(self.gradArray)):
            loadedObjects.append(cPickle.load(f))
        f.close()
        self.paramSetter(loadedObjects)
        self.getStatus()
        return self
    def weightsVisualizer(self, folder, size = (100, 100), color = "L"):
        W1 = self.gradArray[0].get_value()
        W2 = self.gradArray[2].get_value()
        for w in xrange(len(W1)):
            img = W1[w, :].reshape(size[0], size[1])
            Graphic.PicSaver(img, folder, "L1_" + str(w), color)
        #for w in xrange(len(W2)):
        #    img = np.dot(W1.T, W2[w, :]).reshape(size[0], size[1])
        #    Graphic.PicSaver(img, folder, "L2_" + str(w), color)
        return self
#---------------------------------------------------------------------#
# Usefull functions
#---------------------------------------------------------------------#
class NNsupport(object):
    @staticmethod
    def crossV(number, y, x, modelObj):
        ERROR = 1.0 / number * np.sum((y - modelObj.predictCalc(x).out) ** 2 * 0.5)
        return ERROR
    @staticmethod
    def errorG(errorArray, folder, plotsize = 50):
        x = range(len(errorArray))
        y = list(errorArray)
        area = plotsize
        fig = plt.figure(figsize = (10, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x, y, s = area, alpha = 0.5)
        fig.savefig(folder)
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#