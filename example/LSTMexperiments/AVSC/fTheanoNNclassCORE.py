# --------------------------------------------------------------------#
# ------------------------NEW-ARCH----------------CORE-7.1------------#
# --------------------------------------------------------------------#
# http://en.wikipedia.org/wiki/Harder,_Better,_Faster,_Stronger
# ---------------------------------------------------------------------#


# ---------------------------------------------------------------------#
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
#import datetime
import matplotlib.pyplot as plt


#---------------------------------------------------------------------#
# Layer builders
#---------------------------------------------------------------------#


class LayerNN(object):
    def __init__(self,
                 size_in=1,
                 size_out=1,
                 activation=False,
                 weightDecay=False,
                 sparsity=False,
                 beta=False,
                 dropout=False,
                 dropConnect=False,
                 pool_size=False):

        self.size_in = size_in
        self.size_out = size_out
        self.activation = activation
        self.weightDecay = weightDecay
        self.sparsity = sparsity
        self.beta = beta
        self.dropout = dropout
        self.dropConnect = dropConnect
        self.pool_size = pool_size

    def Printer(self):
        print self.__dict__

    def compileWeight(self, net, layerNum):
        random = sqrt(6) / sqrt(net.architecture[0].size_in + net.architecture[-1].size_out)

        #In case MaxOut we have to extend weights times pool_size
        if self.activation != FunctionModel.MaxOut:
            weights = np.random.randn(self.size_out, self.size_in)
        else:
            weights = np.random.randn(self.size_out * self.pool_size, self.size_in)

        #weights rescale
        weights_min = np.min(weights)
        weights = weights - weights_min
        weights_max = np.max(weights)
        weights = weights / weights_max

        w = theano.shared((weights * 2 * random - random).astype(theano.config.floatX), name="w%s" % (layerNum + 1))

        net.varArrayW.append(w)
        if self.activation != FunctionModel.MaxOut:
            b = theano.shared(np.tile(0.0, (self.size_out,)).astype(theano.config.floatX), name="b%s" % (layerNum + 1))
        else:
            b = theano.shared(np.tile(0.0, (self.size_out * self.pool_size,)).astype(theano.config.floatX),
                              name="b%s" % (layerNum + 1))
        net.varArrayB.append(b)

    def compileDropout(self, net, R):
        if self.dropout:
            net.dropOutVectors.append(R.binomial(p=self.dropout, size=(self.size_in,)).astype(theano.config.floatX))
        else:
            net.dropOutVectors.append(1.0)

    def compileSparsity(self, net, layerNum, num):
        sprs = T.sum(net.varArrayA[layerNum], axis=1) / (num + 0.0)
        epsilon = 1e-20
        sprs = T.clip(sprs, epsilon, 1 - epsilon)
        KL = T.sum(
            self.sparsity * T.log(self.sparsity / sprs) + (1 - self.sparsity) * T.log((1 - self.sparsity) / (1 - sprs)))
        net.regularize.append(self.beta * KL)

    def compileActivation(self, net, layerNum):
        variable = net.x if layerNum == 0 else net.varArrayA[layerNum - 1]
        a = self.activation(net.varArrayW[layerNum],
                            variable * (net.dropOutVectors[layerNum].dimshuffle(0, 'x') if self.dropout else 1.0),
                            net.varArrayB[layerNum], self.pool_size)
        net.varArrayA.append(a)

    def compilePredictActivation(self, net, layerNum):
        variable = net.x if layerNum == 0 else net.varArrayAc[layerNum - 1]
        a = self.activation(net.varArrayW[layerNum] * (self.dropout if self.dropout else 1.0), variable,
                            net.varArrayB[layerNum], self.pool_size)
        net.varArrayAc.append(a)

    def compileWeightDecayPenalty(self, net, layerNum):
        penalty = T.sum(net.varArrayW[layerNum] ** 2) * self.weightDecay / 2
        net.regularize.append(penalty)


class LayerRNN(LayerNN):
    def __init__(self, blocks=1, peeholes=False, **kwargs):

        super(LayerRNN, self).__init__(**kwargs)
        self.blocks = blocks

        # Check blocks = size_out
        assert self.blocks == self.size_out, 'In case RNN - size_out must be equivalent(!!) to number of blocks.'

        self.peeholes = peeholes
        if self.peeholes:
            self.size_in += self.blocks

        # Internal NN out size
        self.internalOutSize = self.blocks * 4

        # RNN specific variable
        self.A = None
        self.W_mask = None

    def compileWeight(self, net, layerNum):
        random = 0.1

        # W
        weights = np.random.randn(self.internalOutSize, self.size_in)

        #Scale weights to [-0.1 ... 0.1]
        weights_min = np.min(weights)
        weights = weights - weights_min
        weights_max = np.max(weights)
        weights = weights / weights_max
        weights = weights * 2 * random - random

        w = theano.shared(weights.astype(theano.config.floatX), name="w%s" % (layerNum + 1))
        net.varArrayW.append(w)

        # B
        # German PhD magic recommendations...
        B = [0, 0, -2, 2]
        B = np.array(B)
        B = np.tile(B, (self.blocks,))
        for i in xrange(self.blocks):
            B[i * 4] = np.random.randn() * random / 6.0

        b = theano.shared(B.astype(theano.config.floatX),
                          name="b%s" % (layerNum + 1))
        net.varArrayB.append(b)

        # A
        self.A = theano.shared(np.tile(0.0, (self.blocks, net.options.minibatch_size)).astype(theano.config.floatX),
                               name='A%s' % (layerNum + 1))

        # A for predict
        self.A_predict = theano.shared(np.tile(0.0, (self.blocks, net.options.CV_size)).astype(theano.config.floatX),
                                       name='A_p%s' % (layerNum + 1))

        # Mask - for peeholes connection
        if self.peeholes:
            mask = np.zeros((self.blocks, self.internalOutSize))
            for i in range(self.blocks):
                mask[i, i * 4 + 1:i * 4 + 4] = 1
            mask = np.float32(mask)
            self.W_mask = mask

    def compileActivation(self, net, layerNum):
        variable = net.x if layerNum == 0 else net.varArrayA[layerNum - 1]

        extX = T.zeros((self.size_in, net.options.minibatch_size))
        extX = T.set_subtensor(extX[:self.size_in - self.blocks, :], variable)
        extX = T.set_subtensor(extX[self.size_in - self.blocks:, :], self.A)

        maskedW = T.set_subtensor(net.varArrayW[layerNum].T[-self.blocks:, :],
                                  (net.varArrayW[layerNum].T[-self.blocks:, :] * self.W_mask).astype(
                                      theano.config.floatX)).T

        a = self.activation(maskedW,
                            extX * (net.dropOutVectors[layerNum].dimshuffle(0, 'x') if self.dropout else 1.0),
                            net.varArrayB[layerNum])

        a = a.reshape((self.blocks, 4, net.options.minibatch_size))

        Pi = a[:, 0, :] * a[:, 1, :]
        Pr = a[:, 2, :] * self.A
        Au = Pi + Pr
        Po = a[:, 3, :] * Au

        net.updatesArray.append((self.A, Au))
        net.varArrayA.append(Po)

    def compilePredictActivation(self, net, layerNum):
        variable = net.x if layerNum == 0 else net.varArrayAc[layerNum - 1]

        extX = T.zeros((self.size_in, net.options.CV_size))
        extX = T.set_subtensor(extX[:self.size_in - self.blocks, :], variable)
        extX = T.set_subtensor(extX[self.size_in - self.blocks:, :], self.A_predict)

        maskedW = T.set_subtensor(net.varArrayW[layerNum].T[-self.blocks:, :],
                                  (net.varArrayW[layerNum].T[-self.blocks:, :] * self.W_mask).astype(
                                      theano.config.floatX)).T

        a = self.activation(maskedW * (self.dropout if self.dropout else 1.0),
                            extX,
                            net.varArrayB[layerNum])

        a = a.reshape((self.blocks, 4, net.options.CV_size))

        Pi = a[:, 0, :] * a[:, 1, :]
        Pr = a[:, 2, :] * self.A_predict
        Au = Pi + Pr
        Po = a[:, 3, :] * Au

        net.updatesArrayPredict.append((self.A_predict, Au))
        net.varArrayAc.append(Po)


#---------------------------------------------------------------------#
# Activation functions
#---------------------------------------------------------------------#


class FunctionModel(object):
    @staticmethod  # FunctionModel.Sigmoid
    def Sigmoid(W, X, B, *args):
        z = T.dot(W, X) + B.dimshuffle(0, 'x')
        a = 1 / (1 + T.exp(-z))
        return a

    @staticmethod  # FunctionModel.Tanh
    def Tanh(W, X, B, *args):
        z = T.dot(W, X) + B
        a = (T.exp(z) - T.exp(-z)) / (T.exp(z) + T.exp(-z))
        return a

    @staticmethod  # FunctionModel.SoftMax
    def SoftMax(W, X, B, *args):
        z = T.dot(W, X) + B
        numClasses = W.get_value().shape[0]
        # ___CLASSIC___ #
        # a = T.exp(z) / T.dot(T.alloc(1.0, numClasses, 1), [T.sum(T.exp(z), axis = 0)])
        # _____________ #
        # Second way antinan
        # a = T.exp(z - T.log(T.sum(T.exp(z))))
        # a = T.exp(z - T.log(T.dot(T.alloc(1.0, numClasses, 1), [T.sum(T.exp(z), axis = 0)])))		#FIXED?
        # ___ANTINAN___ #
        z_max = T.max(z, axis=0)
        a = T.exp(z - T.log(T.dot(T.alloc(1.0, numClasses, 1), [T.sum(T.exp(z - z_max), axis=0)])) - z_max)
        # _____________ #
        # Some hacks for fixing float32 GPU problem
        # a = T.clip(a, float(np.finfo(np.float32).tiny), float(np.finfo(np.float32).max))
        # a = T.clip(a, 1e-20, 1e20)
        # http://www.velocityreviews.com/forums/t714189-max-min-smallest-float-value-on-python-2-5-a.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.finfo.html
        # Links about possible approaches to fix nan
        # http://blog.csdn.net/xceman1997/article/details/9974569
        # https://github.com/Theano/Theano/issues/1563
        return a

    @staticmethod  # FunctionModel.MaxOut
    def MaxOut(W, X, B, *args):
        z = T.dot(W, X) + B.dimshuffle(0, 'x')
        d = T.shape(z)
        n_elem = args[0]
        z = z.reshape((d[0] / n_elem, n_elem, d[1]))
        a = T.max(z, axis=1)
        return a


#---------------------------------------------------------------------#
# Options instance
#---------------------------------------------------------------------#


class newOptionsStore(object):
    def __init__(self,
                 learnStep=0.01,
                 rmsProp=False,
                 mmsmin=1e-10,
                 rProp=False,
                 minibatch_size=1,
                 CV_size=1):
        self.learnStep = learnStep  # Learning step for gradient descent
        self.rmsProp = rmsProp  # rmsProp on|off
        self.mmsmin = mmsmin  # Min mms value
        self.rProp = rProp  # For full batch only
        self.minibatch_size = minibatch_size
        self.CV_size = CV_size

    def Printer(self):
        print self.__dict__


#---------------------------------------------------------------------#
# Basic neural net class
#---------------------------------------------------------------------#


class newTheanoNNclass(object):
    def __init__(self, opt, architecture):
        REPORT = "OK"

        self.architecture = architecture
        self.options = opt
        self.lastArrayNum = len(architecture)

        self.varArrayW = []
        self.varArrayB = []

        # Variables
        self.x = T.matrix("x")
        self.y = T.matrix("y")

        # Weights
        for i in xrange(self.lastArrayNum):
            self.architecture[i].compileWeight(self, i)

        # All variables to gradArray
        # TODO - revise to be more flexible in case RNN and CNN networks
        self.gradArray = []
        for i in xrange(self.lastArrayNum):  # Possible use len(self.varArrayB) or len(self.varArrayW) instead
            self.gradArray.append(self.varArrayW[i])
            self.gradArray.append(self.varArrayB[i])

        # Dropout
        self.dropOutVectors = []
        srng = RandomStreams()  # Theano random generator for dropout
        for i in xrange(self.lastArrayNum):
            self.architecture[i].compileDropout(self, srng)

        # Activations list
        self.varArrayA = []

        # Additional penalty list
        self.regularize = []

        # Error calculation
        self.errorArray = []  # Storage for costs
        self.cost = 0

        #Derivatives array
        self.derivativesArray = []

        # RMS
        if self.options.rmsProp:
            self.MMSprev = []
            self.MMSnew = []

        # Update array
        self.updatesArray = []  # Array for train theano function updates input parameter

        # Sometimes there is something to update even for predict (say in RNN)
        self.updatesArrayPredict = []

        # train
        self.train = None

        # predict
        self.predict = None
        self.out = None

        # Predict variables
        self.data = T.matrix("data")
        self.varArrayAc = []

        # List of output variables
        self.outputArray = []

    def trainCompile(self):

        # Activation
        for i in xrange(self.lastArrayNum):
            self.architecture[i].compileActivation(self, i)

        # Sparse penalty
        for i in xrange(self.lastArrayNum):
            l = self.architecture[i]
            if l.sparsity:
                l.compileSparsity(self, i, self.options.minibatch_size)

        # Weight decay penalty
        for i in xrange(self.lastArrayNum):
            l = self.architecture[i]
            if l.weightDecay:
                l.compileWeightDecayPenalty(self, i)

        # Error
        XENT = 1.0 / self.options.minibatch_size * T.sum((self.y - self.varArrayA[-1]) ** 2 * 0.5)
        self.cost = XENT
        for err in self.regularize:
            self.cost += err

        # Update output array
        self.outputArray.append(self.cost)
        self.outputArray.append(XENT)

        # Derivatives
        self.derivativesArray = T.grad(self.cost, self.gradArray)

        # RMS
        if self.options.rmsProp:
            for i in xrange(len(self.derivativesArray)):
                mmsp = theano.shared(np.tile(0.0, self.gradArray[i].get_value().shape).astype(theano.config.floatX),
                                     name="mmsp%s" % (i + 1))  # 0.0 - 1.0 maybe
                self.MMSprev.append(mmsp)
                mmsn = self.options.rmsProp * mmsp + (1 - self.options.rmsProp) * self.derivativesArray[i] ** 2
                mmsn = T.clip(mmsn, self.options.mmsmin, 1e+15)  # Fix nan if rmsProp
                self.MMSnew.append(mmsn)

        # Update values
        for i in xrange(len(self.derivativesArray)):
            if self.options.rmsProp:
                updateVar = self.options.learnStep * self.derivativesArray[i] / self.MMSnew[i] ** 0.5
                self.updatesArray.append((self.MMSprev[i], self.MMSnew[i]))
            else:
                updateVar = self.options.learnStep * self.derivativesArray[i]
            self.updatesArray.append((self.gradArray[i], self.gradArray[i] - updateVar))

        self.train = theano.function(inputs=[self.x, self.y],
                                     outputs=self.outputArray,
                                     updates=self.updatesArray,
                                     allow_input_downcast=True)
        return self

    def trainCalc(self, X, Y, iteration=10, debug=False, errorCollect=False):  # Need to call trainCompile before
        for i in xrange(iteration):
            error, ent = self.train(X, Y)
            if errorCollect:
                self.errorArray.append(ent)
            if debug:
                print ent, error
        return self

    def predictCompile(self):
        # Predict activation
        for i in xrange(self.lastArrayNum):
            self.architecture[i].compilePredictActivation(self, i)

        self.predict = theano.function(inputs=[self.x],
                                       outputs=self.varArrayAc[-1],
                                       updates=self.updatesArrayPredict,
                                       allow_input_downcast=True)
        return self

    def predictCalc(self, X, debug=True):  # Need to call predictCompile before
        self.out = self.predict(X)  # Matrix of outputs. Each column is a picture reshaped in vector of features
        if debug:
            print self.out.shape
        return self

    def getStatus(self):  # Its time for troubles
        print self.REPORT
        return self

    def paramGetter(self):  # Returns the values of model parameters such as [w1, b1, w2, b2] ect.
        self.model = []
        counter = 0
        for obj in self.gradArray:
            L = len(self.gradArray)
            M = range(0, L, 2)
            variable = self.architecture[counter / 2].dropout if counter in M and self.architecture[
                counter / 2].dropout else 1.0
            self.model.append(obj.get_value() * variable)
            counter += 1
        return self.model

    def paramSetter(self, array):       # Setups loaded model parameters
        counter = 0
        for obj in self.gradArray:
            L = len(self.gradArray)
            M = range(0, L, 2)
            variable = self.architecture[counter / 2].dropout if counter in M and self.architecture[
                counter / 2].dropout else 1.0
            obj.set_value((array[self.gradArray.index(obj)] / variable).astype(
                theano.config.floatX))
            counter += 1

    def modelSaver(self, folder):       # In cPickle format in txt file
        f = file(folder, "wb")
        for obj in self.paramGetter():
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        self.getStatus()
        return self

    def modelLoader(self, folder):      # Path to model txt file
        f = file(folder, "rb")
        loadedObjects = []
        for i in xrange(len(self.gradArray)):       # Array that contains w1 b1 w2 b2 etc.
            loadedObjects.append(cPickle.load(f))
        f.close()                                   # Then we need to update W and B parameters
        self.paramSetter(loadedObjects)
        self.getStatus()
        return self


#---------------------------------------------------------------------#
# OLD neural net class
#---------------------------------------------------------------------#


class TheanoNNclass(object):
    def weightsVisualizer(self, folder, size=(100, 100),
                          color="L"):  # For now only for first layer. Second in test mode
        W1 = self.gradArray[0].get_value()
        W2 = self.gradArray[2].get_value()  # Second layer test. Weighted linear combination of the first layer bases
        for w in xrange(len(W1)):
            img = W1[w, :].reshape(size[0], size[1])  # Fix to auto get size TODO
            Graphic.PicSaver(img, folder, "L1_" + str(w), color)
        for w in xrange(len(W2)):
            img = np.dot(W1.T, W2[w, :]).reshape(size[0], size[1])
            Graphic.PicSaver(img, folder, "L2_" + str(w), color)
        return self


#---------------------------------------------------------------------#
# Useful functions
#---------------------------------------------------------------------#


class NNsupport(object):
    @staticmethod
    def crossV(number, y, x, modelObj):
        ERROR = 1.0 / number * np.sum((y - modelObj.predictCalc(x).out) ** 2 * 0.5)
        return ERROR

    @staticmethod
    def errorG(errorArray, folder, plotsize=50):
        x = range(len(errorArray))
        y = list(errorArray)
        area = plotsize
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)  # One row, one column, first plot
        ax.scatter(x, y, s=area, alpha=0.5)
        fig.savefig(folder)


#---------------------------------------------------------------------#
# Can this really be the end? Back to work you go again
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#