# --------------------------------------------------------------------#
# ------------------------NEW-ARCH----------------CORE-7.1------------#
# --------------------------------------------------------------------#
# http://en.wikipedia.org/wiki/Harder,_Better,_Faster,_Stronger
# ---------------------------------------------------------------------#

# ---------------------------------------------------------------------#
# External libraries
# ---------------------------------------------------------------------#


from PIL import Image, ImageOps, ImageFilter
import numpy as np
from numpy import *
from numpy import dot, sqrt, diag
from numpy.linalg import eigh
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import theano.tensor as T
import cPickle
import time  # import datetime
import matplotlib.pyplot as plt
from fImageWorkerCORE import *
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from theano.tensor.signal import downsample


#---------------------------------------------------------------------#
# Activation functions
#---------------------------------------------------------------------#

class FunctionModel(object):
    """
    Collection of activation functions we support.
    """

    @staticmethod
    def Sigmoid(z, *args):
        """
        Standard sigmoid.

        .. math::

           activation = \\frac{1}{1 + e^{-z}}

        :param z: array, raw activation, usually calculated as :math:`z=W^Tx` that will be used for further calculation.
        :param args: array, additional parameters. For now uses for MaxOut.
        :return: array, same size as **z**.
        """
        a = 1 / (1 + T.exp(-z))
        return a

    @staticmethod
    def ReLU(z, *args):
        """
        Rectified Linear Unit. More `info <http://en.wikipedia.org/wiki/Rectifier_%28neural_networks%29>`_.

        .. math::

           activation = \\max(0, z)

        :param z: array, raw activation, usually calculated as :math:`z=W^Tx` that will be used for further calculation.
        :param args: array, additional parameters. For now uses for MaxOut.
        :return: array, same size as **z**.
        """
        a = T.switch(T.gt(z, 0), z, 0)
        return a

    @staticmethod
    def LReLU(z, *args):
        """
        Leaky Rectified Linear Unit. More `info <http://en.wikipedia.org/wiki/Rectifier_%28neural_networks%29>`_.

        .. math::

           activation=
              \\begin{cases}
                z, & if z > 0\\\\
                0.01z, & otherwise
              \\end{cases}

        :param z: array, raw activation, usually calculated as :math:`z=W^Tx` that will be used for further calculation.
        :param args: array, additional parameters. For now uses for MaxOut.
        :return: array, same size as **z**.
        """
        a = T.switch(T.gt(z, 0), z, z * 0.01)
        return a

    @staticmethod
    def Linear(z, *args):
        """
        Linear activation function. Returns input as-is.

        :param z: array, raw activation, usually calculated as :math:`z=W^Tx` that will be used for further calculation.
        :param args: array, additional parameters. For now uses for MaxOut.
        :return: array, same size as **z**.
        """
        return z

    @staticmethod
    def Tanh(z, *args):
        """
        Hyperbolic tangent.

        .. math::

           activation = \\frac{e^z - e^{-z}}{e^z + e^{-z}}

        :param z: array, raw activation, usually calculated as :math:`z=W^Tx` that will be used for further calculation.
        :param args: array, additional parameters. For now uses for MaxOut.
        :return: array, same size as **z**.
        """
        a = (T.exp(z) - T.exp(-z)) / (T.exp(z) + T.exp(-z))
        return a

    @staticmethod
    def SoftMax(z, *args):
        """
        SoftMax activation function with several updates to avoid NaN.

        It is useful for **output layer only**.

        .. math::

           activation = \\frac{1}{\\sum\\limits_{j=1}^k e^{\\theta_j^T x^{(i)}}}
           \\left[\\begin{aligned}
           e&^{\\theta_1^Tx^{(i)}}\\\\
           e&^{\\theta_2^Tx^{(i)}}\\\\
           &\\vdots\\\\
           e&^{\\theta_k^Tx^{(i)}}
           \\end{aligned}\\right]

        Some hacks for fixing float32 GPU problem:

        .. code:: python

           a = T.clip(a, float(np.finfo(np.float32).tiny), float(np.finfo(np.float32).max))
           a = T.clip(a, 1e-20, 1e20)

        Proof links:

        * http://www.velocityreviews.com/forums/t714189-max-min-smallest-float-value-on-python-2-5-a.html
        * http://docs.scipy.org/doc/numpy/reference/generated/numpy.finfo.html

        Links about possible approaches to fix NaN:

        * http://blog.csdn.net/xceman1997/article/details/9974569
        * https://github.com/Theano/Theano/issues/1563

        :param z: array, raw activation, usually calculated as :math:`z=W^Tx` that will be used for further calculation.
        :param args: array, additional parameters. For now uses for MaxOut.
        :return: array, same size as **z**.
        """
        numClasses = T.shape(z)[0]
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

        return a

    @staticmethod
    def MaxOut(z, *args):
        """
        MaxOut activation function.

        Original paper: http://arxiv.org/pdf/1302.4389.pdf

        .. math::

           activation_{i} = \max_{j \in [1,k]} z_{i,j}

        :param z: array, raw activation, usually calculated as :math:`z=W^Tx` that will be used for further calculation.
        :param args: [0] - the number of "lines" to emulate MaxOut in each pool.
                           Say, in case we have here 3 - each output neuron will be emulated as 3 linear functions.
        :return: array, size along [0] axis reduced times "lines".
        """
        d = T.shape(z)
        n_elem = args[0]
        z = z.reshape((d[0] / n_elem, n_elem, d[1]))
        a = T.max(z, axis=1)
        return a


# ---------------------------------------------------------------------#
# Layer builders
# ---------------------------------------------------------------------#


class LayerNN(object):
    """
    Basic layer class. By default - standard NeuralNet fully-connected network.

    :param size_in: int, number of neurons on input
    :param size_out: int, number neurons on out
    :param activation: FunctionModel, activation function to use
    :param weightDecay: float or False, weight decay regularization and its coefficient
    :param sparsity: float or False, sparcity constraint. Make sense only with Sigmoid activation function
    :param beta: float, sparse weight coefficient
    :param dropout: float or False, dropout regularisation with defined coefficient
    :param dropConnect: TBD
    :param pool_size: int, Should be specified only for MaxOut activation function. Number of lines to emulate each neroun.
    :return: layer object.
    """
    def __init__(self,
                 size_in=1,
                 size_out=1,
                 activation=FunctionModel.Sigmoid,
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
        """
        Prints layer properties
        :return:
        """
        print self.__dict__

    def compileWeight(self, net, layerNum):
        """
        Allocates weights to be used as shared variable in Theano

        :param net: TheanoNNclass object
        :param layerNum: layer's index.
        :return:
        """
        random = sqrt(6) / sqrt(net.architecture[0].size_in + net.architecture[-1].size_out)
        W = dict()

        #In case MaxOut we have to extend weights times pool_size
        if self.activation != FunctionModel.MaxOut:
            weights = np.random.randn(self.size_out, self.size_in)
        else:
            weights = np.random.randn(self.size_out * self.pool_size, self.size_in)

        #weights rescale
        #weights_min = np.min(weights)
        #weights = weights - weights_min
        #weights_max = np.max(weights)
        #weights = weights / weights_max

        #w = theano.shared((weights * 2 * random - random).astype(theano.config.floatX), name="w%s" % (layerNum + 1))
        w = theano.shared((weights * 0.01).astype(theano.config.floatX), name="w%s" % (layerNum + 1))

        W['w'] = w

        if self.activation != FunctionModel.MaxOut:
            b = theano.shared(np.tile(0.1, (self.size_out,)).astype(theano.config.floatX), name="b%s" % (layerNum + 1))
        else:
            b = theano.shared(np.tile(0.1, (self.size_out * self.pool_size,)).astype(theano.config.floatX),
                              name="b%s" % (layerNum + 1))
        W['b'] = b
        net.varWeights.append(W)

    def compileDropout(self, net, R):
        """
        Compile necessary mask matrix for dropout regularisation.

        :param net: TheanoNNclass object
        :param R: Theano's RandomGenerator object
        :return:
        """
        if self.dropout:
            net.dropOutVectors.append(R.binomial(p=self.dropout, size=(self.size_in,)).astype(theano.config.floatX))
        else:
            net.dropOutVectors.append(1.0)

    def compileSparsity(self, net, layerNum, num):
        """
        Compile necessary sparsity constraint calculations.

        Average activation of hidden unit *j* (averaged over the training set):

        .. math::

           \\hat{\\rho} = \\frac{1}{m}\\sum\\limits_{i=1}^{m}\\left[a_j(x^{(i)})\\right]

        Then penalty (using Kullback-Leibler):

        .. math::

           penalty = \\sum\\limits_{j=1}^{hiddenUnits}\\rho\\log\\frac{\\rho}{\\hat\\rho_{j}} + (1 - \\rho)\\log\\frac{1 - \\rho}{1 - \\hat\\rho_{j}}

        where :math:`\\rho` - is *sparsity* parameter. Means - the level of average activation we want to achieve.

        :param net: TheanoNNclass object
        :param layerNum: int, layer's index
        :param num: batch size
        :return:
        """
        sprs = T.sum(net.varArrayA[layerNum], axis=1) / (num + 0.0)
        epsilon = 1e-20
        sprs = T.clip(sprs, epsilon, 1 - epsilon)
        KL = T.sum(
            self.sparsity * T.log(self.sparsity / sprs) + (1 - self.sparsity) * T.log((1 - self.sparsity) / (1 - sprs)))
        net.regularize.append(self.beta * KL)

    def compileActivation(self, net, layerNum):
        """
        Compile layer's activation taking into account dropout and specified activation function.
        Used during network's training to calculate activations.

        :param net: TheanoNNclass object
        :param layerNum: int, layer's index
        :return:
        """
        variable = net.x if layerNum == 0 else net.varArrayA[layerNum - 1]

        #W x X + B
        z = T.dot(net.varWeights[layerNum]['w'], variable * (net.dropOutVectors[layerNum].dimshuffle(0, 'x') if self.dropout else 1.0)) + net.varWeights[layerNum]['b'].dimshuffle(0, 'x')

        a = self.activation(z, self.pool_size)

        #a = self.activation(net.varWeights[layerNum]['w'],
        #                    variable * (net.dropOutVectors[layerNum].dimshuffle(0, 'x') if self.dropout else 1.0),
        #                    net.varWeights[layerNum]['b'], self.pool_size)
        net.varArrayA.append(a)

    def compilePredictActivation(self, net, layerNum):
        """
        Compile layer's activation taking into account dropout and specified activation function.
        Used to calculate predictions without training.

        :param net: TheanoNNclass object
        :param layerNum: int, layer's index
        :return:
        """
        variable = net.x if layerNum == 0 else net.varArrayAc[layerNum - 1]

        #W x X + B
        z = T.dot(net.varWeights[layerNum]['w'] * (self.dropout if self.dropout else 1.0), variable) + net.varWeights[layerNum]['b'].dimshuffle(0, 'x')

        a = self.activation(z, self.pool_size)

        #a = self.activation(net.varWeights[layerNum]['w'] * (self.dropout if self.dropout else 1.0), variable,
        #                    net.varWeights[layerNum]['b'], self.pool_size)
        net.varArrayAc.append(a)

    def compileWeightDecayPenalty(self, net, layerNum):
        """
        Adds weight decay penalty to network's error. Useful to decrease absolute weight's values.

        .. math::

           penalty = \\frac{1}{2}\\sum W_{target\\>layer}^2

        :param net: TheanoNNclass object
        :param layerNum: int, layer's index
        :return:
        """
        penalty = T.sum(net.varWeights[layerNum]['w'] ** 2) * self.weightDecay / 2
        net.regularize.append(penalty)


class LayerRNN(LayerNN):
    def __init__(self, blocks=1, peeholes=False, **kwargs):
        """
        Layer class that extends standard LayerNN class and implements RNN (recurrent) type of network.
        Particularly, here we implement LSTM (Long Short-Term Memory).

        You can find more info about it on:

        * `Wiki <http://en.wikipedia.org/wiki/Long_short_term_memory>`_
        * `Original paper <http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf>`_
        * More about `traditional LSTM vs peepholed <http://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf>`_

        :param blocks: int, number of blocks to create. Should be equivalent to *size_out*
        :param peeholes: boolean, whether to use peeholes or not (send Acc to input gate).
        :param kwargs: needs for compatibility.
        :return: LayerRNN object
        """

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
        """
        Allocates weights to be used as shared variable in Theano.

        To initialise bias we use values advised `here <http://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf>`_:

        * Input gate: 0.0
        * Forget gate: -2.0
        * Output gate: +2.0

        :param net: TheanoNNclass object
        :param layerNum: layer's index.
        """
        random = 0.1
        W = dict()

        # W
        weights = np.random.randn(self.internalOutSize, self.size_in)

        #Scale weights to [-0.1 ... 0.1]
        weights_min = np.min(weights)
        weights = weights - weights_min
        weights_max = np.max(weights)
        weights = weights / weights_max
        weights = weights * 2 * random - random

        w = theano.shared(weights.astype(theano.config.floatX), name="w%s" % (layerNum + 1))
        W['w'] = w

        # B
        # German PhD magic recommendations...
        B = [0, 0, -2, 2]
        B = np.array(B)
        B = np.tile(B, (self.blocks,))
        for i in xrange(self.blocks):
            B[i * 4] = np.random.randn() * random / 6.0

        b = theano.shared(B.astype(theano.config.floatX),
                          name="b%s" % (layerNum + 1))
        W['b'] = b

        net.varWeights.append(W)

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
        else:
            self.W_mask = 1.0

    def compileActivation(self, net, layerNum):
        """
        Compile layer's activation taking into account dropout.
        It is meaningful to use Sigmoid activation function (or probably hyperbolic tang).

        Activation calculated as follows:

        #. :math:`Input\\>activation`
        #. :math:`Input\\>gate`
        #. :math:`Forget\\>gate`
        #. :math:`Output\\>gate`

        .. note::
           *All above where calculated in one step*

        5. :math:`Pi = {Input\\>activation} \\times {Input\\>gate}`
        #. :math:`Pr = {Forget\\>gate} \\times {Cell\\>state}`
        #. :math:`{Cell\\>state} = Pi + Pr`
        #. :math:`output = {Output\\>gate} \\times {Cell\\>state}`


        :param net: TheanoNNclass object
        :param layerNum: layer's index.
        """
        variable = net.x if layerNum == 0 else net.varArrayA[layerNum - 1]

        if self.peeholes:
            extX = T.zeros((self.size_in, net.options.minibatch_size))
            extX = T.set_subtensor(extX[:self.size_in - self.blocks, :], variable)
            extX = T.set_subtensor(extX[self.size_in - self.blocks:, :], self.A)

            maskedW = T.set_subtensor(net.varWeights[layerNum]['w'].T[-self.blocks:, :],
                                      (net.varWeights[layerNum]['w'].T[-self.blocks:, :] * self.W_mask).astype(
                                          theano.config.floatX)).T

            #W x X + B
            z = T.dot(maskedW, extX * (net.dropOutVectors[layerNum].dimshuffle(0, 'x') if self.dropout else 1.0)) + net.varWeights[layerNum]['b'].dimshuffle(0, 'x')

            a = self.activation(z, self.pool_size)

            #a = self.activation(maskedW,
            #                    extX * (net.dropOutVectors[layerNum].dimshuffle(0, 'x') if self.dropout else 1.0),
            #                    net.varWeights[layerNum]['b'])
        else:
            #W x X + B
            z = T.dot(net.varWeights[layerNum]['w'], variable * (net.dropOutVectors[layerNum].dimshuffle(0, 'x') if self.dropout else 1.0)) + net.varWeights[layerNum]['b'].dimshuffle(0, 'x')

            a = self.activation(z, self.pool_size)

            #a = self.activation(net.varWeights[layerNum]['w'],
            #                    variable * (net.dropOutVectors[layerNum].dimshuffle(0, 'x') if self.dropout else 1.0),
            #                    net.varWeights[layerNum]['b'])

        a = a.reshape((self.blocks, 4, net.options.minibatch_size))

        Pi = a[:, 0, :] * a[:, 1, :]
        Pr = a[:, 2, :] * self.A
        Au = Pi + Pr
        Po = a[:, 3, :] * Au

        net.updatesArray.append((self.A, Au))
        net.varArrayA.append(Po)

    def compilePredictActivation(self, net, layerNum):
        """
        Compile layer's activation taking into account dropout and specified activation function.
        Used to calculate predictions without training.
        Uses **separate** Accumulator to store cell's state independently from training.

        :param net: TheanoNNclass object
        :param layerNum: layer's index
        """
        variable = net.x if layerNum == 0 else net.varArrayAc[layerNum - 1]

        if self.peeholes:
            extX = T.zeros((self.size_in, net.options.CV_size))
            extX = T.set_subtensor(extX[:self.size_in - self.blocks, :], variable)
            extX = T.set_subtensor(extX[self.size_in - self.blocks:, :], self.A_predict)

            maskedW = T.set_subtensor(net.varWeights[layerNum]['w'].T[-self.blocks:, :],
                                      (net.varWeights[layerNum]['w'].T[-self.blocks:, :] * self.W_mask).astype(
                                          theano.config.floatX)).T

            #W x X + B
            z = T.dot(maskedW * (self.dropout if self.dropout else 1.0), extX) + net.varWeights[layerNum]['b'].dimshuffle(0, 'x')

            a = self.activation(z, self.pool_size)

            #a = self.activation(maskedW * (self.dropout if self.dropout else 1.0),
            #                    extX,
            #                    net.varWeights[layerNum]['b'])
        else:
            #W x X + B
            z = T.dot(net.varWeights[layerNum]['w'] * (self.dropout if self.dropout else 1.0), variable) + net.varWeights[layerNum]['b'].dimshuffle(0, 'x')

            a = self.activation(z, self.pool_size)

            #a = self.activation(net.varWeights[layerNum]['w'] * (self.dropout if self.dropout else 1.0),
            #                    variable,
            #                    net.varWeights[layerNum]['b'])

        a = a.reshape((self.blocks, 4, net.options.CV_size))

        Pi = a[:, 0, :] * a[:, 1, :]
        Pr = a[:, 2, :] * self.A_predict
        Au = Pi + Pr
        Po = a[:, 3, :] * Au

        net.updatesArrayPredict.append((self.A_predict, Au))
        net.varArrayAc.append(Po)


class LayerCNN(LayerNN):
    def __init__(self, kernel_shape=None, stride=1, pooling=False, pooling_shape=None, optimized=False,
                 validConvolution=True, **kwargs):
        """
        Layer class that extends standard LayerNN class and implements CNN (convolution, **not fully connected**) type of network.
        The most useful type of network to apply for image processing beyond others NN algorithms.
        It implements the most brain-like way to process data (applies the same weights to small parts of input data).
        Read more about convolution here:

        * http://deeplearning.net/tutorial/lenet.html
        * http://en.wikipedia.org/wiki/Convolutional_neural_network

        :param kernel_shape: tuple of int, kernels to use (number of kernels, colors, shape X, shape Y)
        :param stride: int, step between *windows* in *pixels*
        :param pooling: boolean, whether to use pooling after convolution or not
        :param pooling_shape: int, pooling window's shape. Stride will be the same, so only standard non-overlapping pooling is available.
        :param optimized: boolean, whether to use highly optimized version or not. In case TRUE - it is able to run only on GPU.
        :param validConvolution: whether to use valid (convolve fully overlapped parts) or full (convolve partially overlapped parts) convolution.
        :param kwargs: other parameters are inherited from LayerNN.__init__()

        .. note::
           In case :code:`optimized = True` there are number of restrictions you have take into account:

           * The **number of channels must be even, or less than or equal to 3**.
             If you want to compute the gradient, it should be divisible by 4.
             Valid numbers of input channels are: 1, 2, 3, 4, 8, 12, 16, ...
           * **Filters** must be square.
           * The **number of filters** must be a multiple of **16**.
           * All minibatch sizes are supported, but the best performance is achieved when the minibatch size is a multiple of 128.
           * Only "valid" convolutions are supported.
             If you want to perform a "full" convolution, you will need to use zero-padding (more on this later).
           * Only works on the GPU. You cannot run your Theano code on the CPU if you use it.
             But still possible to train on GPU and to load & run on CPU.
        """

        super(LayerCNN, self).__init__(**kwargs)

        self.kernel_shape = kernel_shape        # (number of kernels, colors, shape[0], shape[1])
        self.stride = stride                    # stride for convolution
        self.pooling = pooling                  # whether we want to use pooling at all
        self.pooling_shape = pooling_shape      # stride for pooling
        self.optimized = optimized
        self.validConvolution = validConvolution


    def compileWeight(self, net, layerNum):
        """
        Allocates weights to be used as shared variable in Theano. It is impossible to use MaxOut as activation function yet.
        In case you experience train issues - try to change init random values.

        :param net: TheanoNNclass object
        :param layerNum: layer's index.
        """
        random = sqrt(6) / sqrt(self.kernel_shape[-1] * self.kernel_shape[-2] * self.kernel_shape[0])
        W = dict()

        #MAXOUT will be implemented later
        if self.activation == FunctionModel.MaxOut:
            raise NotImplementedError('MaxOut activation function for Convolution nets is not implemented yet!')

        #Random init for CNN. Without reshape. Init exact kernel shape
        weights = np.random.randn(*self.kernel_shape)

        #if self.activation != FunctionModel.MaxOut:
        #    #Random init for CNN. Without reshape. Init exact kernel shape
        #    weights = np.random.standard_normal(size=self.kernel_shape)
        #else:
        #    weights = np.random.standard_normal(size=(self.kernel_shape[0] * self.pool_size, self.kernel_shape[1], self.kernel_shape[2], self.kernel_shape[3]))

        #weights rescale
        #weights_min = np.min(weights)
        #weights = weights - weights_min
        #weights_max = np.max(weights)
        #weights = weights / weights_max

        #w = theano.shared((weights * 2 * random - random).astype(theano.config.floatX), name="w%s" % (layerNum + 1))
        w = theano.shared((weights * 0.01).astype(theano.config.floatX), name="w%s" % (layerNum + 1))

        W['w'] = w

        #Bias shape == number of kernels
        b = theano.shared(np.tile(0.1, (self.kernel_shape[0],)).astype(theano.config.floatX), name="b%s" % (layerNum + 1))

        W['b'] = b
        net.varWeights.append(W)

    def compileDropout(self, net, R):
        """
        Compile necessary mask matrix for dropout regularisation.

        :param net: TheanoNNclass object
        :param R: Theano's RandomGenerator object
        """
        if self.dropout:
            net.dropOutVectors.append(R.binomial(p=self.dropout, size=(self.kernel_shape[-2], self.kernel_shape[-1]))
                                      .astype(theano.config.floatX))
        else:
            net.dropOutVectors.append(1.0)

    def compileSparsity(self, net, layerNum, num):
        """
        In general, method does the same as :func:`~fTheanoNNclassCORE.LayerNN.compileSparsity`.
        Can be used in combination with :func:`~fTheanoNNclassCORE.FunctionModel.Sigmoid` only.

        But concretely for CNN it was a little bit modified, to be able to calculates average activations from *bc01* format.

        .. note::

           *bc01 - mean: batch x color x size_X x size_Y*

        :param net: TheanoNNclass object
        :param layerNum: int, layer's index
        :param num: batch size
        """
        a = net.varArrayA[layerNum]
        out_size = T.cast(T.sqrt(T.shape(a)[0] / self.kernel_shape[0]), 'int16')
        a = T.reshape(a, (net.options.minibatch_size, self.kernel_shape[0], out_size, out_size))
        #sprs = T.mean(a, axis=(1, 2, 3))
        sprs = T.mean(a, axis=1)
        epsilon = 1e-20
        sprs = T.clip(sprs, epsilon, 1 - epsilon)
        KL = T.sum(self.sparsity * T.log(self.sparsity / sprs) + (1 - self.sparsity) * T.log((1 - self.sparsity) / (1 - sprs))) / (out_size * out_size)
        net.regularize.append(self.beta * KL)

    def compileActivation(self, net, layerNum):
        variable = net.x if layerNum == 0 else net.varArrayA[layerNum - 1]

        #Calc shapes for reshape function on-the-fly. Assume we have square images as input.
        sX = T.cast(T.sqrt(T.shape(variable)[0] / self.kernel_shape[1]), 'int16')

        #Converts input from 2 to 4 dimensions
        Xr = T.reshape(variable.T, (T.shape(variable)[1], self.kernel_shape[1], sX, sX))

        if self.optimized:
            out_size = T.cast(
                T.ceil((T.shape(Xr)[-1] - T.shape(net.varWeights[layerNum]['w'])[-1] + 1) / np.float32(self.stride)),
                'int32')

            conv_op = FilterActs(stride=self.stride, pad=0 if self.validConvolution else self.kernel_shape[-1] - 1)
            input_shuffled = Xr.dimshuffle(1, 2, 3, 0)  # bc01 to c01b
            filters_shuffled = net.varWeights[layerNum]['w'].dimshuffle(1, 2, 3, 0)  # bc01 to c01b
            filters_flipped = filters_shuffled[:, ::-1, ::-1, :] # flip rows and columns
            contiguous_input = gpu_contiguous(input_shuffled)
            contiguous_filters = gpu_contiguous(filters_flipped *
                                                (net.dropOutVectors[layerNum].dimshuffle('x', 0, 1, 'x') if self.dropout else 1.0))
            a = conv_op(contiguous_input, contiguous_filters)
            a = a[:, :out_size, :out_size, :]
            #Add bias
            a = a + net.varWeights[layerNum]['b'].dimshuffle(0, 'x', 'x', 'x')
        else:
            a = T.nnet.conv2d(Xr, net.varWeights[layerNum]['w'] *
                              (net.dropOutVectors[layerNum].dimshuffle('x', 'x', 0, 1) if self.dropout else 1.0),
                              border_mode='valid' if self.validConvolution else 'full',
                              subsample=(self.stride, self.stride))
            #Add bias
            a = a + net.varWeights[layerNum]['b'].dimshuffle('x', 0, 'x', 'x')

        if self.pooling:
            if self.optimized:
                #Pooling
                # ds - side of square pool window
                # stride - Defines the stride size between successive pooling squares.
                # Setting this parameter smaller than sizeX produces overlapping pools.
                # Setting it equal to sizeX gives the usual, non-overlapping pools. Values greater than sizeX are not allowed.
                pool_op = MaxPool(ds=self.pooling_shape, stride=self.pooling_shape)

                contiguous_input = gpu_contiguous(a)
                a = pool_op(contiguous_input)
                a = a.dimshuffle(3, 0, 1, 2)       # c01b to bc01
            else:
                #a = downsample.max_pool_2d(a, (self.pooling_shape, self.pooling_shape), ignore_border=False)
                a = pool.max_pool2D(a, (self.pooling_shape, self.pooling_shape), ignore_border=False)
        else:
            if self.optimized:
                a = a.dimshuffle(3, 0, 1, 2)       # c01b to bc01

        a = T.flatten(a, outdim=2).T

        #Sigmoid
        a = self.activation(a, self.pool_size)

        net.varArrayA.append(a)

    def compilePredictActivation(self, net, layerNum):
        variable = net.x if layerNum == 0 else net.varArrayAc[layerNum - 1]

        #Calc shapes for reshape function on-the-fly. Assume we have square images as input.
        sX = T.cast(T.sqrt(T.shape(variable)[0] / self.kernel_shape[1]), 'int32')

        #Converts input from 2 to 4 dimensions
        Xr = T.reshape(variable.T, (T.shape(variable)[1], self.kernel_shape[1], sX, sX))

        if self.optimized:
            out_size = T.cast(
                T.ceil((T.shape(Xr)[-1] - T.shape(net.varWeights[layerNum]['w'])[-1] + 1) / np.float32(self.stride)),
                'int32')

            conv_op = FilterActs(stride=self.stride, pad=0 if self.validConvolution else self.kernel_shape[-1] - 1)
            input_shuffled = Xr.dimshuffle(1, 2, 3, 0)  # bc01 to c01b
            filters_shuffled = net.varWeights[layerNum]['w'].dimshuffle(1, 2, 3, 0)  # bc01 to c01b
            filters_flipped = filters_shuffled[:, ::-1, ::-1, :]    # flip rows and columns
            contiguous_input = gpu_contiguous(input_shuffled)
            contiguous_filters = gpu_contiguous(filters_flipped * (self.dropout if self.dropout else 1.0))
            a = conv_op(contiguous_input, contiguous_filters)
            a = a[:, :out_size, :out_size, :]
            #Add bias
            a = a + net.varWeights[layerNum]['b'].dimshuffle(0, 'x', 'x', 'x')
        else:
            a = T.nnet.conv2d(Xr, net.varWeights[layerNum]['w'] *
                              (net.dropOutVectors[layerNum].dimshuffle('x', 'x', 0, 1) if self.dropout else 1.0),
                              border_mode='valid' if self.validConvolution else 'full',
                              subsample=(self.stride, self.stride))

            #Add bias
            a = a + net.varWeights[layerNum]['b'].dimshuffle('x', 0, 'x', 'x')

        if self.pooling:
            if self.optimized:
                #Pooling
                # ds - side of square pool window
                # stride - Defines the stride size between successive pooling squares.
                # Setting this parameter smaller than sizeX produces overlapping pools.
                # Setting it equal to sizeX gives the usual, non-overlapping pools. Values greater than sizeX are not allowed.
                pool_op = MaxPool(ds=self.pooling_shape, stride=self.pooling_shape)
                contiguous_input = gpu_contiguous(a.astype(theano.config.floatX))
                a = pool_op(contiguous_input)
                a = a.dimshuffle(3, 0, 1, 2)       # c01b to bc01
            else:
                #a = downsample.max_pool_2d(a, (self.pooling_shape, self.pooling_shape), ignore_border=False)
                a = pool.max_pool2D(a, (self.pooling_shape, self.pooling_shape), ignore_border=False)
        else:
            if self.optimized:
                a = a.dimshuffle(3, 0, 1, 2)       # c01b to bc01

        a = T.flatten(a, outdim=2).T

        #Sigmoid
        a = self.activation(a, self.pool_size)

        net.varArrayAc.append(a)

#---------------------------------------------------------------------#
# Options instance
#---------------------------------------------------------------------#


class OptionsStore(object):
    def __init__(self,
                 learnStep=0.01,
                 rmsProp=False,
                 mmsmin=1e-10,
                 rProp=False,
                 minibatch_size=1,
                 CV_size=1):
        """
        Container for global network's options.

        :param learnStep: float, learn step to use in *gradient descent* or RMSprop
        :param rmsProp: False or float, whether to use RMSprop or not. If yes - rate of RootMeanSquare. Usually 0.9
        :param mmsmin: float, clip RootMeanSquare to avoid NaN. Default: 1e-10. Reasonable: down to 1e-20
        :param rProp: False or float, use only for **full batch**. If yes - rate to increase next weight's change.
        :param minibatch_size: int, size of batch you use. Can't be changed compiling.
        :param CV_size: int, size of cross validation set. Can't be changed compiling.
        :return: OptionStore object.
        """
        self.learnStep = learnStep  # Learning step for gradient descent
        self.rmsProp = rmsProp  # rmsProp on|off
        self.mmsmin = mmsmin  # Min mms value
        self.rProp = rProp  # For full batch only
        self.minibatch_size = minibatch_size
        self.CV_size = CV_size

    def Printer(self):
        """
        Print out to stdout current options. Useful for debug.

        :return: nothing
        """
        print self.__dict__


#---------------------------------------------------------------------#
# Basic neural net class
#---------------------------------------------------------------------#


class TheanoNNclass(object):
    def __init__(self, opt, architecture):
        """
        The most important class. Here everything combines together.

        Using info defined in OptionStore and Layers - compile Network object.

        :param opt: OptionStore, general network's options.
        :param architecture: list, list of layers to build a network.
        :return: TheanoNNclass object
        """
        self.REPORT = "OK"

        self.architecture = architecture
        self.options = opt
        self.lastArrayNum = len(architecture)

        self.varWeights = []

        # Variables
        self.x = T.matrix("x")
        self.y = T.matrix("y")

        # Weights
        for i in xrange(self.lastArrayNum):
            self.architecture[i].compileWeight(self, i)

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
        self.trainExternal = None

        # predict
        self.predict = None
        self.out = None

        #For external train
        self.metadata = None
        self.unrolledModel = None
        self.unroll()

        # Predict variables
        self.data = T.matrix("data")
        self.varArrayAc = []

        # List of output variables
        self.outputArray = []

    def trainCompile(self):
        """
        Using OptionsStore, Layers - create shared variable and Theno's function to train network.
        Usually, should be call only once for each network.

        :return:  link self.train  with appropriate theano's function
        """

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
        self.outputArray.append(self.varArrayA[-1])

        # Derivatives
        # All variables to gradArray list to show to Theano on which variables we need an gradient
        gradArray = []
        for i in xrange(self.lastArrayNum):
            for k in self.varWeights[i].keys():
                gradArray.append(self.varWeights[i][k])
        self.derivativesArray = T.grad(self.cost, gradArray)

        # RMS
        if self.options.rmsProp:
            for i in xrange(len(self.derivativesArray)):
                mmsp = theano.shared(np.tile(0.0, gradArray[i].get_value().shape).astype(theano.config.floatX),
                                     name="mmsp%s" % (i + 1))  # 0.0 - 1.0 maybe
                self.MMSprev.append(mmsp)
                mmsn = self.options.rmsProp * mmsp + (1 - self.options.rmsProp) * self.derivativesArray[i] ** 2
                #mmsn = T.clip(mmsn, self.options.mmsmin, 1e+15)  # Fix nan if rmsProp
                mmsn = T.clip(mmsn, self.options.mmsmin, np.finfo(np.float32).max)  # Fix nan if rmsProp
                self.MMSnew.append(mmsn)

        # Update values
        for i in xrange(len(self.derivativesArray)):
            if self.options.rmsProp:
                updateVar = self.options.learnStep * self.derivativesArray[i] / self.MMSnew[i] ** 0.5
                self.updatesArray.append((self.MMSprev[i], self.MMSnew[i]))
            else:
                updateVar = self.options.learnStep * self.derivativesArray[i]
            self.updatesArray.append((gradArray[i], gradArray[i] - updateVar))

        self.train = theano.function(inputs=[self.x, self.y],
                                     outputs=self.outputArray,
                                     updates=self.updatesArray,
                                     allow_input_downcast=True)
        return self

    def trainCompileExternal(self):
        """
        It is possible to use external optimisation.

        In case yu decide to use something external - this method will prepare necessary functions.
        So after you should be able to use returned gradient and load updated weights.

        :return:
        """

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

        # Derivatives
        # All variables to gradArray list to show to Theano on which variables we need an gradient
        gradArray = []
        for i in xrange(self.lastArrayNum):
            for k in self.varWeights[i].keys():
                gradArray.append(self.varWeights[i][k])
        self.derivativesArray = T.grad(self.cost, gradArray)

        # Update output array
        self.outputArray.append(self.cost)
        self.outputArray.append(XENT)
        self.outputArray.extend(self.derivativesArray)

        self.trainExternal = theano.function(inputs=[self.x, self.y],
                                             outputs=self.outputArray,
                                             allow_input_downcast=True)
        return self

    def trainCalcExternal(self, model, X, Y):
        """
        Call this method in case you want to use external optimizer.

        :param model: vector, new weights for network.
        :param X: array, data to train on.
        :param Y: array, labels for data.
        :return: (float, array), network's error and weight's gradients
        """

        #Roll, reshape and update shared variables using vector of weights
        self.roll(model)

        #error, ent, grads = self.trainExternal(X, Y)
        res = self.trainExternal(X, Y)

        ent = res[1]
        error = res[0]
        grads = res[2:]

        #Unroll returned by network gradients in the same way as weights.
        count = 0
        for g in grads:
            if count == 0:
                if len(g.shape) == 1:
                    grad = g
                else:
                    grad = g.reshape((-1, ))
            else:
                if len(g.shape) == 1:
                    grad = np.concatenate((grad, g))
                else:
                    r = g.shape[0]
                    c = g.shape[1]
                    grad = np.concatenate((grad, g.reshape((r * c, ))))
            count += 1

        self.errorArray.append(ent)
        print ent, error
        return error, np.float64(grad)

    def trainCalc(self, X, Y, iteration=10, debug=False, errorCollect=False):  # Need to call trainCompile before
        """
        Standard method to train network using labeled data.

        :param X: array, data to train network on.
        :param Y: array, data's labels.
        :param iteration: number of cycles you want network to train on current X
        :param debug: boolean, whether to print some useful info.
        :param errorCollect: boolean, whether to collect network's error in *self.errorArray* field
        :return:
        """

        for i in xrange(iteration):
            error, ent, out = self.train(X, Y)
            self.train_out = out
            if errorCollect:
                self.errorArray.append(ent)
            if debug:
                print ent, error
        return self

    def predictCompile(self, layerNum=-1):
        # Predict activation
        for i in xrange(self.lastArrayNum):
            self.architecture[i].compilePredictActivation(self, i)

        self.predict = theano.function(inputs=[self.x],
                                       outputs=self.varArrayAc[layerNum],
                                       updates=self.updatesArrayPredict,
                                       allow_input_downcast=True)
        return self

    def predictCalc(self, X, debug=False):  # Need to call predictCompile before
        self.out = self.predict(X)  # Matrix of outputs. Each column is a picture reshaped in vector of features
        if debug:
            print 'out.shape:', self.out.shape
        return self

    def getStatus(self):  # Its time for troubles
        print self.REPORT
        return self

    def paramGetter(self):  # Returns the values of model parameters such as [w1, b1, w2, b2] ect.
        model = []
        for i in xrange(self.lastArrayNum):  # Possible use len(self.varArrayB) or len(self.varArrayW) instead
            D = dict()
            for k in self.varWeights[i].keys():
                D[k] = self.varWeights[i][k].get_value()
            model.append(D)
        return model

    def paramSetter(self, loaded):  # Setups loaded model parameters
        assert len(loaded) == self.lastArrayNum, 'Number of loaded and declared layers differs.'
        count = 0
        for l in loaded:
            for k in l.keys():
                self.varWeights[count][k].set_value(np.float32(l[k]))
            count += 1

    def modelSaver(self, folder):  # In cPickle format in txt file
        f = file(folder, "wb")

        #Fix weights with dropout values
        model = self.paramGetter()
        for i in xrange(self.lastArrayNum):
            if self.architecture[i].dropout:
                for k in model[i].keys():
                    if k != 'b':
                        model[i][k] = model[i][k] * self.architecture[i].dropout

        cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        self.getStatus()
        return self

    def modelLoader(self, folder):  # Path to model txt file
        f = file(folder, "rb")
        loadedObject = cPickle.load(f)
        f.close()  # Then we need to update W and B parameters

        #Fix model with declared dropout values
        for i in xrange(self.lastArrayNum):
            if self.architecture[i].dropout:
                for k in loadedObject[i].keys():
                    if k != 'b':
                        loadedObject[i][k] = np.true_divide(loadedObject[i][k], self.architecture[i].dropout)

        self.paramSetter(loadedObject)
        self.getStatus()
        return self

    # For now only for first layer. Second in test mode
    def weightsVisualizer(self, folder, size=(100, 100), color="L", second=False, name='weights'):
        # gradArray -> [w0, b0, w1, b1, ...] RANDOM!11
        W = []
        for i in xrange(self.lastArrayNum):  # Possible use len(self.varArrayB) or len(self.varArrayW) instead
            W.append(self.varWeights[i]['w'])
        # print len(W), W[0].get_value().shape, W[1].get_value().shape DONT CLEAR THIS PLZ
        W1 = W[0].get_value()
        W2 = W[1].get_value()  # Second layer test. Weighted linear combination of the first layer bases
        for w in xrange(len(W1)):
            img = W1[w, :].reshape(size[0], size[1])  # Fix to auto get size TODO
            Graphic.PicSaver(img, folder, "L1_" + str(w), color)
        if second:
            for w in xrange(len(W2)):
                img = np.dot(W1.T, W2[w, :]).reshape(size[0], size[1])
                Graphic.PicSaver(img, folder, "L2_" + str(w), color)
        return self

    def unroll(self):
        l = self.paramGetter()
        meta = []
        count = 0
        for d in l:
            layers_meta = dict()
            for k in sorted(d.keys()):
                layers_meta[k] = d[k].shape
                if count == 0:
                    res = d[k].reshape((-1, ))
                else:
                    res = np.concatenate((res, d[k].reshape((-1, ))))
                count += 1
            meta.append(layers_meta)

        self.unrolledModel = res
        self.metadata = meta

        return self

    def roll(self, a):
        m = self.metadata
        start = 0
        res = []
        for d in m:
            layer = dict()
            for k in sorted(d.keys()):
                if k != 'b':
                    r = d[k][0]
                    c = d[k][1]
                    end = start + r * c
                    layer[k] = a[start:end].reshape((r, c))
                    start = end
                else:
                    r = d[k][0]
                    end = start + r
                    layer[k] = a[start:end].reshape((r, ))
                    start = end

            res.append(layer)

        self.paramSetter(res)
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
