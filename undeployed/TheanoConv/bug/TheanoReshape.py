__author__ = 'rhrub'


import theano.tensor as T
import theano
import numpy as np




data = theano.shared(np.random.randn(10, 3, 7, 7).astype(theano.config.floatX), name='data')

#res = data.reshape((T.shape(data)[0], -1))
res = T.mean(data, axis=1)
res = T.sum(res, axis=0)

t = theano.function(inputs=[],
                    outputs=res)

print t().shape