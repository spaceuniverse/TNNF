import numpy as np
import unittest
import os
import sys
sys.path.append('../../../CORE')
from fTheanoNNclassCORE import OptionsStore, TheanoNNclass, NNsupport, FunctionModel, LayerNN
from fDataWorkerCORE import csvDataLoader
from matplotlib.pylab import plot, title, xlabel, ylabel, legend, grid, margins, savefig, close, xlim, ylim

dataSize = 1000
dataFeatures = 2

#Supposed boundary line
#Where x - [0] row, f(x) - [1] row in data
# if f(x) < -x + 1 - then label = 1
#                    else label = 0

#Create random data [0, 1)
data = np.random.rand(dataFeatures, dataSize)

#Create random cross-validation [0, 1)
CV = np.random.rand(dataFeatures, dataSize)

#Create array for labeled data
labels = np.zeros((1, dataSize))

#Calc labels based on supposed boundary decision function analytically
labels[0, :] = -data[0, :] + 1 > data[1, :]

#Let's draw our data and decision boundary we use to divide it
#Calc decision boundary
x = np.arange(0, 1.0, 0.02)
y = -x + 1

#Draw labeled data
#Uncomment next part if you want to visualise input data
'''
ones = np.array([[], []])
zeros = np.array([[], []])
for i in xrange(dataSize):
    if labels[0, i] == 0:
        zeros = np.append(zeros, data[:, i].reshape(-1, 1), axis=1)
    else:
        ones = np.append(ones, data[:, i].reshape(-1, 1), axis=1)

xlim(0, 1)
ylim(0, 1)

plot(ones[0, :], ones[1, :], 'gx', markeredgewidth=1, label='Ones')
plot(zeros[0, :], zeros[1, :], 'bv', markeredgewidth=1, label='Zeros')
plot(x, y, 'r.', markeredgewidth=0, label='Decision boundary')
xlabel('X_1')
ylabel('X_2')

legend(loc='upper right', fontsize=10, numpoints=3, shadow=True, fancybox=True)
grid()
savefig('data_visualisation.png', dpi=120)
close()
'''

#Check average
avgLabel = np.average(labels)

print data.shape
print 'Data:\n', data[:, :6]
print labels.shape
print 'Average label (should be ~ 0.5):', avgLabel

#For now we have labeled and checked data.
#Let's try to train NN to see, how it solves such task
#NN part

#Common options for whole NN
options = OptionsStore(learnStep=0.05,
                       minibatch_size=dataSize,
                       CV_size=dataSize)

#Layer architecture
#We will use only one layer with 2 neurons on input and 1 on output
L1 = LayerNN(size_in=dataFeatures,
             size_out=1,
             activation=FunctionModel.Linear)

#Compile NN
NN = TheanoNNclass(options, (L1, ))

#Compile NN train
NN.trainCompile()

#Compile NN oredict
NN.predictCompile()

#Number of iterations (cycles of training)
iterations = 1

#Set step to draw
drawEveryStep = 1000

#Main cycle
for i in xrange(iterations):

    #Train NN using given data and labels
    NN.trainCalc(data, labels, iteration=1, debug=True)

    #Draw data, original and current decision boundary every drawEveryStep's step
    if i % drawEveryStep == 0:

        #Get current coefficient for our network
        b = NN.varWeights[0]['b'].get_value()[0]
        w1 = NN.varWeights[0]['w'].get_value()[0][0]
        w2 = NN.varWeights[0]['w'].get_value()[0][1]

        #Recalculate predicted decision boundary
        y_predicted = -w1 * x / w2 + (0.5 - b) / w2

        #Limit our plot by axes
        xlim(0, 1)
        ylim(0, 1)

        #Plot predicted decision boundary
        plot(x, y_predicted, 'g.', markeredgewidth=0, label='Predicted boundary')

        #Plot original decision boundary
        plot(x, y, 'r.', markeredgewidth=0, label='Original boundary')

        #Plot raw data
        plot(data[0, :], data[1, :], 'b,', label='data')

        #Draw legend
        legend(loc='upper right', fontsize=10, numpoints=3, shadow=True, fancybox=True)

        #Eanble grid
        grid()

        #Save plot to file
        savefig('data' + str(i) + '.png', dpi=120)

        #Close and clear current plot
        close()