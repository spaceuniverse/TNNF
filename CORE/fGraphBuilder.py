# ---------------------------------------------------------------------#
from matplotlib.pylab import plot, title, xlabel, ylabel, legend, grid, margins, savefig, close, any
import numpy as np
#---------------------------------------------------------------------#


class Graph(object):
    @staticmethod
    def Builder(error=False,
                cv=False,
                accuracy=False,
                name='./GraphBuilder_default_name.png',
                **kwargs):

        #Colors
        Colors = ['c', 'm', 'y', 'k']
        color_idx = 0

        #Add declared dict
        allDicts = dict()
        allDicts['error'] = error
        allDicts['cv'] = cv
        allDicts['accuracy'] = accuracy

        #Add kwargs
        for k in kwargs.keys():
            allDicts[k] = kwargs[k]

        #Define valid dicts
        validDicts = dict()
        for k in allDicts.keys():
            if any(allDicts[k]):
                validDicts[k] = allDicts[k]

        #Define longest (with more points) graph
        lenLongestDict = 0
        keylongestDict = None
        for k in validDicts.keys():
            if len(validDicts[k]) > lenLongestDict:
                lenLongestDict = len(validDicts[k])
                keylongestDict = k

        #Calc axes's step
        stepsDict = dict()
        for k in validDicts.keys():
            s = np.true_divide(lenLongestDict, len(validDicts[k]))
            stepsDict[k] = np.round(np.arange(0, lenLongestDict, s))

        #plot
        for k in validDicts.keys():
            if k == 'error':
                plot(stepsDict[k], validDicts[k], 'r,', markeredgewidth=0, label='Error')
            elif k == 'cv':
                plot(stepsDict[k], validDicts[k], 'g.', markeredgewidth=0, label='CV')
            elif k == 'accuracy':
                plot(stepsDict[k], validDicts[k], 'b.', markeredgewidth=0, label='Accuracy')
            else:
                plot(stepsDict[k], validDicts[k], str(Colors[color_idx] + '.'), markeredgewidth=0, label=k)
                color_idx += 1

        #Titles
        title('Error vs epochs', fontsize=12)
        xlabel('epochs', fontsize=10)
        ylabel('Error', fontsize=10)
        legend(loc='upper right', fontsize=10, numpoints=3, shadow=True, fancybox=True)

        #Grid
        grid()
        margins(0.04)
        savefig(name, dpi=120)
        close()


#---------------------------------------------------------------------#
# EXAMPLE
#d = np.sin(np.arange(0, 20, 0.01))
#c = np.sin(np.arange(0, 20, 0.3))
#t = np.arange(0, 4, 0.1)
#Graph.Builder(error=d, cv=c, test=t)
#---------------------------------------------------------------------#