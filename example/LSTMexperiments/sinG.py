# ---------------------------------------------------------------------#


import numpy as np
import matplotlib.pylab as plt


# ---------------------------------------------------------------------#


def noisedSin():
    #
    number = 10000  # number of points
    time = np.linspace(-np.pi * 100, np.pi * 100, number)
    #
    # y = sin(x / 2) * 2 + 3
    #
    series = np.sin(time / 2.0) * 2.0 + 3.0
    noise = np.random.uniform(0.0, 0.05, number)
    #
    data = series + noise  # input data
    return data, time


# ---------------------------------------------------------------------#

'''
y, x = noisedSin()

fig = plt.figure(figsize=(200, 9))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x, y, s=5, alpha=0.5, color="blue")
fig.savefig("sinGf.png")
'''

# ---------------------------------------------------------------------#
