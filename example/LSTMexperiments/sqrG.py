# ---------------------------------------------------------------------#


import numpy as np
import matplotlib.pylab as plt
from scipy import signal


# ---------------------------------------------------------------------#


def noisedSqr():
    #
    number = 10000  # number of points
    time = np.linspace(-np.pi * 100, np.pi * 100, number)
    #
    series = signal.square(time / 100)
    noise = np.random.uniform(0.0, 0.05, number)
    #
    data = series + noise  # input data
    return data, time


# ---------------------------------------------------------------------#

y, x = noisedSqr()

fig = plt.figure(figsize=(200, 9))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x, y, s=5, alpha=0.5, color="blue")
fig.savefig("sqr.png")


# ---------------------------------------------------------------------#
