Simple AutoEncoder
==================

* `Data`_
* `Neural Network`_
* `How it performs`_

Here I'll describe *second step* in understanding what TNNF can do for you.
Using MNIST data - let's create simple (one layer) sparse *AutoEncoder (AE)*, train it and visualise its weights.
This will give understanding of how to compose a little bit complicate networks in TNNF (two layers) and how sparse AE works.


Data
----

To train our AE - let's use widely known MNIST data set.

You can download it in *.csv* format here: http://www.pjreddie.com/projects/mnist-in-csv

* `Train set <http://www.pjreddie.com/media/files/mnist_train.csv>`_
* `Test set <http://www.pjreddie.com/media/files/mnist_test.csv>`_

As *.csv* format is comparatively slow and increases train time significantly we recommend to
use `HDF <https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_ to store data on your drive.

It takes you to install **h5py** package to start use it. `Here <http://www.h5py.org/>`_ is more about it.

.. note::

   It's better to `download <https://pypi.python.org/pypi/h5py>`_ their package instead of trying to install through *pip*.

Once you install *h5py* you need to convert *.csv* to *HDF*. I've prepared a short script in Python for you to do this.

You can download it directly from `GitHub <https://github.com/spaceuniverse/TNNF/tree/master/doc/src/HOWTOs/Data/CSVtoHDF.py>`_