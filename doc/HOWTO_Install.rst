How to Install and Run
======================

* `Python installation`_
* `Dependencies`_ (numpy, cuda, theano etc.)
* `TNNF installation`_

Python installation
-------------------

We use Python **2.7.4+**
It is true that some (or all) of libraries may work with Python3, but we didn't test this case.

* | Linux
  | Python included in most recent distributive and you can easily run it and check its version.

  In console::

     user$ python -V

  If you don't have Python, you can easily installed it on most distributive executing:

  * For Ubuntu/Debian::

      user$ sudo apt-get install python

  * For Red Hat/CentOS::

      user$ sudo yum install python

  or equivalent for your UNIX based OS.

* | Windows:
  | Choose and download correct version for your system from `here <https://www.python.org/downloads/>`_.
  | Then run and proceed with installation.

  To check python's version run::

    C:\Users\user> python -V

Dependencies
------------

Here are libraries that should be installed before running code:

* Numpy
* Theano
* PIL
* cPickle
* matplotlib
* h5py

Each of these libraries has its own dependencies (which may overlap). All dependencies should be satisfied.

The best suggestion is to use installation instruction for each particular library on their official site:

* Theano (includes numpy installation) - http://deeplearning.net/software/theano/install.html

  * Direct link for fast installation on `Ubuntu <http://deeplearning.net/software/theano/install_ubuntu.html#install-ubuntu>`_
  * Direct link for fast installation on `CentOS <http://deeplearning.net/software/theano/install_centos6.html#install-centos6>`_

* PIL - http://en.wikibooks.org/wiki/Python_Imaging_Library/Getting_PIL
* matplotlib - http://matplotlib.org/1.3.1/users/installing.html
* h5py - http://docs.h5py.org/en/latest/build.html

.. caution::
   Installing Theano on Windows machine may be difficult, buggy and frustrating.

It is obvious Theano's advantage that its (properly written) code is easily can be run on CPU or GPU without editing.

There is nothing special you have to do to run it on CPU. All dependencies will be installed during Theano installation. When talking about GPU - there are a few constraints.

Theano's GPU *limitations*:

* **nVidia GPU only**
* supports only `CUDA <https://developer.nvidia.com/cuda-downloads>`_ (OpenCL support is expected in near future)
* While computing on GPU - use **float32** only. (Float64 is requested, but not implemented yet)

To perform GPU calculation you have to have:

* Supported nVidia GPU
* Appropriate nVidia driver installed
* Appropriate CUDA toolkit installed (both 5.5 and 5.0 are supported)

TNNF installation
--------------------

There is no "installation" needed.

SSBrain is a number of Python modules.

The only thing you need:

#. Download latest version from `GitHub <https://github.com/spaceuniverse/TNNF.git>`_
#. Import it using standard Python syntax:

   .. code:: python

      import fTheanoNNclassCORE 

#. You are ready to go!
