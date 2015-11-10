# README

This repo **is a fork of original** repo, copied for experiments.
All actual development is going in original repo.

###

Some file and methods names:

TNNF - Tiny Neural Net Framework (All files and classes)

Online auto-generated docs: http://tnnf.readthedocs.org/en/latest/index.html

###

Features:

+ feedforward neural nets
+ lstm neural nets
+ maxout
+ softmax
+ sigmoid
+ dropout
+ unsupervised feature learning and deep learning
+ saving and loading models
+ rmsProp
+ rProp
+ minibatches
+ PCA whitening
+ convolution

###

All CORE files of library:

fTheanoNNclassCORE.py - main theano neural net class
fSimpleTest.py - simple tests for framework methods and functions
fSimpleExperiments.py - file for different trash experiments with framework
fGraphBuilder.py - methods for building graphic, such as "train error" or "cv error"
fDataWorkerCORE.py - all data manipulation methods
fCutClassCORE.py - methods which could be usefuul for working with images (can cut them for squares etc.)

###

For using test or examples you may need MNIST dataset which not contained in repo.
U can download it here:
http://yann.lecun.com/exdb/mnist/ - original
http://www.pjreddie.com/projects/mnist-in-csv/ - in csv format

###