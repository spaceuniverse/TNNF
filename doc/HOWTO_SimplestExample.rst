Simplest example
================

I'll try do describe how to use TNNF to solve a simplest artificial task I was able to design.

Let's create a simple classifier that will assign 0 or 1 class to data from input. And as we don't want random classification, let's try to train our classifier on previously manually labeled data.

Data
----

Let's imagine we have some amount of manually labeled data (to label data we will use particular "decision rule"). This data will be used to train our classifier.

In this particular task I assume we have to classify a pair of numbers :math:`(X1, X2)`, where :math:`X1, X2 \in [0, 1)` and assign to each pair 0 or 1.