# Assignment
classification of gyro/accel data

I approached the assignment using two types of classifiers: a neural network and a k-nn classifier. 

It is important to note that I did not own an AWS account, so I created a free one for this assignment. Free AWS accounts do not offer powerful software, therefore the training tasks, especially for the neural network run very slowly. For this reason, I used a somewhat simple network, and only 2 folds for cross validation.

I am providing three python scripts:

- gridSearch.py: script to do a grid search to find the optimal hyperparameters for the neural network. It is small search, with a small range of parameters due to the lack of powerful hardware

- neuralNetwork.py: this is a script to train an evaluate a neural network using cross validation. It also provides a visualization function based in the t-distributed Stochastic Neighbor Embedding algorithm. This algorithm allows for visualization of high dimensional data

- knnClassification: script to train and evaluate a k nearest neighbor classifier using cross validation.

The neural network data does not achieve a very high accuracy. This is probably due to the fact that some classes have a very small amount of data, and that the training and network architecture is very “shallow” due to the lack of powerful hardware. The visualization of the output of the layers reveals clusters of data that start to separate through layers, however in the output of layer 2 there is still some “confusion” among classes. 

The AMI used was vict0rsch-1.0 - ami-10762170.

Since these are relatively simple scripts I did not organize functions into libraries.
