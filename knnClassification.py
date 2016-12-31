# script to train and evaluate a k nearest neighbor classifier using cross validation.

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# fix random seed for reproducibility
seed = 5
np.random.seed(seed)

# Read the dataset
dataSetFile =  '/home/ubuntu/leticia/data.csv'
df = pd.read_csv(dataSetFile)

#normalize data (z-scores)
std_scale = preprocessing.StandardScaler().fit(df[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']])
df_std = std_scale.transform(df[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']])
dataset = df.values
Y = dataset[:,1]
X = df_std

# one hot encoding of the output variable to encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)

#this is a configurable parameter, i just chose 15 arbitrarily
n_neighbors = 15

for weights in ['uniform', 'distance']:
    
    print(weights)
        
    # define k-fold cross validation
    kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    cvscores = []
    cFold = 1
    for train, test in kfold.split(X, Y):
        print('cFold' + str(cFold))
        # use k nearest Neighbours Classifier, fit and evaluate the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X[train], Y[train])
        accuracy = clf.score(X[test],Y[test])           
        print("%s: %.2f%%" % ('accuracy', accuracy*100))
        cvscores.append(accuracy * 100)
        cFold = cFold + 1
        y_pred = clf.predict(X[test])
        
        targetLabels = encoder.inverse_transform([0,1,2,3,4,5,6]);        
        print(classification_report(Y[test], y_pred,target_names=targetLabels))
        print(confusion_matrix(Y[test],y_pred))

    
    print('%.2f%% (+/- %.2f%%)' % (np.mean(cvscores), np.std(cvscores)))

	

    
    
 