#this script does a grid search to find the optimal hyperparameters

import pickle
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import Adam
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

# neural network:
def create_model(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0):
	# create model
	model = Sequential()	
	model.add(Dense(12, input_dim=6, init='uniform', activation='relu'))
	model.add(Dense(8, init='uniform', activation='relu'))
	model.add(Dense(7, init='uniform', activation='sigmoid'))
	# Compile model    
	optimizer = Adam(lr, beta_1, beta_2, epsilon, decay)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

# fix random seed for reproducibility
seed = 5
np.random.seed(seed)

# Read the dataset 
dataSetFile = '/home/ubuntu/leticia/data.csv'
df = pd.read_csv(dataSetFile)

#normalization
std_scale = preprocessing.StandardScaler().fit(df[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']])
df_std = std_scale.transform(df[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']])
print(df_std)
dataset = df.values
Y = dataset[:,1]
X = df_std

# one hot encoding of the output variable to encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

print('keras classifier model')
model = KerasClassifier(build_fn=create_model, verbose=1)

print('grid search')
#define the grid search parameters 
#due to the lack of time I only did a small search
#batch_size = [10, 20, 40, 60, 80, 100]
batch_size = [10, 20]
#epochs = [10, 50, 100]
epochs = [10, 13]
#lrs = [0.001, 0.01, 0.1, 0.2, 0.3]
lrs = [0.001, 0.01]
#beta_1s = [0.8,0.9,0.99]
#beta_2s = [0.8,0.9,0.99]
#epsilons = [1e-09, 1e-08, 1e-07]
#decays = [0.0, 0.001, 0.01]
#param_grid = dict(batch_size=batch_size, nb_epoch=epochs, lr = lrs, beta_1=beta_1s, beta_2=beta_2s, epsilon=epsilons, decay = decays)
param_grid = dict(batch_size=batch_size, nb_epoch=epochs, lr = lrs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, encoded_Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


