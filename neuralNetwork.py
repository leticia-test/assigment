# this is a script to train an evaluate a neural network using cross validation. It also provides a visualization function based in the t-distributed Stochastic Neighbor Embedding algorithm. This algorithm allows for visualization of high dimensional data
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
from keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
import matplotlib
import pylab as Plot

#this function allows to visualize high dimensional data
def visualize_data(inputData,labels,plotTitle):
	modelT = TSNE(n_components=2, random_state=0)
	YY = modelT.fit_transform(inputData) 
	Plot.scatter(YY[:,0], YY[:,1], 20, labels);
	Plot.title(plotTitle)
	Plot.legend()
	cmap = matplotlib.cm.get_cmap(None)
	patch0 = mpatches.Patch(color=cmap(0.0/6.0), label=encoder.inverse_transform(0))
	patch1 = mpatches.Patch(color=cmap(1.0/6.0), label=encoder.inverse_transform(1))
	patch2 = mpatches.Patch(color=cmap(2.0/6.0), label=encoder.inverse_transform(2))
	patch3 = mpatches.Patch(color=cmap(3.0/6.0), label=encoder.inverse_transform(3))
	patch4 = mpatches.Patch(color=cmap(4.0/6.0), label=encoder.inverse_transform(4))
	patch5 = mpatches.Patch(color=cmap(5.0/6.0), label=encoder.inverse_transform(5))
	patch6 = mpatches.Patch(color=cmap(6.0/6.0), label=encoder.inverse_transform(6))
	Plot.legend(handles=[patch0,patch1,patch2,patch3,patch4,patch5,patch6],fontsize=8)
	Plot.show();

#ax = Plot.gca()
#leg = ax.get_legend()
#leg.legendHandles[0].set_color('red')
#leg.legendHandles[1].set_color('yellow')

#Plot.legend((0,1,2,3,4,5,6),encoder.inverse_transform([0,1,2,3,4,5,6]))


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Read the dataset
dataSetFileName = '/home/ubuntu/leticia/data.csv'
df = pd.read_csv(dataSetFileName)

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
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# define k-fold cross validation  
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
cvscores = []
cFold = 1
for train, test in kfold.split(X, Y):
	print('cFold' + str(cFold))	
  # create model
	model = Sequential()
	model.add(Dense(12, input_dim=6, init='uniform', activation='relu'))
	model.add(Dense(8, init='uniform', activation='relu'))
	model.add(Dense(7, init='uniform', activation='sigmoid'))	
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
	model.fit(X[train], dummy_y[train], nb_epoch=10, batch_size=10, verbose=1)
	# evaluate the model
	scores = model.evaluate(X[test], dummy_y[test], verbose=0)
	#save model
	#model.save('/home/ubuntu/leticia/model_' + str(cFold) + '.h5')     
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
	cFold = cFold + 1
 
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))



#y_pred = model.predict(X[test])
#y_pred = np.argmax(y_pred, axis=1)
y_pred = model.predict_classes(X[test])
#p = model.predict_proba(X[test])

numLabels = encoder.transform(['walk','trot','canter','standing','backwards','jump','other'])
#print(encoder.transform(['walk','trot','canter','standing','backwards','jump','other']))
targetLabels = encoder.inverse_transform([0,1,2,3,4,5,6]);
#target_names = []
#print('4-------------------------------------')
print(classification_report(np.argmax(dummy_y[test],axis=1), y_pred,target_names=targetLabels))
print(confusion_matrix(np.argmax(dummy_y[test],axis=1),y_pred))

#layer_name = model.layers[0].name
#intermediate_layer_model = Model(input=model.input,
                                # output=model.get_layer(layer_name).output)
#intermediate_output = intermediate_layer_model.predict(X[train])
#print(intermediate_output)



print( "Run t-SNE through layers")
#I run a small subset to avoid memory issues
smallSubSet = '/home/ubuntu/leticia/smallData.csv'
df2 = pd.read_csv(smallSubSet)

#normalize and preprocess input
std_scale2 = preprocessing.StandardScaler().fit(df2[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']])
df_std2 = std_scale.transform(df2[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']])
dataset2 = df2.values
Y2 = dataset2[:,1]
X2 = df_std2
encoder2 = LabelEncoder()
encoder2.fit(Y2)
encoded_Y2 = encoder2.transform(Y2)
labels = encoded_Y2;

visualize_data(X2,labels,'input')

#obtain and visualize data from intermediate layers
layer_name = model.layers[0].name
intermediate_layer_model = Model(input=model.input,output=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(X2)
visualize_data(intermediate_output,labels,'layer 0')

layer_name = model.layers[1].name
intermediate_layer_model = Model(input=model.input,output=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(X2)
visualize_data(intermediate_output,labels,'layer 1')

layer_name = model.layers[2].name
intermediate_layer_model = Model(input=model.input,output=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(X2)
visualize_data(intermediate_output,labels,'layer 2')


