import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from scipy import io as sio

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

#files = ['m1.mat']
files = ['raw_data_features_female_3.mat']
feature_set = []
y = []
for f in files:
	print(f)
	a = sio.loadmat(f)
	cyl = a['cyl_1']
	cyl2 = a['cyl_2']
	cyl = np.append(cyl,cyl2,axis=1)
	hook = a['hook_1']
	hook2 = a['hook_2']
	hook = np.append(hook,hook2,axis=1)
	lat = a['lat_1'] 
	lat2=a['lat_2']
	lat = np.append(lat,lat2,axis=1)
	palm = a['palm_1']
	palm2 = a['palm_2']
	palm = np.append(palm,palm2,axis=1)
	spher = a['spher_1']
	spher2 = a['spher_2']
	spher = np.append(spher,spher2,axis=1)
	tip = a['tip_1']
	tip2 = a['tip_2']
	tip= np.append(tip,tip2,axis=1)
	temp = np.vstack([cyl,hook,lat,palm,spher,tip])
	
	g = np.array([0]*30 + [1]*30 + [2]*30 + [3]*30 + [4]*30 + [5]*30)
	#y = np.array([1]*30 + [0]*150)
	
	one_hot_labels = np.zeros((180, 6))
	for i in range(180):
	    one_hot_labels[i, g[i]] = 1
		   
#	w, h = 6, 180;
#	ok = [[0 for x in range(w)] for y in range(h)] 
#	for i in range(180):
#		ok = [0]*6
#		for j in range(6):
#			ok[j] = one_hot_labels[i][j]
#		y.append(ok)
	for i in one_hot_labels:
		y.append(i)
	for i in temp:
		feature_set.append(i)

y = np.array(y)
feature_setn = np.array(feature_set)
X = feature_setn

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = None)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 30))

classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))
#classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'softmax'))
#classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)
count=0
totl=0
for i in range(36):
	for j in range(6):
		if (y_test[i][j]==1):
			if(y_pred[i][j]==True):
				count=count+1
	totl=totl+1
sd=count/totl
sd=sd*100
print(count)
				

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

acc=0

for i in range(6):
	su = 0
	for j in range(6):
		su = su + cm[j][i]
	acc = acc + (cm[i][i]/su)

acc = acc/6
acc = acc*100
pr = 'Accuracy = '
pr += str(acc)
pr+=' '
pr+=str(sd)
print(pr)

