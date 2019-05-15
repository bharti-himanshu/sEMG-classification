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

simplefilter(action='ignore', category=FutureWarning)

#ALL
files = ['raw_data_features_female_1.mat','raw_data_features_female_2.mat','raw_data_features_female_3.mat','raw_data_features_male_1.mat','raw_data_features_male_2.mat']
files_imf = ['imf_1_data_features_female_1.mat','imf_1_data_features_fehhmale_2.mat','imf_1_data_features_female_3.mat','imf_1_data_features_male_1.mat','imf_1_data_features_male_2.mat']
##Females
#files = ['raw_data_features_female_1.mat','raw_data_features_female_2.mat','raw_data_features_female_3.mat']
#files_imf = ['imf_1_data_features_female_1.mat','imf_1_data_features_female_2.mat','imf_1_data_features_female_3.mat']
##Males
#files = ['raw_data_features_male_1.mat','raw_data_features_male_2.mat']
#files_imf = ['imf_1_data_features_male_1.mat','imf_1_data_features_male_2.mat']


maxacc  = 0
for itern in range(10):
	feature_set = []
	y = []
	for f in range(1):
		print(f)
		a = sio.loadmat(files[f])
		imf = sio.loadmat(files_imf[f])
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
		icyl = imf['cyl_1']
		icyl2 = imf['cyl_2']
		icyl = np.append(icyl,icyl2,axis=1)
		cyl = np.append(cyl,icyl,axis=1)
		ihook = imf['hook_1']
		ihook2 = imf['hook_2']
		ihook = np.append(ihook,ihook2,axis=1)
		hook = np.append(hook,ihook,axis=1)
		ilat = imf['lat_1']
		ilat2=imf['lat_2']
		ilat = np.append(ilat,ilat2,axis=1)
		lat = np.append(lat,ilat,axis=1)
		ipalm = imf['palm_1']
		ipalm2 = imf['palm_2']
		ipalm = np.append(ipalm,ipalm2,axis=1)
		palm = np.append(palm,ipalm,axis=1)
		ispher = imf['spher_1']
		ispher2 = imf['spher_2']
		ispher = np.append(ispher,ispher2,axis=1)
		spher = np.append(spher,ispher,axis=1)
		itip = imf['tip_1']
		itip2 = imf['tip_2']
		itip= np.append(itip,itip2,axis=1)
		tip= np.append(tip,itip,axis=1)
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
	
	classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 60))
	
	classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))
	#classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))
	classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'softmax'))
	#classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
	
	classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
	
	classifier.fit(X_train, y_train, batch_size = 10, epochs = 100, verbose=1)
	
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
	
	
	# serialize weights to HDF5
	classifier.save("model1.h5")
	
	acc=0
	
	for i in range(6):
		su = 0
		for j in range(6):
			su = su + cm[j][i]
		acc = acc + (cm[i][i]/su)
	
	acc = acc/6
	acc = acc*100
	if acc > maxacc :
		maxacc = acc
		classifier.save("neural.h5")
	print(acc)
	print('-------------------------')
pr = 'Accuracy = '
pr += str(maxacc)
print(pr)
	
