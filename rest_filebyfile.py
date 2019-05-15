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

files = ['raw_data_features_female_1.mat','raw_data_features_female_2.mat','raw_data_features_female_3.mat','raw_data_features_male_1.mat','raw_data_features_male_2.mat']
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
	feature_set = np.vstack([cyl,hook,lat,palm,spher,tip])
	
	X = feature_set
	g = np.array([0]*30 + [1]*30 + [2]*30 + [3]*30 + [4]*30 + [5]*30)
	#y = np.array([1]*30 + [0]*150)
	one_hot_labels = np.zeros((180, 6))
	for i in range(180):
	    one_hot_labels[i, g[i]] = 1
	
	y = one_hot_labels
	
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 31)
	
	
	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)
	
	
	from sklearn.multiclass import OneVsRestClassifier
	from sklearn.svm import *
	from sklearn import metrics
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.linear_model import *
	
	#classifier = OneVsRestClassifier(LinearSVC(loss='squared_hinge', penalty='l2', dual=False, tol=1, C=10e5))
	for base_clf in ( SVC(kernel='linear'),LinearRegression(),LogisticRegression()):
		classifier = OneVsRestClassifier(base_clf).fit(X_train, y_train)
	#	pred = classifier.predict(X_test)
		score = classifier.score(X_test,y_test)
		print(score)
	# = metrics.f1_score(y_test, pred, average="micro")
	print('--------------------------------------------')