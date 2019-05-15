import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import io as sio
a = sio.loadmat('raw_data_features_male_1.mat')
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
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


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

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'softmax'))
#classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))