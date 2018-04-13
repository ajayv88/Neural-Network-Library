import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile,join
# from sklearn

from skimage.feature import hog
import cv2
import sys
# from tempfile import TemporaryFile
from keras.models import model_from_json
from sklearn import decomposition

values = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

k = 0
dic = {}
for p in values:
	dic[k] = p
	k += 1


df = pd.read_csv("color.csv")
test_df = pd.read_csv("test.csv")
df = df.iloc[0:11000]
test_df = test_df.iloc[0:400]
classes = df['0'].iloc[0:11000].values
test_classes = test_df['0'].values
del df['0']
del test_df['0']
x_train = df.iloc[0:11000].values
x_train = x_train.astype('float32')
x_train /= np.max(x_train)
x_test = test_df.iloc[0:400].values
print x_test.shape
x_test = x_test.astype('float32')
x_test /= np.max(x_test)


# from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import SGD

# train_data = df.values
print x_train.shape, x_test.shape
input_shape = x_train.shape[0]
test_input_shape = x_test.shape[0]
x_train = x_train.reshape(input_shape, 25, 25, 3)
x_test = x_test.reshape(test_input_shape, 25, 25, 3)
one_hot_labels = keras.utils.to_categorical(classes, num_classes=36)
one_hot_labels1 = keras.utils.to_categorical(test_classes, num_classes=36)

targets = np.zeros(x_train.shape[0],36) + 0.01 
i = 0
for index, row in x_train.iterrows():
	value = classes.iloc[index]
	print value, type(value)
	target = np.zeros(36) + 0.01
	target[int(value)] = 0.99
	categories.iloc[i] = target
	print i
	i += 1

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(25,25,3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(36, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, one_hot_labels, batch_size=32, epochs=200, verbose=1, validation_data=(x_test, one_hot_labels1))
score = model.evaluate(x_test, one_hot_labels1, batch_size=32,verbose=1)
# print score

prediction = model.predict(x_test)
check = []
pred = []
# print prediction.shape
for row in prediction:
	pred.append(np.argmax(row))

for row in one_hot_labels1:
	check.append(np.argmax(row))

# print check

# print check

# for x in np.nditer(prediction):
# 	pred.append(x)
scorecard = 0.0
for i in range(len(check)):
	if pred[i] == check[i]:
		scorecard += 1.0

print float(float(scorecard)/float(len(check)))


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print "Saved model to disk"

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print "Loaded model from disk"

mypath = sys.argv[1]
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]


def validate(files):
	score = 0.0
	total_files = len(files)
	wrong_pred = []
	for file_name in files:
		file_name = mypath+"/"+file_name
		img = cv2.imread(file_name)
		result = file_name[13]
		# print result
		img = cv2.resize(img, (25,25))
		img = img.reshape(1,25,25,3)		
		#print file_name
		loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		arr = loaded_model.predict_classes(img)
		# res = np.argmax(arr)
		print dic[arr[0]], result
		if dic[arr[0]] == result:
			score += 1.0
		else:
			wrong_pred.append((result, dic[arr[0]]))
		
	final_score = float(float(score)/float(total_files))
	return final_score*100.0, wrong_pred


final_score, wrong_preds = validate(files)
print final_score
print wrong_preds
# load json and create model


# score = loaded_model.evaluate(x_test, one_hot_labels1, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
