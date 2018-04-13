from neural_network import NeuralNetwork
from neural_network import NeuralNetwork
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from os import listdir
from os.path import isfile,join
import hickle as hkl

values = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35']
values2 = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


k = 0
dic = {}
dic2 = {}

for p in range(0,len(values)):
	dic[values[p]] = k
	dic2[k] = values2[p] 
	k += 1

nn = [0,0,0,0,0,0,0]
input_nodes = 625
output_nodes = 36
learning_rate = 0.01
activation = 'sigmoid'
optimizer = 'adam'
batch_size=32

nn[0] = NeuralNetwork(input_nodes,(200, 100),output_nodes,learning_rate, activation, batch_size,optimizer)
nn[1] = NeuralNetwork(input_nodes,(300, 200),output_nodes,learning_rate, activation, batch_size,optimizer)
nn[2] = NeuralNetwork(input_nodes,(400, 300),output_nodes,learning_rate, activation, batch_size,optimizer)
nn[3] = NeuralNetwork(input_nodes,(300, 200),output_nodes,learning_rate, 'relu', batch_size,optimizer)
nn[4] = NeuralNetwork(input_nodes,(600, 500),output_nodes,learning_rate, activation, batch_size,optimizer)
nn[5] = NeuralNetwork(input_nodes,(600, 500),output_nodes,learning_rate, activation, batch_size,optimizer)
nn[6] = NeuralNetwork(input_nodes,(600, 600),output_nodes,learning_rate, activation, batch_size,optimizer)

hickle_file = [0,0,0,0,0,0,0]

hickle_file[0] = './hickle_files/1sigmoid_200_100_86.hkl'
hickle_file[1] = './hickle_files/1sigmoid_300_200_88.hkl'
hickle_file[2] = './hickle_files/1sigmoid_400_300_85.hkl'
hickle_file[3] = './hickle_files/1relu_300_200_87.hkl'
hickle_file[4] = './hickle_files/1sigmoid_500_500_90.hkl'
hickle_file[5] = './hickle_files/1sigmoid_600_500_88.hkl'
hickle_file[6] = './hickle_files/1sigmoid_600_600_90.hkl'

i = 0

for hickle in hickle_file:
	data2 = hkl.load(hickle)
	for j in range(0,3):
		nn[i].theta[j] = data2['theta'+str(j+1)]
		nn[i].b[j] = data2['b'+str(j+1)]
	i += 1


img = cv2.imread('./number_plates/download.png')

#grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.waitKey(0)

#binarize 
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
cv2.waitKey(0)

#find contours
ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
prediction = ''
for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = img[y:y+h+5, x:x+w+5]
    
    if roi.shape[0] > 70 or roi.shape[0] < 60:
    	continue
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(gray_roi, (25,25)).reshape(1,625)
    res = [0, 0, 0, 0, 0, 0, 0]
    arr = [0, 0, 0, 0, 0, 0, 0]
    for i in range(0,7):
	res[i] = nn[i].classify(np.asfarray(roi)/255.0 * 0.99 + 0.01)
	arr[i] = np.argmax(res[i])
    sorted(arr)
    max_occur = 1
    max_val = arr[0]
    hashmap = {}
    for i in arr:
	if i in hashmap:
		hashmap[i] += 1
		if hashmap[i] > max_occur:
			max_val = i
			max_occur = hashmap[i]
	else:
		hashmap[i] = 1
    prediction += dic2[max_val]
    cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)


print "The number plate is: " + prediction
cv2.imshow('marked areas',img)
cv2.waitKey(0)
