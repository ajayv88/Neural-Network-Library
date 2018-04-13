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


input_nodes = 625
output_nodes = 36
learning_rate = 0.01

f = open("full_train.csv","r")
f_list = f.readlines()[1:48001]
f.close()


def validate(files):
	score = 0.0
	total_files = len(files)
	wrong_pred = []
	d = {}
	for file_name in files:
		file_name = mypath+"/"+file_name
		img = cv2.imread(file_name)
		result = file_name[14].upper()
		# print result
		img = cv2.resize(img,(25,25))
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(1,625)
		res = nn.classify(np.asfarray(gray_img)/255.0 * 0.99 + 0.01)
		arr = np.argmax(res)
		# print arr
		prediction = dic2[arr]
		if prediction == result:
			score += 1.0
		else:
			if result not in d:
				d[result] = 1
			else:
				d[result] += 1
			wrong_pred.append(result)
		
	final_score = float(float(score)/float(total_files))
	# print d
	return final_score*100.0, wrong_pred


mypath = sys.argv[1]
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
hidden_inp1 = int(sys.argv[2])
hidden_inp2 = int(sys.argv[3])
hidden_inp3 = int(sys.argv[4])
activation = sys.argv[5]


print "--- Training with activation: " + activation + " hidden1 size: " + str(hidden_inp1) + " and hidden2 size: " + str(hidden_inp2)
batch_size = 64
nn = NeuralNetwork(input_nodes,(hidden_inp1, hidden_inp2, hidden_inp3),output_nodes,learning_rate, activation, batch_size, 'adam', iterations)
epochs = 25
ep = 1
validation_accuracy = 0.0

X_train = []
Y_train = []

for record in f_list:
	all_values = record.split(',')
	scaled_input = np.asfarray(all_values[1:])/255.0 * 0.99 + 0.01
	# print scaled_input.shape
	y = np.zeros(output_nodes) + 0.01
	y[int(dic[all_values[0]])] = 0.99
	X_train.append([scaled_input])
	Y_train.append([y])


X_train = np.concatenate(X_train)
Y_train = np.concatenate(Y_train)


for i in range(epochs):
	k = 0
	loss = 0.0
	nn.t = 0
	for i in tqdm(range(0,len(X_train), batch_size)):
		x_train = X_train[i:i+batch_size]
		y_train = Y_train[i:i+batch_size]
		nn.t += 1
		loss += nn.train(x_train, y_train)

	validation_accuracy, wrong_predictions = validate(files)
	print "Loss after epoch " + str(ep) + " is : " + str(float(loss)/float(48500))
	print "Validation accuracy after epoch " + str(ep) + " is " + str(validation_accuracy)
	past_loss.append(float(loss)/float(48500))
	ep += 1	


data = {'theta': nn.theta, 'b': nn.b}
hickle_file = './hickle_files/1' + activation + '_' + str(hidden_inp1) + '_' + str(hidden_inp2) + '_' + str(hidden_inp3) + '_' + str(int(validation_accuracy)) + '.hkl'
hkl.dump(data, hickle_file)