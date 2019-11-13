#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
from os import walk
import random
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Input, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import load_model
from keras import backend as K
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from imutils import paths
from keras.callbacks import TensorBoard
import time
import cv2
import os
#10Hz
#Train_path = 'C:/Users/denni/Desktop/實驗/DateSet/兩變數/train_2 (測試用)'
Train_path = 'C:/Users/denni/Desktop/實驗/DateSet/兩變數/train_2 (原始)'
Test_path = 'C:/Users/denni/Desktop/實驗/2019-06-01-1.txt'
Image_Test_path = 'result/'
NUM_CLASS = 2
windows = 50
BATCH_SIZE = 32
EPOCHS = 500
height = 240
weight = 320
depth = 3

def load_data():
	print("[INFO] loading data...")	
	label = []
	dataSet = []
	filenames = []
	flag = 0
	count = 1
	for root, dirs, files in walk(Train_path):
		for x in range(len(files)):
			filename = str(root) + "/" + str(files[x])				
			filenames.append(filename)
	random.seed(42)
	random.shuffle(filenames)	
	for x in range(len(filenames)):
		label.append(filenames[x].split("\\")[1].split("/")[0])
		with open(filenames[x], "r") as fp:
			all_lines = fp.readlines()
			data1 = []
			data2 = []
			data3 = []
			for y in range(100):
				if(len(all_lines) == y):
					flag = 1
					break
				try:
					data1.append(float(all_lines[y].split()[0]))
					data2.append(float(all_lines[y].split()[1]))
					data3.append(float(all_lines[y].split()[2]))					
				except ValueError:
					pass
			if(flag == 1):
				less = 100 - len(data1)
				for x in range(less):
					data1.append(sum(data1[len(data1) - 10 : len(data1)])/ 10)
					data2.append(sum(data2[len(data2) - 10 : len(data2)])/ 10)
					data3.append(sum(data3[len(data3) - 10 : len(data3)])/ 10)
					#data1.append(sum(data1)/len(data1))
					#data2.append(sum(data2)/len(data2))
					#data3.append(sum(data3)/len(data3))							
				flag = 0
		if(len(data1) != 0):
			A_SMV = []
			for x in range(len(data1)):
				A_SMV.append(np.sqrt(np.power(data1[x], 2) + np.power(data2[x], 2) + np.power(data3[x], 2)))
			tmp = np.array([data1, data2, data3, A_SMV]).T		
			#tmp = np.array([data1, data2, data3]).T		
			#tmp = np.array([data1, data2, data3, A_SMV])
			dataSet.append(tmp)	
	dataSet = np.array(dataSet)			
	
	label = to_categorical(label)	
	filenames = np.array(filenames)
	
	print("filenames:" + str(len(filenames)))
	print("DataSet:" + str(dataSet.shape))
	print("Label:" + str(len(label)))	
	return dataSet, label, filenames
	

def splid_data(X, Y):
	print("[INFO] splid the dataset...")
	#split_Train = int(X.shape[0] * 0.5)
	split_Train = int(X.shape[0] * 0.1)
	split_Valid = int(X.shape[0] * 0.6)	
	tmpX, tmpX1, tmpX2 = X[:split_Train], X[split_Train:split_Valid], X[split_Valid:]
	tmpY, tmpY1, tmpY2 = Y[:split_Train], Y[split_Train:split_Valid], Y[split_Valid:]
	#trainX, testX = X[:split_Train], X[split_Train:]
	#trainY, testY = Y[:split_Train], Y[split_Train:]	
	
	tmpX = list(tmpX)
	for x in range(len(tmpX2)):
		tmpX.append(tmpX2[x])
	#trainX = np.array(tmpX)	
	#testX = tmpX1	
	testX = np.array(tmpX)	
	trainX = tmpX1
	
	tmpY = list(tmpY)
	for x in range(len(tmpY2)):
		tmpY.append(tmpY2[x])
	#trainY = np.array(tmpY)	
	#testY = tmpY1		
	testY = np.array(tmpY)	
	trainY = tmpY1		
    

	print ("trainX: " + str(trainX.shape) + " testX: " + str(testX.shape))
	print ("trainY: " + str(trainY.shape) + " testY: " + str(testY.shape))
	return X, Y, trainX, trainY, testX, testY
	'''	
	#split_Train = int(X.shape[0] * 0.7)
	#split_Valid = int(X.shape[0] * 0.3)	
	split_Train = int(X.shape[0] * 0.5)
	split_Valid = int(X.shape[0] * 0.5)	
	#trainX, validX, testX = X[:split_Train], X[split_Train:split_Train+split_Valid], X[split_Train+split_Valid:]
	#trainY, validY, testY = Y[:split_Train], Y[split_Train:split_Train+split_Valid], Y[split_Train+split_Valid:]	
	trainX, testX = X[:split_Train], X[split_Train:]
	trainY, testY = Y[:split_Train], Y[split_Train:]
	print ("trainX: " + str(trainX.shape) + " testX: " + str(testX.shape))
	#print("trainY: " + str(len(trainY)) + " validY: " + str(len(validY)) + " testY: " + str(len(testY)))
	print ("trainY: " + str(trainY.shape) + " testY: " + str(testY.shape))	
	return X, Y, trainX, trainY, testX, testY
	'''
	
def ANN():
	model = Sequential()
	model.add(Dense(35, input_shape = (100,), activation = 'relu'))
	#model.add(Dense(15, activation = 'relu'))
	#model.add(Dense(10, activation = 'relu'))
	model.add(Dense(2 , activation = 'softmax'))
	
	model.compile(loss = "categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])
	model.summary()
	return model
	
def Neural_Network(trainX, trainY, testX, testY):
	model = ANN()
	H = model.fit(trainX, trainY, batch_size = BATCH_SIZE, epochs = EPOCHS, verbose = 1)
	print("[INFO] serializing network")
	model.save("ANN2.h5")
	
	plt.style.use("ggplot")
	plt.figure()
	N = EPOCHS
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	#plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
	#plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy on traffic-sign classifier")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig("ANN2.png")
	
	scores = model.evaluate(testX, testY) 	
	print()  
	print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))  

def load_test_data():
	print("[INFO] loading test data...")
	TestSet = []
	with open(Test_path, "r") as fp:
		all_lines = fp.readlines()
		data = []
		data1 = []
		data2 = []
		SMV = []
		for x in range(len(all_lines)):
			#if (x % 6 == 0):
				data.append(float(all_lines[x].split(",")[0]))
				data1.append(float(all_lines[x].split(",")[1]))
				data2.append(float(all_lines[x].split(",")[2]))
				SMV.append(float(all_lines[x].split(",")[3]))
		tmp = np.array([data, data1, data2, SMV]).T		
		TestSet = np.array(tmp)
	print("TestSet:" + str(TestSet.shape))
	return TestSet
	
def split_100(TestSet):
	print (len(TestSet))
	index = 0
	data = []
	for x in range(len(TestSet)):
		if ((index + 100) > len(TestSet)):
			break
		tmp = TestSet[index:index + 100]
		tmp = np.array(tmp)
		data.append(tmp)
		index = index + windows
	data = np.array(data)
	print(data.shape)		
	return data
	
def Data_Extract(TestSet):	
	magNoG = []
	for x in range(len(TestSet)):		
		data0, data1, data2, data3 = [], [], [], []
		for y in range(len(TestSet[x])):
			data0.append(TestSet[x][y][0])
			data1.append(TestSet[x][y][1])
			data2.append(TestSet[x][y][2])	
			data3.append(TestSet[x][y][3])	
		data0, data1, data2, data3 = np.array(data0), np.array(data1), np.array(data2), np.array(data3)		
		
		Ax = []
		Ax_SUM = []
		Ax_MIN = min(data0)
		Ax_MAX = max(data0)		
		for y in range(len(data0)):
			tmp = data0[y] - ((data0[y] - Ax_MIN) / (Ax_MAX - Ax_MIN))	
			Ax_SUM.append((data0[y] - Ax_MIN) / (Ax_MAX - Ax_MIN))
			Ax.append(tmp)
		Ax = np.array(Ax)
		Ax_SUM = np.array(Ax_SUM)
		
		Ay = []
		Ay_MIN = min(data1)
		Ay_MAX = max(data1)
		
		for y in range(len(data1)):
			tmp = data1[y] - ((data1[y] - Ay_MIN) / (Ay_MAX - Ay_MIN)) 
			Ay.append(tmp)
		Ay = np.array(Ay)
		
		Az = []
		Az_MIN = min(data2)
		Az_MAX = max(data2)
		
		for y in range(len(data2)):
			tmp = data2[y] - ((data2[y] - Az_MIN) / (Az_MAX - Az_MIN))
			Az.append(tmp)
		Az = np.array(Ax)
		
		A_SMV = []
		for y in range(len(Ax)):
			A_SMV.append(np.sqrt(np.power(Ax[y], 2) + np.power(Ay[y], 2) + np.power(Az[y], 2)))
		A_SMV = np.array(A_SMV)				
		A_SMVNorm = []
		for y in range(len(A_SMV)):			
			A_SMVNorm.append(A_SMV[y] - np.mean(Ax_SUM))
		A_SMVNorm = np.array(A_SMVNorm)
		magNoG.append(A_SMVNorm)
		
	magNoG = np.array(magNoG)
	print(magNoG.shape)
	return magNoG
	
def model_predict(test, testY, filenames):				
	count0 = 0
	count1 = 0
	count3 = 0
	count4 = 0
	index = []	
	model = load_model("ANN2.h5")
	predict_test = model.predict_classes(test)	
	for x in range(len(predict_test)):
		'''
		if (predict_test[x] == 0):
			count0 = count0 + 1
		elif(predict_test[x] == 1):
			count1 = count1 + 1
			index.append(x)
		'''
		if (predict_test[x] == 0 and testY[x][0] == 1): #日常動作
			count0 = count0 + 1			
			#index.append(x)
		elif(predict_test[x] == 1 and testY[x][1] == 1): #跌倒動作
			count1 = count1 + 1	
			#index.append(x)			
		elif(predict_test[x] == 0 and testY[x][1] == 1): #跌倒動作誤判為日常
			count3 = count3 + 1
			#index.append(x)
			#print(str(x) + ": " + str(filenames[x]))
		elif(predict_test[x] == 1 and testY[x][0] == 1): #日常動作誤判為跌倒
			count4 = count4 + 1
			#index.append(x)	
			print(str(x) + ": " + str(filenames[x]))
			
	print("\n\n\n\n")
	#print(len(predict_test), count0, count1)
	print(len(predict_test), count0, count1, count3, count4)
	print(index)	
		
def main():
	Origin_trainX, Origin_trainY, filenames = load_data()	
	Origin_trainX = Data_Extract(Origin_trainX)
	#Origin_trainX, Origin_trainY, trainX, trainY, testX, testY = splid_data(Origin_trainX, Origin_trainY)
	#Neural_Network(trainX, trainY, testX, testY)
	#Test
	#TestSet = load_test_data()
	#test = split_100(TestSet)
	#test = Data_Extract(test)
	model_predict(Origin_trainX, Origin_trainY, filenames)

if __name__ == '__main__':
	main()				