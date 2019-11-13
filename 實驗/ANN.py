#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
from os import walk
import random
import matplotlib.pyplot as plt
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Input
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import load_model
from imutils import paths
import time
import cv2
import os

#Train_path = 'C:/Users/denni/Desktop/實驗/DateSet/兩變數/train_2 (測試用)'
Train_path = 'C:/Users/denni/Desktop/實驗/DateSet/兩變數/train_2 (原始)'
Test_path = 'C:/Users/denni/Desktop/實驗/2019-05-30-3.txt'
Image_Test_path = 'result/'
NUM_CLASS = 2
windows = 64
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
			data1, data2, data3, data4, data5, data6 = [], [], [], [], [], []
			for y in range(128):				
				if(len(all_lines) == y):
					flag = 1
					break
				try:
					data1.append(float(all_lines[y].split()[0]))
					data2.append(float(all_lines[y].split()[1]))
					data3.append(float(all_lines[y].split()[2]))
					data4.append(float(all_lines[y].split()[3]))
					data5.append(float(all_lines[y].split()[4]))
					data6.append(float(all_lines[y].split()[5]))
				except ValueError:
					pass
			if(flag == 1):
				less = 128 - len(data1)
				for x in range(less):
					data1.append(sum(data1[len(data1) - 10 : len(data1)])/ 10)
					data2.append(sum(data2[len(data2) - 10 : len(data2)])/ 10)
					data3.append(sum(data3[len(data3) - 10 : len(data3)])/ 10)
					data4.append(sum(data4[len(data4) - 10 : len(data4)])/ 10)
					data5.append(sum(data5[len(data5) - 10 : len(data5)])/ 10)
					data6.append(sum(data6[len(data6) - 10 : len(data6)])/ 10)
				flag = 0
		if(len(data1) != 0):
			A_SMV, G_SMV = [], []
			for x in range(len(data1)):
				A_SMV.append(np.sqrt(np.power(data1[x], 2) + np.power(data2[x], 2) + np.power(data3[x], 2)))
				G_SMV.append(np.sqrt(np.power(data4[x], 2) + np.power(data5[x], 2) + np.power(data6[x], 2)))
			tmp = A_SMV + G_SMV
			dataSet.append(tmp)
	dataSet = np.array(dataSet)
	label = to_categorical(label)
	filenames = np.array(filenames)
	print("DataSet: " + str(dataSet.shape))
	print("Label: " + str(len(label)))	
	print("filenames:" + str(len(filenames)))
	return dataSet, label, filenames

def splid_data(X, Y):
	print("[INFO] splid the dataset...")	
	split_Train = int(X.shape[0] * 0.5)
	#split_Train = int(X.shape[0] * 0.1)
	#split_Valid = int(X.shape[0] * 0.6)	
	#tmpX, tmpX1, tmpX2 = X[:split_Train], X[split_Train:split_Valid], X[split_Valid:]
	#tmpY, tmpY1, tmpY2 = Y[:split_Train], Y[split_Train:split_Valid], Y[split_Valid:]
	trainX, testX = X[:split_Train], X[split_Train:]
	trainY, testY = Y[:split_Train], Y[split_Train:]	
	'''
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
    '''

	print ("trainX: " + str(trainX.shape) + " testX: " + str(testX.shape))
	print ("trainY: " + str(trainY.shape) + " testY: " + str(testY.shape))
	return X, Y, trainX, trainY, testX, testY
	'''
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
	
def makeplot(dataSet, label):
	list = np.arange(0, 256, 1)
	for x in range(len(label)):
		if(label[x][0] == 1):
			msg = Image_Test_path + "0_" + str(x) + ".png"
		elif(label[x][1] == 1):
			msg = Image_Test_path + "1_" + str(x) + ".png"
		plt.figure()
		plt.plot(list, dataSet[x], color='red', linewidth = 1.0)
		plt.axis("off")
		plt.savefig(msg)
		plt.close()

def ANN():
	model = Sequential()
	model.add(Dense(20, input_shape = (256,), activation = 'relu'))
	model.add(Dense(15, activation = 'relu'))
	model.add(Dense(10, activation = 'relu'))
	model.add(Dense(2 , activation = 'softmax'))
	
	model.compile(loss = "categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])
	model.summary()
	return model
	
def Neural_Network(trainX, trainY, testX, testY):
	model = ANN()
	H = model.fit(trainX, trainY, batch_size = BATCH_SIZE, epochs = EPOCHS, verbose = 1)
	print("[INFO] serializing network")
	model.save("ANN.h5")
	
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
	plt.savefig("ANN.png")
	
	scores = model.evaluate(testX, testY) 	
	print()  
	print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))  
		
def Predict(Origin_trainX, Origin_trainY, filenames):
	model = load_model("ANN.h5")
	data = []
	label = []
	count0 = 0
	count1 = 0
	count3 = 0
	count4 = 0
	Predict1 = []
	for x in range(len(Origin_trainX)):
		A_SMV, G_SMV = Origin_trainX[x][:128], Origin_trainX[x][128:]
		#print(max(A_SMV), max(G_SMV))
		#if (max(A_SMV) > 8.5 and max(G_SMV) > 8.5):	
		if (max(A_SMV) > 3):
			data.append(Origin_trainX[x])
			label.append(x)
	data = np.array(data)
	print(data.shape)
	predict = model.predict_classes(data)	
	for x in range(len(predict)):
		#print("Label: " + str(Origin_trainY[label[x]]) + " Predict: " + str(predict[x]) + "\n")
		'''
		if(Origin_trainY[label[x]][0] == predict[x]):
			count0+=1
		elif(Origin_trainY[label[x]][1] == predict[x]):
			count1+=1
			Predict1.append(x)
		'''		
		if (predict[x] == 0 and Origin_trainY[label[x]][0] == 1): #日常動作
			count0 = count0 + 1			
			#index.append(x)
		elif(predict[x] == 1 and Origin_trainY[label[x]][1] == 1): #跌倒動作
			count1 = count1 + 1	
			#index.append(x)			
		elif(predict[x] == 0 and Origin_trainY[label[x]][1] == 1): #跌倒動作誤判為日常
			count3 = count3 + 1
			#index.append(x)
			#print(str(x) + ": " + str(filenames[x]))
		elif(predict[x] == 1 and Origin_trainY[label[x]][0] == 1): #日常動作誤判為跌倒
			count4 = count4 + 1
			#index.append(x)	
			print(str(x) + ": " + str(filenames[x]))
		
	print(len(Origin_trainX))
	#print("\n共" + str(len(data)) + "通過閾值, " + "Label 0: " + str(count0) + " Label 1: " + str(count1))
	print(len(predict), count0, count1, count3, count4)
	#print(str(label) + ",      " + str(Predict1))

def makePlot(index, test):
	list = np.arange(0, 256, 1)
	plt.figure()
	plt.plot(list, test, color='red', linewidth=1.0)
	plt.axis('off')		
	msg = "result/" + str(index) + ".png"
	plt.savefig(msg)
	plt.close()

def Predict_ADLs():
	model = load_model("ANN.h5")
	print("[INFO] loading data...")
	TestSet = []
	WriteSet = []
	Index = []
	MAX_Index = []
	A_SMV_MAX = []
	G_SMV_MAX = []
	count = 0
	count_layer1 = 0
	count_layer2_0 = 0
	count_layer2_1 = 0
	with open(Test_path, "r") as fp:
		index = 0
		all_lines = fp.readlines()
		A_SMV, G_SMV = [], []
		data1, data2, data3, data4, data5, data6 = [], [], [], [], [], []
		for x in range(len(all_lines)):	
			#if (x % 6 == 0):
				A_SMV.append(float(all_lines[x].split(",")[3])) #Asmv		
				G_SMV.append(float(all_lines[x].split(",")[7])) #Gsmv	
				data1.append(float(all_lines[x].split(",")[0]))
				data2.append(float(all_lines[x].split(",")[1]))
				data3.append(float(all_lines[x].split(",")[2]))
				data4.append(float(all_lines[x].split(",")[4]))
				data5.append(float(all_lines[x].split(",")[5]))
				data6.append(float(all_lines[x].split(",")[6]))
		for x in range(len(all_lines)):			
			#if (index + 128 > len(all_lines) / 6):		
			if (index + 128 > len(all_lines)):			
				break
			tmp = A_SMV[index : index + 128] + G_SMV[index : index + 128]
			WriteTmp = np.array([data1, data2, data3, data4, data5, data6]).T
			A_SMV_MAX.append(max(tmp[:128]))
			G_SMV_MAX.append(max(tmp[128:]))
			TestSet.append(tmp)
			WriteSet.append(WriteTmp)
			#print(max(A_SMV_MAX), max(G_SMV_MAX))
			count+=1
			index = index + windows
	TestSet = np.array(TestSet)
	print(len(TestSet))
	A_SMV_MAX = np.array(A_SMV_MAX)
	G_SMV_MAX = np.array(G_SMV_MAX)
	for x in range(len(TestSet)):		
		if(A_SMV_MAX[x] > 30 and G_SMV_MAX[x] > 30):
			count_layer1+=1
			MAX_Index.append(x)
	tStart = time.time()
	predict = model.predict_classes(TestSet)
	tEnd = time.time()
	print ("It cost %f sec" % (tEnd - tStart))
	for x in range(len(predict)):
		if(predict[x] == 0):
			count_layer2_0+=1
		elif(predict[x] == 1):
			count_layer2_1+=1
			makePlot(x, TestSet[x])		
			Index.append(x)
	print("Index: " + str(Index) + ", MAX_Index: " + str(MAX_Index))
	print(" ")
	#print(Index, MAX_Index)
	print("Count: " + str(count) + ", count_layer1: " + str(count_layer1) + ", count_layer2_0: " + str(count_layer2_0) + ", count_layer2_1: " + str(count_layer2_1))
	#print(count, count_layer1, count_layer2_0, count_layer2_1)	
	writeFile(Index, WriteSet)
	
def writeFile(index, data):
	print("[INFO] Writing file...")
	for x in index:
		filename = "result/" + str(x) + ".txt"		
		fp = open(filename, "w")
		for y in range(len(data[x])):			
			msg = str(data[x][y][0]) + " " + str(data[x][y][1]) + " " + str(data[x][y][2]) + " " + str(data[x][y][3]) + " " + str(data[x][y][4]) + " " + str(data[x][y][5])
			fp.write(msg + "\n")
		fp.close()
		
	
def main():
	Origin_trainX, Origin_trainY, filenames = load_data()	
	#makeplot(Origin_trainX, Origin_trainY)
	#Origin_trainX, Origin_trainY, trainX, trainY, testX, testY = splid_data(Origin_trainX, Origin_trainY)
	#Neural_Network(trainX, trainY, testX, testY)
	Predict(Origin_trainX, Origin_trainY, filenames)
	#Predict_ADLs()
	
	

if __name__ == '__main__':
	main()				