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
#from sklearn.linear_model import LogisticRegression
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import confusion_matrix
#from sklearn.externals import joblib
from imutils import paths
from keras.callbacks import TensorBoard
import time
import cv2
import os

Train_path = 'DateSet/兩變數/train_2 (原始)'
Test_path = '2019-05-19-1.txt'
Image_Test_path = 'result/'
windows = 50
NUM_CLASS = 2
height = 173 #345 173(50%) 87(25%) 173
weight = 270 #460 270(50%) 115(25%) 230
Alpha = 0.9
CONV1D_BATCH_SIZE = 32
CONV1D_EPOCHS = 1000

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
	print("DataSet:" + str(dataSet.shape))
	print("Label:" + str(len(label)))	
	print("filenames:" + str(len(filenames)))
	return dataSet, label, filenames
	
def splid_data_50(X, Y):
	print("[INFO] splid the dataset...")	
	split_Train = int(X.shape[0] * 0.5)	
	trainX, testX = X[:split_Train], X[split_Train:]
	trainY, testY = Y[:split_Train], Y[split_Train:]	
	
	'''
	#split_Train = int(X.shape[0] * 0.4)
	#split_Valid = int(X.shape[0] * 0.9)	
	#tmpX, tmpX1, tmpX2 = X[:split_Train], X[split_Train:split_Valid], X[split_Valid:]
	#tmpY, tmpY1, tmpY2 = Y[:split_Train], Y[split_Train:split_Valid], Y[split_Valid:]	
	tmpX = list(tmpX)
	for x in range(len(tmpX2)):
		tmpX.append(tmpX2[x])
	trainX = np.array(tmpX)	
	testX = tmpX1	
	#testX = np.array(tmpX)	
	#trainX = tmpX1
	
	tmpY = list(tmpY)
	for x in range(len(tmpY2)):
		tmpY.append(tmpY2[x])
	trainY = np.array(tmpY)	
	testY = tmpY1		
	#testY = np.array(tmpY)	
	#trainY = tmpY1		
    '''

	print ("trainX: " + str(trainX.shape) + " testX: " + str(testX.shape))
	print ("trainY: " + str(trainY.shape) + " testY: " + str(testY.shape))
	return trainX, trainY, testX, testY

def splid_data_Filenames(X):
	print("[INFO] splid the dataset...")	
	split_Train = int(X.shape[0] * 0.5)	
	trainX, testX = X[:split_Train], X[split_Train:]
	
	'''
	split_Train = int(X.shape[0] * 0.4)
	split_Valid = int(X.shape[0] * 0.9)	
	tmpX, tmpX1, tmpX2 = X[:split_Train], X[split_Train:split_Valid], X[split_Valid:]	
	
	tmpX = list(tmpX)
	for x in range(len(tmpX2)):
		tmpX.append(tmpX2[x])
	#trainX = np.array(tmpX)	
	#testX = tmpX1	
	testX = np.array(tmpX)	
	trainX = tmpX1
	'''
	print ("trainX: " + str(trainX.shape) + " testX: " + str(testX.shape))	
	return trainX, testX
	
def splid_data_Image(X):
	print("[INFO] splid the dataset...")	
	split_Train = int(X.shape[0] * 0.5)			
	trainX, testX = X[:split_Train], X[split_Train:]
	
	'''
	#split_Train = int(X.shape[0] * 0.1)
	#split_Valid = int(X.shape[0] * 0.6)		
	tmpX, tmpX1, tmpX2 = X[:split_Train], X[split_Train:split_Valid], X[split_Valid:]
	tmpX = list(tmpX)
	for x in range(len(tmpX2)):
		tmpX.append(tmpX2[x])
	trainX = np.array(tmpX)	
	testX = tmpX1
	#testX = np.array(tmpX)	
	#trainX = tmpX1
	'''
	print ("trainX: " + str(trainX.shape) + " testX: " + str(testX.shape))
	return trainX, testX	
	
def splid_data(X, Y):
	print("[INFO] splid the dataset...")	
	split_Train = int(X.shape[0] * 0.7)
	split_Valid = int(X.shape[0] * 0.2)	
	trainX, validX, testX = X[:split_Train], X[split_Train:split_Train+split_Valid], X[split_Train+split_Valid:]
	trainY, validY, testY = Y[:split_Train], Y[split_Train:split_Train+split_Valid], Y[split_Train+split_Valid:]	
	print ("trainX: " + str(trainX.shape) + " validX: " + str(validX.shape) + " testX: " + str(testX.shape))
	print ("trainY: " + str(trainY.shape) + " validY: " + str(validY.shape) + " testY: " + str(testY.shape))
	return X, Y, trainX, trainY, validX, validY, testX, testY
	
def NN_Conv1D():  
	model = Sequential((	
		Conv1D(activation = 'relu', input_shape = (100, 4), filters = 18, kernel_size = 2, strides = 1, padding = 'same'),		
		MaxPooling1D(pool_size = 2, strides = 2, padding = 'same'),		
		Conv1D(activation = 'relu', filters = 36, kernel_size = 2, strides = 1, padding = 'same'),
		MaxPooling1D(pool_size = 2, strides = 2, padding = 'same'),
		Conv1D(activation = 'relu', filters = 72, kernel_size = 2, strides = 1, padding = 'same'),
		MaxPooling1D(pool_size = 2, strides = 2, padding = 'same'),
		Conv1D(activation = 'relu', filters = 144, kernel_size = 2, strides = 1, padding = 'same'),
		MaxPooling1D(pool_size = 2, strides = 2, padding = 'same'),
		Flatten(),
		BatchNormalization(),
		Dense(NUM_CLASS, activation='softmax'),
	))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	return model	
	
def Neural_Network(trainX, trainY, validX, validY, testX, testY):
	model = NN_Conv1D()
	#model = load_model("time_classification.h5")
	H = model.fit(trainX, trainY, batch_size = CONV1D_BATCH_SIZE, epochs = CONV1D_EPOCHS, verbose = 1, validation_data=(validX, validY), callbacks=[TensorBoard(log_dir='./tmp/log')])
	print("[INFO] serializing network...")
	model.save("time_classification.h5")
	
	plt.style.use("ggplot")
	plt.figure()
	N = CONV1D_EPOCHS
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy on traffic-sign classifier")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig("time_classification.png")
	
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
			#if (x % 6 == 0): #有些我收集的數據是60Hz 因此要除6
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
	
def model_predict(test, data):				
	count0 = 0
	count1 = 0
	index = []	
	model = load_model("time_classification.h5")
	predict_test = model.predict_classes(test)	
	for x in range(len(predict_test)):	
		if (predict_test[x] == 0):
			count0 = count0 + 1			
			#index.append(x)
		elif(predict_test[x] == 1):
			count1 = count1 + 1
			index.append(x)		
		
	print("\n\n\n\n")
	print(len(predict_test), count0, count1)
	print(index)	
	
	writeFile(index, data)
	makePlot(index, data)	
	
	if (len(index) != 0):
		Image = load_data_Image(Image_Test_path)
		Predict_Conv2D(Image)
		
def model_predict_50(test, data, testY, filenames):				
	count0 = 0
	count1 = 0
	count3 = 0
	count4 = 0
	index = []	
	a = time.time()
	model = load_model("time_classification.h5")
	predict_test = model.predict_classes(test)	
	for x in range(len(predict_test)):	
		
		if (predict_test[x] == 0 and testY[x][0] == 1): #日常動作
			count0 = count0 + 1			
			#index.append(x)
		elif(predict_test[x] == 1 and testY[x][1] == 1): #跌倒動作
			count1 = count1 + 1	
			#index.append(x)
			#print(str(x) + ": " + str(filenames[x]))
		elif(predict_test[x] == 0 and testY[x][1] == 1): #跌倒動作誤判為日常
			count3 = count3 + 1
			#index.append(x)
			#print(str(x) + ": " + str(filenames[x]))
		elif(predict_test[x] == 1 and testY[x][0] == 1): #日常動作誤判為跌倒
			count4 = count4 + 1
			index.append(x)	
			print(str(x) + ": " + str(filenames[x]))
		'''		
		if (testY[x][0] == 1):
			count0 = count0 + 1			
			index.append(x)
		elif(testY[x][1] == 1):
			count1 = count1 + 1
			#index.append(x)	
		'''
	print("\n\n\n\n")
	print(len(predict_test), count0, count1, count3, count4)
	b = time.time()
	print(b - a)
	print(index)		
	writeFile(index, data)
	
	c = time.time()
	makePlot(index, data)		
	if (len(index) != 0):		
		Image = load_data_Image(Image_Test_path)
		Predict_Conv2D_50(Image, index, testY)
	d = time.time()
	print(d - c)
	
		
def load_data_Image(path):
	print("[INFO] loading images...")
	data = []
	# grab the image paths and randomly shuffle them
	imagePaths = sorted(list(paths.list_images(path)))		
	random.seed(42)
	random.shuffle(imagePaths)
	# loop over the input images
	
	for imagePath in imagePaths:
		# load the image, pre_process it, and store it in the data list		
		image = cv2.imread(imagePath)		
		image = cv2.resize(image, (weight, height))		
		image = img_to_array(image)		
		data.append(image)
		
		# extract the class label from the image path and update the
        # labels list
		# print(imagePath.split('\\', 2)[1])
		# ['DateSet/train', '1', 'xxxx.png']
	# scale the raw pixel intensities to the range [0, 1]
	data = np.array(data, dtype = "float32") / 255.0	
	
	#split_Train = int(data.shape[0] * 0.5)
	#trainX, testX = data[:split_Train], data[split_Train:]
	#convert the labels from integers to vectors	
	print ("data.shape: " + str(data.shape))	
	#print ("trainX: " + str(trainX.shape) + " testX: " + str(testX.shape))
	#return trainX, testX
	return data
	
def writeFile(index, data):
	print("[INFO] Writing file...")
	for x in index:
		filename = "result/" + str(x) + ".txt"		
		fp = open(filename, "w")
		for y in range(len(data[x])):			
			msg = str(data[x][y][0]) + " " + str(data[x][y][1]) + " " + str(data[x][y][2]) + " " + str(data[x][y][3])
			fp.write(msg + "\n")
		fp.close()
		
def makePlot(index, test):
	print("[INFO] Makeing plot")	
	for x in index:
		data0, data1, data2, data3 = [], [], [], []
		for y in range(100):
			data0.append(test[x][y][0])
			data1.append(test[x][y][1])
			data2.append(test[x][y][2])	
			data3.append(test[x][y][3])
		data0, data1, data2, data3 = np.array(data0), np.array(data1), np.array(data2), np.array(data3)		
		
		#加最小值
		tmp = []
		Low = abs(min(data0))
		for y in range(len(data0)):
			tmp.append(data0[y] + Low)
		A_X = np.array(tmp)
		
		Equal = 255 / max(A_X)		
		#重塑
		tmp = []		
		for y in range(len(A_X)):		
			math = (abs(int(A_X[y])) % max(A_X)) * Equal
			if(math >= 0 and math <= 255):
				tmp.append(math)
			else:
				print("A_X")
				print(math)
		A_X = np.array(tmp)
		
		#加最小值
		tmp = []
		Low = abs(min(data1))
		for y in range(len(data1)):
			tmp.append(data1[y] + Low)
		A_Y = np.array(tmp)				
		
		Equal = 255 / max(A_Y)		
		#重塑
		tmp = []		
		for y in range(len(A_Y)):
			math = (abs(int(A_Y[y])) % max(A_Y)) * Equal
			if(math >= 0 and math <= 255):
				tmp.append(math)	
			else:
				print("A_Y")
				print(math)
		A_Y = np.array(tmp)	
		
		#加最小值
		tmp = []
		Low = abs(min(data2))
		for y in range(len(data2)):
			tmp.append(data2[y] + Low)
		A_Z = np.array(tmp)
		
		Equal = 255 / max(A_Z)
		#重塑
		tmp = []		
		for y in range(len(A_Z)):
			math = (abs(int(A_Z[y])) % max(A_Z)) * Equal
			if(math >= 0 and math <= 255):
				tmp.append(math)	
			else:
				print("A_Z")
				print(math)
		A_Z = np.array(tmp)
		
		'''
		tmp = []	
		tmp.append(0)
		for y in range(len(data3) - 1):
			math = int(data3[y] - data3[y + 1])
			#math = int(abs(data3[y] - data3[y + 1] % 255))
			tmp.append(math)				
		A_SMV = np.array(tmp)	
		print(A_SMV.shape)
		'''
		
		#print(A_X.shape, A_Y.shape, A_Z.shape)
		A_X = np.reshape(A_X, (10, 10))
		A_Y = np.reshape(A_Y, (10, 10))
		A_Z = np.reshape(A_Z, (10, 10))
		#Zeros = np.zeros((10, 10), dtype = int)
		#tmp = np.array([A_X, A_Y, A_Z]).T		
		
		#A_X = np.array([A_X, Zeros, Zeros]).T
		#A_Y = np.array([Zeros, A_Y, Zeros]).T
		#A_Z = np.array([Zeros, Zeros, A_Z]).T		
		
		#data0 = np.reshape(data0, (10, 10))
		#data1 = np.reshape(data1, (10, 10))
		#data2 = np.reshape(data2, (10, 10))		
		
		#A_SMV = np.reshape(A_SMV, (10, 10))
		
		plt.figure()
		plt.subplot(1,3,1)
		#plt.imshow(A_X, interpolation='gaussian', cmap='Reds', origin='lower')		#高斯模糊
		plt.imshow(A_X, interpolation='nearest', cmap='Reds', origin='lower')       #沒模糊 由下往上排
		#plt.imshow(data0, interpolation='nearest', cmap='Reds', origin='upper')    #沒模糊 由上往下排
		plt.axis('off')	
		#plt.colorbar()
		
		plt.subplot(1,3,2)
		#plt.imshow(A_Y, interpolation='gaussian', cmap='Greens', origin='lower')	#高斯模糊
		plt.imshow(A_Y, interpolation='nearest', cmap='Greens', origin='lower')	    #沒模糊 由下往上排
		#plt.imshow(data1, interpolation='nearest', cmap='Greens', origin='upper')	#沒模糊 由上往下排
		plt.axis('off')	
		#plt.colorbar()
		
		plt.subplot(1,3,3)
		#plt.imshow(A_Z, interpolation='gaussian', cmap='Blues', origin='lower')	#高斯模糊	
		plt.imshow(A_Z, interpolation='nearest', cmap='Blues', origin='lower')		#沒模糊 由下往上排
		#plt.imshow(data2, interpolation='nearest', cmap='Blues', origin='upper')	#沒模糊 由上往下排
		plt.axis('off')	
		#plt.colorbar()
		
		
		plt.gca().xaxis.set_major_locator(plt.NullLocator())
		plt.gca().yaxis.set_major_locator(plt.NullLocator())
		plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
		plt.margins(0,0)
		msg = "result/" + str(x) + ".png"
		plt.savefig(msg, transparent=True, dpi=72, pad_inches = 0)
		plt.close()						
	print("[INFO] Done")
	
def Predict_Conv2D(test):	
	model = load_model("MobileNetv2.h5")
	predict_test = model.predict(test)
	count0 = 0
	count1 = 0
	#print(predict_test)
	for x in range(len(predict_test)):
		if (predict_test[x][0] > predict_test[x][1]):
			print("0")
			count0+=1
		elif(predict_test[x][0] < predict_test[x][1]):
			print("1")	
			count1+=1
	#print(count0, count1)
	
def Predict_Conv2D_50(test, index, testY):		
	#model = load_model("Conv2D.h5")
	model = load_model("MobileNetv2.h5")	
	predict_test = model.predict(test)
	count0 = 0
	count1 = 0
	count3 = 0
	count4 = 0
	#print(predict_test)
	for x in range(len(predict_test)):
		if (predict_test[x][0] > predict_test[x][1] and testY[index[x]][0] == 1): #日常動作
			count0 = count0 + 1	
			#print(index[x])
		elif(predict_test[x][0] < predict_test[x][1] and testY[index[x]][1] == 1): #跌倒動作
			count1 = count1 + 1		
			#print(index[x])
		elif(predict_test[x][0] > predict_test[x][1] and testY[index[x]][1] == 1): #跌倒動作誤判為日常
			count3 = count3 + 1
			#print(index[x])
		elif(predict_test[x][0] < predict_test[x][1] and testY[index[x]][0] == 1): #日常動作誤判為跌倒
			count4 = count4 + 1
			print(index[x])

	print(count0, count1, count3, count4)
	
def Transform_FFT(data):
	trainX_FFT = []
	print("[INFO] FFT...")
	for x in range(len(data)):
		data0, data1, data2, data3 = [], [], [], []
		for y in range(100):	
			data0.append(data[x][y][0])
			data1.append(data[x][y][1])
			data2.append(data[x][y][2])
			data3.append(data[x][y][3])
		data0, data1, data2, data3 = np.array(data0), np.array(data1), np.array(data2), np.array(data3)		
		
		#低通濾波
		tmp = []	
		tmp.append(data0[0])
		for y in range(1, len(data0)):
			tmp.append(Alpha * data0[y] + (1 - Alpha) * tmp[y - 1])			
		A_X = np.array(tmp)
		
		AVG_A_X = np.mean(A_X)
		#與平均值的偏差
		tmp = []		
		for y in range(len(A_X)):
			tmp.append(A_X[y] - AVG_A_X)
		A_X = np.array(tmp)		
		
		#低通濾波
		tmp = []	
		tmp.append(data1[0])
		for y in range(1, len(data1)):
			tmp.append(Alpha * data1[y] + (1 - Alpha) * tmp[y - 1])
		A_Y = np.array(tmp)
		
		AVG_A_Y = np.mean(A_Y)
		#與平均值的偏差
		tmp = []		
		for y in range(len(A_Y)):
			tmp.append(A_Y[y] - AVG_A_Y)
		A_Y = np.array(tmp)
		
		#低通濾波
		tmp = []	
		tmp.append(data2[0])
		for y in range(1, len(data2)):
			tmp.append(Alpha * data2[y] + (1 - Alpha) * tmp[y - 1])
		A_Z = np.array(tmp)
		
		AVG_A_Z = np.mean(A_Z)
		#與平均值的偏差
		tmp = []		
		for y in range(len(A_Z)):
			tmp.append(A_Z[y] - AVG_A_Z)
		A_Z = np.array(tmp)				
		
		#低通濾波
		tmp = []	
		tmp.append(data3[0])
		for y in range(1, len(data3)):
			tmp.append(Alpha * data3[y] + (1 - Alpha) * tmp[y - 1])
		A_SMV = np.array(tmp)
		
		#SMV差值
		tmp = []		
		tmp.append(0)
		for y in range(len(A_SMV) - 1):
			math = int(A_SMV[y] - A_SMV[y + 1])
			tmp.append(math)				
		A_SMV = np.array(tmp)	
		
		yf1 = abs(np.fft.fft(A_X))/len(A_X)
		yf2 = abs(np.fft.fft(A_Y))/len(A_Y)
		yf3 = abs(np.fft.fft(A_Z))/len(A_Z)		
		#yf4 = abs(np.fft.fft(A_SMV))/len(A_SMV)
		
		#yf5 = abs(np.fft.fft(data3))/len(data3)
		
		#yf1 = abs(np.fft.fft(A_X))
		#yf2 = abs(np.fft.fft(A_Y))
		#yf3 = abs(np.fft.fft(A_Z))	
		#yf4 = abs(np.fft.fft(A_SMV))
		#yf5 = abs(np.fft.fft(data3))
		
		#tmp = np.array([yf1, yf2, yf3, A_SMV, data3]).T
		#tmp = np.array([yf1, yf2, yf3, yf4]).T
		#tmp = np.array([yf1, yf2, yf3, yf5]).T
		#tmp = np.array([yf1, yf2, yf3, yf4, yf5]).T
		tmp = np.array([yf1, yf2, yf3, A_SMV]).T
		#tmp = np.array([yf1, yf2, yf3]).T
		trainX_FFT.append(tmp)
		
	trainX_FFT = np.array(trainX_FFT)			
	print("TrainX_FFT SHAPE" + str(trainX_FFT.shape))
	#return trainX_FFT, data #自己收集的數據用
	return trainX_FFT #學姊數據用
	
def main():
	#np.set_printoptions(threshold=np.inf)
	#學姊數據
	a = time.time()
	trainX, trainY, filenames = load_data()	
	trainX_FFT = Transform_FFT(trainX)	
	#Train Conv1D	
	trainX_50, trainY_50, trainX_100, trainY_100 = splid_data_50(trainX, trainY)
	trainX_FFT_50, trainY_FFT_50, trainX_FFT_100, trainY_FFT_100 = splid_data_50(trainX_FFT, trainY)
	filenames_50, filenames_100 = splid_data_Filenames(filenames)
	b = time.time()
	print(b - a)	
	#origin_trainX, origin_trainY, trainX, trainY, validX, validY, testX, testY = splid_data(trainX_FFT_100, trainY_FFT_100)
	#Neural_Network(trainX, trainY, validX, validY, testX, testY)
	#驗證用
	#model_predict(trainX_FFT, trainX)
	model_predict_50(trainX_FFT_50, trainX_50, trainY_FFT_50, filenames_50) 
	
	#自己的數據
	#Test
	#TestSet = load_test_data()
	#test = split_100(TestSet)
	#trainX_FFT, trainX = Transform_FFT(test)
	#model_predict(trainX_FFT, trainX)
	
	
	#單獨測試圖
	#Image
	#Image = load_data_Image(Image_Test_path)		
	#trainX, testX = splid_data_Image(Image)
	#print(Image[0][:][:][2])
	#Predict_Conv2D(trainX)
	
if __name__ == '__main__':
	main()	