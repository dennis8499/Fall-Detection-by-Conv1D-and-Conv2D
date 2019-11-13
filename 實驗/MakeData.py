#!/usr/bin/python
# -*- coding: utf-8 -*-
from os import walk

from keras.models import Model
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Input, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import load_model
from keras import backend as K
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import seaborn 
import random

#Train_path = 'C:/Users/denni/Desktop/實驗/DateSet/兩變數/train_2 (測試用)'
Train_path = 'C:/Users/denni/Desktop/實驗/DateSet/兩變數/train_2 (原始)'
Test_path = 'C:/Users/denni/Desktop/實驗/2019-01-04-1.txt'
windows = 50
NUM_CLASS = 2
CONV1D_BATCH_SIZE = 256
CONV1D_EPOCHS = 1000
Alpha = 0.9

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
	#random.seed(42)
	#random.shuffle(filenames)	
	for x in range(len(filenames)):
		#print(filenames[x].split("\\")[1].split("/")[1])
		#label.append(filenames[x].split("\\")[1].split("/")[0])
		label.append(filenames[x].split("\\")[1].split("/")[1])
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
					#data1.append(sum(data1)/len(data1))
					#data2.append(sum(data2)/len(data2))
					#data3.append(sum(data3)/len(data3))	
					data1.append(sum(data1[len(data1) - 10 : len(data1)])/ 10)
					data2.append(sum(data2[len(data2) - 10 : len(data2)])/ 10)
					data3.append(sum(data3[len(data3) - 10 : len(data3)])/ 10)
				flag = 0
		if(len(data1) != 0):
			A_SMV = []
			for x in range(len(data1)):
				A_SMV.append(np.sqrt(np.power(data1[x], 2) + np.power(data2[x], 2) + np.power(data3[x], 2)))
			tmp = np.array([data1, data2, data3, A_SMV]).T
			#tmp = np.array(A_SMV)
			dataSet.append(tmp)	
	dataSet = np.array(dataSet)			
	
	#label = to_categorical(label)	
	print("DataSet:" + str(dataSet.shape))
	print("Label:" + str(len(label)))	
	return dataSet, label
	
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
	
def writeFile(data):
	print("[INFO] Writing file...")
	for x in range(len(data)):
		filename = "result/" + str(x) + ".txt"		
		fp = open(filename, "w")
		for y in range(len(data[x])):			
			msg = str(data[x][y][0]) + " " + str(data[x][y][1]) + " " + str(data[x][y][2]) + " " + str(data[x][y][3])
			fp.write(msg + "\n")
		fp.close()
		
def makePlot(test):
	print("[INFO] Makeing plot")
	#list = np.arange(0, 100, 1)	
	for x in range(len(test)):		
		data0, data1, data2, data3 = [], [], [], []
		for y in range(100):
			data0.append(test[x][y][0])
			data1.append(test[x][y][1])
			data2.append(test[x][y][2])	
			data3.append(test[x][y][3])	
		data0, data1, data2, data3 = np.array(data0), np.array(data1), np.array(data2), np.array(data3)		
		
		
		tmp = []	
		tmp.append(data0[0])
		for y in range(1, len(data0)):
			tmp.append(Alpha * data0[y] + (1 - Alpha) * tmp[y - 1])
			#tmp.append(Alpha * tmp[y - 1] + (1 - Alpha) * data0[y])
		A_X = np.array(tmp)		
		
		tmp = []
		for y in range(len(A_X)):
			tmp.append(data0[y] - A_X[y])
		A_X = np.array(tmp)
		
		tmp = []
		Low = abs(min(A_X))
		for y in range(len(data0)):
			tmp.append(A_X[y] + Low)
		A_X = np.array(tmp)

		Equal = 255 / max(A_X)
		tmp = []
		for y in range(len(A_X)):		
			#math = int(abs(A_X[y] % 100))
			math = (abs(int(A_X[y])) % max(A_X)) * Equal
			if(math >= 0 and math <= 255):
				tmp.append(math)
			else:
				print("A_X")
				print(math)
		A_X = np.array(tmp)
		
		tmp = []	
		tmp.append(data1[0])
		for y in range(1, len(data1)):
			tmp.append(Alpha * data1[y] + (1 - Alpha) * tmp[y - 1])
			#tmp.append(Alpha * tmp[y - 1] + (1 - Alpha) * data1[y])
		A_Y = np.array(tmp)
		
		tmp = []
		for y in range(len(A_Y)):
			tmp.append(data1[y] - A_Y[y])
		A_Y = np.array(tmp)
		
		tmp = []
		Low = abs(min(A_Y))
		for y in range(len(data1)):
			tmp.append(A_Y[y] + Low)
		A_Y = np.array(tmp)
		

		Equal = 255 / max(A_Y)
		tmp = []		
		for y in range(len(A_Y)):
			#math = int(abs(A_Y[y] % 100))
			math = (abs(int(A_Y[y])) % max(A_Y)) * Equal
			if(math >= 0 and math <= 255):
				tmp.append(math)	
			else:
				print("A_Y")
				print(math)
		A_Y = np.array(tmp)
		
		tmp = []	
		tmp.append(data2[0])
		for y in range(1, len(data2)):
			tmp.append(Alpha * data2[y] + (1 - Alpha) * tmp[y - 1])
			#tmp.append(Alpha * tmp[y - 1] + (1 - Alpha) * data2[y])
		A_Z = np.array(tmp)
		
		tmp = []
		for y in range(len(A_Z)):
			tmp.append(data2[y] - A_Z[y])
		A_Z = np.array(tmp)
		
		tmp = []
		Low = abs(min(A_Z))
		for y in range(len(data2)):
			tmp.append(A_Z[y] + Low)
		A_Z = np.array(tmp)
		
		
		Equal = 255 / max(A_Z)
		tmp = []		
		for y in range(len(A_Z)):
			#math = int(abs(A_Z[y] % 100))
			math = (abs(int(A_Z[y])) % max(A_Z)) * Equal
			if(math >= 0 and math <= 255):
				tmp.append(math)	
			else:
				print("A_Z")
				print(math)
		A_Z = np.array(tmp)	
		'''
		tmp = []	
		tmp.append(data3[0])
		for y in range(1, len(data3)):
			tmp.append(Alpha * data3[y] + (1 - Alpha) * tmp[y - 1])
		A_SMV = np.array(tmp)		
		
		tmp = []	
		tmp.append(0)
		for y in range(len(A_SMV) - 1):
			math = int(data3[y] - data3[y + 1])			
			tmp.append(math)	
		
		A_SMV = np.array(tmp)	
		
		tmp = []
		Low = abs(min(A_SMV))
		for y in range(len(A_SMV)):
			tmp.append(A_SMV[y] + Low)
		A_SMV = np.array(tmp)
		#print(A_SMV.shape)
		
		Equal = 255 / max(A_SMV)
		tmp = []		
		for y in range(len(A_SMV)):
			#math = int(abs(A_Z[y] % 100))
			math = (abs(int(A_SMV[y])) % max(A_SMV)) * Equal
			if(math >= 0 and math <= 255):
				tmp.append(math)	
			else:
				print("A_SMV")
				print(math)
		A_SMV = np.array(tmp)	
		
		'''
		#print(A_X.shape, A_Y.shape, A_Z.shape, A_SMV.shape)
		A_X = np.reshape(A_X, (10, 10))
		A_Y = np.reshape(A_Y, (10, 10))
		A_Z = np.reshape(A_Z, (10, 10))
		#A_SMV = np.reshape(A_SMV, (10, 10))
		#Zeros = np.zeros((10, 10), dtype = int)
		#tmp = np.array([A_X, A_Y, A_Z]).T
		#A_X = np.array([A_X, Zeros, Zeros]).T
		#A_Y = np.array([Zeros, A_Y, Zeros]).T
		#A_Z = np.array([Zeros, Zeros, A_Z]).T
		
		
		#data0 = np.reshape(data0, (10, 10))
		#data1 = np.reshape(data1, (10, 10))
		#data2 = np.reshape(data2, (10, 10))
		
		plt.figure()
		
		#plt.subplot(2,1,1)
		#plt.imshow(tmp, interpolation='sinc', cmap='viridis', origin='lower')
		#plt.axis('off')	
		#plt.subplot(2,1,2)				
		#plt.imshow(A_SMV, interpolation='sinc', cmap='viridis', origin='lower')		
		#plt.axis('off')	
		
		
		#plt.imshow(tmp, interpolation='sinc', cmap='viridis', origin='lower')		
		#plt.imshow(tmp, interpolation='gaussian', cmap='viridis', origin='lower')		
		#plt.axis('off')	
	
		#index = np.linspace(0, 255, 255)
		#index = np.reshape(index, (10, 10))
		#plt.subplot(2,2,1)
		plt.subplot(1,3,1)				
		#plt.imshow(A_X, interpolation='gaussian', cmap='Reds', origin='lower')		
		plt.imshow(A_X, interpolation='nearest', cmap='Reds', origin='lower')	
		#plt.imshow(A_X, interpolation='gaussian', cmap='Reds', origin='upper')		
		#plt.imshow(A_X, interpolation='nearest', cmap='Reds', origin='upper')	
		plt.axis('off')	
		#plt.colorbar()
		#plt.subplot(2,2,2)
		plt.subplot(1,3,2)		
		#plt.imshow(A_Y, interpolation='gaussian', cmap='Greens', origin='lower')	
		plt.imshow(A_Y, interpolation='nearest', cmap='Greens', origin='lower')	
		#plt.imshow(A_Y, interpolation='gaussian', cmap='Greens', origin='upper')	
		#plt.imshow(A_Y, interpolation='nearest', cmap='Greens', origin='upper')				
		plt.axis('off')	
		#plt.colorbar()
		#plt.subplot(2,2,3)
		plt.subplot(1,3,3)		
		#plt.imshow(A_Z, interpolation='gaussian', cmap='Blues', origin='lower')		
		plt.imshow(A_Z, interpolation='nearest', cmap='Blues', origin='lower')
		#plt.imshow(data2, interpolation='gaussian', cmap='Blues', origin='upper')		
		#plt.imshow(A_Z, interpolation='nearest', cmap='Blues', origin='upper')
		plt.axis('off')	
		#plt.colorbar()
		#plt.subplot(1,4,4)
		#plt.imshow(A_SMV, interpolation='gaussian', cmap='Oranges', origin='lower')		
		#plt.imshow(A_SMV, interpolation='nearest', cmap='Oranges', origin='lower')
		#plt.imshow(A_SMV, interpolation='gaussian', cmap='Greys', origin='lower')		
		#plt.imshow(A_SMV, interpolation='nearest', cmap='Greys', origin='lower')
		#plt.axis('off')	
		#plt.colorbar()
	
		
		#index = np.linspace(0, 100, 100)
		#plt.figure()
		#plt.style.use('bmh') #顯示表格現
		#plt.plot(index, A_X, color='red', label = 'Ax')
		#plt.plot(index, A_Y, color='green', label = 'Ay')
		#plt.plot(index, A_Z, color='blue', label = 'Az')
		#plt.plot(index, A_SMV, color='black', label = 'A_SMV')
		#plt.plot(index, data0, color='red', label = 'Ax')
		#plt.plot(index, data1, color='green', label = 'Ay')
		#plt.plot(index, data2, color='blue', label = 'Az')
		#plt.plot(index, data3, color='red', label = 'SMV')
		#plt.legend(loc='upper right')
		
		plt.gca().xaxis.set_major_locator(plt.NullLocator())
		plt.gca().yaxis.set_major_locator(plt.NullLocator())
		plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
		plt.margins(0,0)
		
		msg = "result/" + str(x) + ".png"
		plt.savefig(msg, transparent=True, dpi=72, pad_inches = 0)
		#plt.savefig(msg)
		plt.close()		
		
	print("[INFO] Done")
	
def FFT(trainX, trainY):
	trainX_FFT = []
	print("[INFO] FFT...")
	for x in range(len(trainX)):
		data0, data1, data2, data3 = [], [], [], []
		for y in range(100):	
			data0.append(trainX[x][y][0])
			data1.append(trainX[x][y][1])
			data2.append(trainX[x][y][2])
			data3.append(trainX[x][y][3])
		data0, data1, data2, data3 = np.array(data0), np.array(data1), np.array(data2), np.array(data3)		
		
		#AVG_A_X = np.mean(data0)
		#AVG_A_Y = np.mean(data1)
		#AVG_A_Z = np.mean(data2)
		#AVG_A_SMV = np.mean(data3)		
		
		tmp = []	
		tmp.append(data0[0])
		for y in range(1, len(data0)):
			tmp.append(Alpha * data0[y] + (1 - Alpha) * tmp[y - 1])
		A_X = np.array(tmp)
		
		AVG_A_X = np.mean(A_X)
		
		tmp = []		
		for y in range(len(A_X)):
			tmp.append(A_X[y] - AVG_A_X)
		A_X = np.array(tmp)		
		
		
		tmp = []	
		tmp.append(data1[0])
		for y in range(1, len(data1)):
			tmp.append(Alpha * data1[y] + (1 - Alpha) * tmp[y - 1])
		A_Y = np.array(tmp)
		
		AVG_A_Y = np.mean(A_Y)
		
		tmp = []		
		for y in range(len(A_Y)):
			tmp.append(A_Y[y] - AVG_A_Y)
		A_Y = np.array(tmp)
		
		tmp = []	
		tmp.append(data2[0])
		for y in range(1, len(data2)):
			tmp.append(Alpha * data2[y] + (1 - Alpha) * tmp[y - 1])
		A_Z = np.array(tmp)
		
		AVG_A_Z = np.mean(A_Z)
		
		tmp = []		
		for y in range(len(A_Z)):
			tmp.append(A_Z[y] - AVG_A_Z)
		A_Z = np.array(tmp)				
		
		
		tmp = []	
		tmp.append(data3[0])
		for y in range(1, len(data3)):
			tmp.append(Alpha * data3[y] + (1 - Alpha) * tmp[y - 1])
		A_SMV = np.array(tmp)
		
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
		#yf5 = abs(np.fft.fft(A_SMV1))/len(A_SMV1)
		
		#yf1 = abs(np.fft.fft(A_X))
		#yf2 = abs(np.fft.fft(A_Y))
		#yf3 = abs(np.fft.fft(A_Z))
		#yf4 = abs(np.fft.fft(A_SMV))
		#sinc1 = abs(np.sinc(data3))
		#sinc2 = abs(np.sinc(A_SMV))
		tmp = np.array([yf1, yf2, yf3, A_SMV]).T
		#tmp = np.array([yf1, yf2, yf3]).T
		#tmp = np.array([data3, yf4, sinc1, sinc2])
		trainX_FFT.append(tmp)
		'''
		Fs = 10.0;
		n = len(A_X)     # length of the signal
		k = np.arange(n)
		T = n/Fs
		frq = k/T
		
		fig, ax = plt.subplots(4, 1)		
		ax[0].plot(frq, data0)
		ax[0].set_xlabel('A_X')
		ax[0].set_ylabel('Count')
		
		ax[1].plot(infrqdex, data1,'r') # plotting the spectrum
		ax[1].set_xlabel('A_Y')
		ax[1].set_ylabel('Count')
		
		ax[2].plot(infrqdex, data2,'G') # plotting the spectrum
		ax[2].set_xlabel('A_Z')
		ax[2].set_ylabel('Count')
			
		ax[3].plot(indefrqx, data3,'B') # plotting the spectrum
		ax[3].set_xlabel('A_SMV')
		ax[3].set_ylabel('Count')
		
		msg = "result/" + str(trainY[x]) + ".png"
		plt.savefig(msg)
		plt.close()
		'''
		'''		
		Fs = 10.0;
		n = len(A_X)     # length of the signal
		k = np.arange(n)
		T = n/Fs
		frq = k/T
		'''
		
		#index = np.linspace(-5, 5, 100)
		index = np.linspace(0, 10, 100)
		plt.figure()
		plt.style.use('bmh') #顯示表格現		
		#plt.plot(index, A_X, color='red', label = 'Ax')
		#plt.plot(index, A_Y, color='green', label = 'Ay')
		#plt.plot(index, A_Z, color='blue', label = 'Az')
		plt.plot(index, yf1, color='red', label = 'Ax')
		plt.plot(index, yf2, color='green', label = 'Ay')
		plt.plot(index, yf3, color='blue', label = 'Az')
		#plt.plot(index, data3, color='red', label = 'SMV')
		plt.legend(loc='upper right')
		
		'''
		fig, ax = plt.subplots(4, 1)			
		
		plt.subplots_adjust(wspace = 0, hspace = 0.5)
		
		index1 = np.linspace(0, 10, 100)
		index = np.linspace(0, 100, 100)
		ax[0].plot(index1, yf1, 'Red')
		#ax[0].set_xlabel('Ax_FFT')
		#ax[0].set_ylabel('Hz')
		
		ax[1].plot(index1, yf2, 'Green') # plotting the spectrum
		#ax[1].set_xlabel('Ay_FFT')
		#ax[1].set_ylabel('Hz')
		
		ax[2].plot(index1, yf3, 'Blue') # plotting the spectrum
		#ax[2].set_xlabel('Az_FFT')
		#ax[2].set_ylabel('Hz')
			
		ax[3].plot(index, A_SMV,'Black') # plotting the spectrum
		#ax[3].set_xlabel('SMV_Diff')
		#ax[3].set_ylabel('Sample_No')
		'''
		#msg = "result/" + str(trainY[x]) + ".png"
		msg = "result/" + str(x) + ".png"
		plt.savefig(msg)
		plt.close()
		
	trainX_FFT = np.array(trainX_FFT)
	print("TrainX_FFT SHAPE" + str(trainX_FFT.shape))	
	return trainX_FFT
	print("[INFO] Done")	

	
	
def main():
	#trainX, trainY = load_data()
	#trainX_FFT = FFT(trainX, trainY)	
	TestSet = load_test_data()
	test = split_100(TestSet)	
	trainX_FFT = FFT(test, test)
	#writeFile(test)
	#makePlot(trainX)
	
if __name__ == '__main__':
	main()			