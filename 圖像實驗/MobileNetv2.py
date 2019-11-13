#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
from os import walk
import random
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, Dense, MaxPooling1D, Flatten, Input, GlobalAveragePooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, BatchNormalization, add, Reshape
from keras.applications import MobileNetV2
from keras.utils import get_file
from keras.layers import DepthwiseConv2D
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from imutils import paths
import cv2
import os

BATCH_SIZE = 64 #每次跑多少大小
EPOCHS = 200 #跑幾次
CLASS_NUM = 2 #分幾類
height = 173 #345 173(50%) 87(25%) 173
weight = 270 #460 270(50%) 115(25%) 230
depth = 3

Train_path = 'DateSet/train_2(XYZ_3_X255_None_lower)'

def load_data(path):
	print("[INFO] loading images...")
	data = []
	labels = []
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
		label = int(imagePath.split('\\', 2)[1])		
		labels.append(label)
	# scale the raw pixel intensities to the range [0, 1]
	data = np.array(data, dtype = "float32") / 255.0	
	labels = np.array(labels)	
	
	#convert the labels from integers to vectors
	label = to_categorical(labels, num_classes = CLASS_NUM)	
	
	print ("data.shape: " + str(data.shape))
	print ("label.shape: " + str(label.shape))
	return data, labels
	
def splid_data_50(X, Y):
	print("[INFO] splid the dataset...")	
	split_Train = int(X.shape[0] * 0.5)	
	trainX, testX = X[:split_Train], X[split_Train:]
	trainY, testY = Y[:split_Train], Y[split_Train:]	
	'''
	split_Train = int(X.shape[0] * 0.4)
	split_Valid = int(X.shape[0] * 0.9)	
	tmpX, tmpX1, tmpX2 = X[:split_Train], X[split_Train:split_Valid], X[split_Valid:]
	tmpY, tmpY1, tmpY2 = Y[:split_Train], Y[split_Train:split_Valid], Y[split_Valid:]
	
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
	
def split_data(X, Y):	
	print("[INFO] split the dataset...")
	
	split_Train = int(X.shape[0] * 0.7)
	split_Valid = int(X.shape[0] * 0.2)
	
	trainX, validX, testX = X[:split_Train], X[split_Train:split_Train+split_Valid], X[split_Train+split_Valid:]
	trainY, validY, testY = Y[:split_Train], Y[split_Train:split_Train+split_Valid], Y[split_Train+split_Valid:]
	
	print ("trainX: " + str(trainX.shape) + " validX: " + str(validX.shape) + " testX: " + str(testX.shape))
	print ("trainY: " + str(trainY.shape) + " validY: " + str(validY.shape) + " testY: " + str(testY.shape))

	return trainX, trainY, validX, validY, testX, testY
	
def batch_iter(data, labels, batch_size, shuffle=True):
	num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

	def data_generator():
		data_size = len(data)
		while True:
			if shuffle:
				shuffle_indices = np.random.permutation(np.arange(data_size))
				shuffled_data = data[shuffle_indices]
				shuffled_labels = labels[shuffle_indices]
			else:
				shuffled_data = data
				shuffled_labels = labels

			for batch_num in range(num_batches_per_epoch):
				start_index = batch_num * batch_size
				end_index = min((batch_num + 1) * batch_size, data_size)
				X, y = shuffled_data[start_index: end_index], shuffled_labels[start_index: end_index]
				yield X, y
	return num_batches_per_epoch, data_generator()

def Neural_Network_MobileNetV2(trainX, trainY, validX, validY, testX, testY):
	trainY = to_categorical(trainY, CLASS_NUM)
	validY = to_categorical(validY, CLASS_NUM)
	testY = to_categorical(testY, CLASS_NUM)
	
	inputShape = (height, weight, depth)
	if K.image_data_format() == "channels_first":
		inputShape = (depth, height, weight)
	
	model = MobileNetV2(input_shape = inputShape, alpha=1.0, include_top=True, weights=None, pooling='max', classes=CLASS_NUM)
	
	#model.summary()
	
	print("[INFO] compiling model...")
	model.compile(loss = "categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])
	train_steps, train_batches = batch_iter(trainX, trainY, BATCH_SIZE)
	valid_steps, valid_batches = batch_iter(validX, validY, BATCH_SIZE)
	H = model.fit_generator(train_batches, train_steps, epochs=EPOCHS, validation_data=valid_batches, validation_steps=valid_steps)
	
	print("[INFO] serializing network...")
	model.save("MobileNetv2.h5")

	# plot the training loss and accuracy
	plt.style.use("ggplot")
	plt.figure()
	N = EPOCHS
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy on traffic-sign classifier")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig("MobileNetv2.png")

	scores = model.evaluate(testX, testY, batch_size = BATCH_SIZE)  
	print()  
	print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0)) 
	
def main():	
	dtype = 'float32'
	K.set_floatx(dtype)	
	np.set_printoptions(threshold=np.inf)
	
	trainX, trainY = load_data(Train_path)
	trainX_50, trainY_50, trainX_100, trainY_100 = splid_data_50(trainX, trainY) #5 * 2 fold cross-validation
	trainX, trainY, validX, validY, testX, testY  = split_data(trainX_50, trainY_50)
	#trainX, trainY, validX, validY, testX, testY  = split_data(trainX, trainY)
	Neural_Network_MobileNetV2(trainX, trainY, validX, validY, testX, testY)
	
if __name__ == '__main__':
   main() 