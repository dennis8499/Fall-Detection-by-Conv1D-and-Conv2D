#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from os import walk

savePath = "C:/Users/denni/Desktop/論文/實驗數據/新的數據_頻率10HZ/新的數據_頻率10HZ/new/"
saveImagePath = "C:/Users/denni/Desktop/論文/實驗數據/新的數據_頻率10HZ/新的數據_頻率10HZ/Image/"
for root, dirs, files in walk(savePath):	
	for x in range(len(files)):
		filename = str(root) + "/" + str(files[x])
		saveFile = str(files[x]).split(".")[0]		
		saveFileName = saveImagePath + saveFile + ".png"
		print (saveFileName + "\n")
		with open(filename, "r") as fp:
			all_lines = fp.readlines()			
			data = np.zeros((len(all_lines), 3), dtype=np.dtype(float))
			smv = np.zeros(len(all_lines), dtype = np.dtype(float))
			list = np.arange(0, len(all_lines), 1)
			for y in range(len(all_lines)):			
				data[y][0] = float(all_lines[y].split()[0])
				data[y][1] = float(all_lines[y].split()[1])
				data[y][2] = float(all_lines[y].split()[2])
				smv[y] = np.sqrt(np.power(data[y][0], 2) + np.power(data[y][1], 2) + np.power(data[y][2], 2)) 	
				#print (all_lines[y].split()[0] + " : " + all_lines[y].split()[1] + " : " + all_lines[y].split()[2] + "\n" )
			plt.figure()
			plt.plot(list, smv, color='red', linewidth=1.0)
			plt.axis('off')		
			plt.savefig(saveFileName)
			plt.close()
			