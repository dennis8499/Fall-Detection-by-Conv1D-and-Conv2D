#!/usr/bin/python
# -*- coding: utf-8 -*-
from os import walk
import numpy as np

mypath = "新的數據_頻率10HZ/"
savepath = "新的數據_頻率10HZ/new/"
name = ["文祥", "宗華", "昕容", "泓岑"]
type = ["日常", "跌倒"]		


for x in range(len(name)):
	for y in range(len(type)):
		path = mypath + name[x] + "/" + type[y]
		for root, dirs, files in walk(path):
			for z in range(len(files)):
				filename = str(root) + "/" + str(files[z])
				with open (filename, "r") as fp:
					all_lines = fp.readlines()
					data = np.zeros((len(all_lines), 23), dtype=np.dtype(float))
					for a in range(len(all_lines)):	
						#print (all_lines[a].split()[0] + " : " + all_lines[a].split()[1] + " : " + all_lines[a].split()[2] + "\n" )
						try:
							float(all_lines[a].split()[0])
							float(all_lines[a].split()[1])
							float(all_lines[a].split()[2])	
							'''
							float(all_lines[a].split()[3])
							float(all_lines[a].split()[4])
							float(all_lines[a].split()[5])							
							float(all_lines[a].split()[6])
							float(all_lines[a].split()[7])
							float(all_lines[a].split()[8])
							float(all_lines[a].split()[9])
							float(all_lines[a].split()[10])
							float(all_lines[a].split()[11])
							float(all_lines[a].split()[12])
							float(all_lines[a].split()[13])
							float(all_lines[a].split()[14])
							float(all_lines[a].split()[15])
							'''
							float(all_lines[a].split()[16])
							float(all_lines[a].split()[17])
							float(all_lines[a].split()[18])
							'''
							float(all_lines[a].split()[19])
							float(all_lines[a].split()[20])
							float(all_lines[a].split()[21])							
							float(all_lines[a].split()[22])	
							'''
							data[a][0] = float(all_lines[a].split()[0]) #Ax
							data[a][1] = float(all_lines[a].split()[1]) #Ay
							data[a][2] = float(all_lines[a].split()[2]) #Az		
							'''
							data[a][3] = float(all_lines[a].split()[3]) #Angx
							data[a][4] = float(all_lines[a].split()[4]) #Angy
							data[a][5] = float(all_lines[a].split()[5]) #Angz							
							data[a][6] = float(all_lines[a].split()[6]) #Mx
							data[a][7] = float(all_lines[a].split()[7]) #My
							data[a][8] = float(all_lines[a].split()[8]) #Mz
							data[a][9] = float(all_lines[a].split()[9]) #Lx
							data[a][10] = float(all_lines[a].split()[10]) #Ly
							data[a][11] = float(all_lines[a].split()[11]) #Lz
							data[a][12] = float(all_lines[a].split()[12]) #RVx
							data[a][13] = float(all_lines[a].split()[13]) #RVy
							data[a][14] = float(all_lines[a].split()[14]) #RVz
							data[a][15] = float(all_lines[a].split()[15]) #RVcos	
							'''
							data[a][16] = float(all_lines[a].split()[16]) #Gx
							data[a][17] = float(all_lines[a].split()[17]) #Gy
							data[a][18] = float(all_lines[a].split()[18]) #Gz
							'''
							data[a][19] = float(all_lines[a].split()[19]) #gX
							data[a][20] = float(all_lines[a].split()[20]) #gY
							data[a][21] = float(all_lines[a].split()[21]) #gZ							
							data[a][22] = float(all_lines[a].split()[22]) #deltaA	
							'''
						except ValueError:
							pass						
					writeFileName = savepath + str(files[z])
					print (writeFileName)
					with open(writeFileName, "w") as wp:
						for b in range(len(all_lines)):
							if (data[b][0] != 0.0 and data[b][1] != 0.0 and data[b][2] != 0.0):
								#msg = str(data[b][0]) + " " + str(data[b][1]) + " " + str(data[b][2]) + " " + str(data[b][3]) + " " + str(data[b][4]) + " " + str(data[b][5]) + " " + str(data[b][6]) + " " + str(data[b][7]) + " " + str(data[b][8]) + " " + str(data[b][9]) + " " + str(data[b][10]) + " " + str(data[b][11]) + " " + str(data[b][12]) + " " + str(data[b][13]) + " " + str(data[b][14]) + " " + str(data[b][15]) + " " + str(data[b][16]) + " " + str(data[b][17]) + " " + str(data[b][18]) + " " + str(data[b][19]) + " " + str(data[b][20]) + " " + str(data[b][21]) + " " + str(data[b][22])
								msg = str(data[b][0]) + " " + str(data[b][1]) + " " + str(data[b][2]) + " " + str(data[b][16]) + " " + str(data[b][17]) + " " + str(data[b][18])
								#print (msg + "\n")
								wp.write(msg)
								wp.write("\n")					
print ("Done")