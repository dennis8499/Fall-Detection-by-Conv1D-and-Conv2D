#!/usr/bin/python
# -*- coding: utf-8 -*-
import glob
import os
 
allfiles = glob.glob('*.png')
for afile in allfiles:
  os.rename(afile, 't_'+ afile)
 
allfiles = glob.glob('*.png')
count=1
for afile in allfiles:
  new_filename = str(count) + '.png'
  print (new_filename)
  os.rename(afile, new_filename)
  count += 1
print("Done")