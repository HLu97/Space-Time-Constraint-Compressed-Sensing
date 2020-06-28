# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 15:47:09 2019

@author: 朱震东
"""
import  numpy as np    #对应numpy包
from PIL import Image  #对应pillow包
import scipy.misc

#data = np.loadtxt('0.png.txt')

#scipy.misc.imsave('0.png', 255*data)

A = np.zeros([3,4])
B = np.zeros(4)
B[1] = 1
print(A)
print(B)
print(B[2]>0)