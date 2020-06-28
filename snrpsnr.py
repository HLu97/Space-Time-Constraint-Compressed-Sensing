# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 10:33:00 2018

@author: 朱震东
"""



# 导入所需的第三方库文件
   #对应numpy包
from PIL import Image  #对应pillow包
import scipy.misc
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random
import  numpy as np
import math
im = np.fromfile("out.bin",dtype=np.float)
# re = np.fromfile("result.bin",dtype=np.float)
re = np.fromfile("sample.bin",dtype=np.float)
d = np.linalg.norm(im)
s = np.linalg.norm(im-re)
print(10*math.log(pow(d,2)/pow(s,2),10))
print(10*math.log(pow(np.max(d),2)/s,10))
