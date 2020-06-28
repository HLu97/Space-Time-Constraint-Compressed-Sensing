# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 09:33:10 2018

@author: 朱震东
"""

#导入集成库
import math

# 导入所需的第三方库文件
import  numpy as np    #对应numpy包
from PIL import Image  #对应pillow包
import scipy.misc
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random

class CS(object):
    #CoSaMP算法函数
    def cs_CoSaMP(self,y,D):     
        S=math.floor(y.shape[0]/4)  #稀疏度    
        residual=y  #初始化残差
        pos_last=np.array([],dtype=np.int64)
        result=np.zeros((r))
    
        for j in range(S):  #迭代次数
            product=np.fabs(np.dot(D.T,residual))       
            pos_temp=np.argsort(product)
            pos_temp=pos_temp[::-1]#反向，得到前面L个大的位置
            pos_temp=pos_temp[0:2*S]#对应步骤3
            pos=np.union1d(pos_temp,pos_last)   
    
            result_temp=np.zeros((r))
            result_temp[pos]=np.dot(np.linalg.pinv(D[:,pos]),y)
    
            pos_temp=np.argsort(np.fabs(result_temp))
            pos_temp=pos_temp[::-1]#反向，得到前面L个大的位置
            result[pos_temp[:S]]=result_temp[pos_temp[:S]]
            pos_last=pos_temp
            residual=y-np.dot(D,result)
    
        return  result
    
    #IHT算法函数
    def cs_IHT(self,y,D):    
        K=math.floor(y.shape[0]/3)  #稀疏度    
        result_temp=np.zeros((r))  #初始化重建信号   
        u=0.5  #影响因子
        result=result_temp
        for j in range(K):  #迭代次数
            x_increase=np.dot(D.T,(y-np.dot(D,result_temp)))    #x=D*(y-D*y0)
            result=result_temp+np.dot(x_increase,u) #   x(t+1)=x(t)+D*(y-D*y0)
            temp=np.fabs(result)
            pos=temp.argsort() 
            pos=pos[::-1]#反向，得到前面L个大的位置
            result[pos[K:]]=0
            result_temp=result       
        return  result
    
    #IRLS算法函数
    def cs_irls(self,y,T_Mat):   
        L=math.floor((y.shape[0])/4)
        hat_x_tp=np.dot(T_Mat.T ,y)
        epsilong=1
        p=1 # solution for l-norm p
        times=1
        while (epsilong>10e-9) and (times<L):  #迭代次数
            weight=(hat_x_tp**2+epsilong)**(p/2-1)
            Q_Mat=np.diag(1/weight)
            #hat_x=Q_Mat*T_Mat'*inv(T_Mat*Q_Mat*T_Mat')*y
            temp=np.dot(np.dot(T_Mat,Q_Mat),T_Mat.T)
            temp=np.dot(np.dot(Q_Mat,T_Mat.T),np.linalg.inv(temp))
            hat_x=np.dot(temp,y)        
            if(np.linalg.norm(hat_x-hat_x_tp,2) < np.sqrt(epsilong)/100):
                epsilong = epsilong/10
            hat_x_tp=hat_x
            times=times+1
        return hat_x
    
    #OLS算法函数（未完成待修改）
    def cs_ols(self,y,D):    
        L=D.shape[1]
        residual=y  #初始化残差
        index=np.zeros((L),dtype=int)
        for i in range(L):
            index[i]= -1
        result=np.zeros((L))
        for j in range(L):  #迭代次数
            product=np.fabs(np.dot(D.T,residual))
            pos=np.argmax(product)  #最大投影系数对应的位置 
            pos_temp=np.argsort(np.fabs(np.dot(np.linalg.pinv(D[:,pos]),y)))#对应步骤2
            index[j]=pos
            my=np.linalg.pinv(D[:,index>=0])        
            a=np.dot(my,y)      
            residual=y-np.dot(D[:,index>=0],a)
        result[index>=0]=a
        return  result
    
    #OMP算法函数
    def cs_omp(self,y,D):    
        #L=math.floor(3*(y.shape[0])/4)
        L=D.shape[1];
        residual=y  #初始化残差
        index=np.zeros((L),dtype=int)
        for i in range(L):
            index[i]= -1
        result=np.zeros((L))
        for j in range(L):  #迭代次数
            product=np.fabs(np.dot(D.T,residual))
            pos=np.argmax(product)  #最大投影系数对应的位置        
            index[j]=pos
            my=np.linalg.pinv(D[:,index>=0]) #最小二乘,看参考文献1      
            a=np.dot(my,y) #最小二乘,看参考文献1     
            residual=y-np.dot(D[:,index>=0],a)
        result[index>=0]=a
        return  result
    
    #SP算法函数
    def cs_sp(self,y,D):     
        K=math.floor(y.shape[0]/3)  
        pos_last=np.array([],dtype=np.int64)
        result=np.zeros((l))
    
        product=np.fabs(np.dot(D.T,y))
        pos_temp=product.argsort() 
        pos_temp=pos_temp[::-1]#反向，得到前面L个大的位置
        pos_current=pos_temp[0:K]#初始化索引集 对应初始化步骤1
        residual_current=y-np.dot(D[:,pos_current],np.dot(np.linalg.pinv(D[:,pos_current]),y))#初始化残差 对应初始化步骤2
    
        while True:  #迭代次数
            product=np.fabs(np.dot(D.T,residual_current))       
            pos_temp=np.argsort(product)
            pos_temp=pos_temp[::-1]#反向，得到前面L个大的位置
            pos=np.union1d(pos_current,pos_temp[0:K])#对应步骤1     
            pos_temp=np.argsort(np.fabs(np.dot(np.linalg.pinv(D[:,pos]),y)))#对应步骤2  
            pos_temp=pos_temp[::-1]
            pos_last=pos_temp[0:K]#对应步骤3    
            residual_last=y-np.dot(D[:,pos_last],np.dot(np.linalg.pinv(D[:,pos_last]),y))#更新残差 #对应步骤4
            if np.linalg.norm(residual_last)>=np.linalg.norm(residual_current): #对应步骤5  
                pos_last=pos_current
                break
            residual_current=residual_last
            pos_current=pos_last
        result[pos_last[0:K]]=np.dot(np.linalg.pinv(D[:,pos_last[0:K]]),y) #对应输出步骤  
        return  result
    
    def train(self,img_cs_1d,mat_dct_1d,Phi,_type = 'SP'):
        l = mat_dct_1d.shape[0]
        r = img_cs_1d.shape[1]
        #重建
        sparse_rec_1d=np.zeros((l,r))   # 初始化稀疏系数矩阵    
        Theta_1d=np.dot(Phi,mat_dct_1d)   #测量矩阵乘上基矩阵
        for i in range(r):
            print('正在重建第',i,'列。。。')
            #print(img_cs_1d[:,i].shape)
            #print(Theta_1d.shape)
            if _type == "SP":
                column_rec=self.cs_sp(img_cs_1d[:,i],Theta_1d)  #利用SP算法计算稀疏系数
            elif _type == "OMP":
                column_rec=self.cs_omp(img_cs_1d[:,i],Theta_1d) #利用OMP算法计算稀疏系数
            elif _type == "IRLS":
                column_rec=self.cs_irls(img_cs_1d[:,i],Theta_1d)  #利用IRLS算法计算稀疏系数
            elif _type == "IHT":
                column_rec=self.cs_IHT(img_cs_1d[:,i],Theta_1d)  #利用IHT算法计算稀疏系数
            elif _type == "CoSaMP":
                column_rec=self.cs_CoSaMP(img_cs_1d[:,i],Theta_1d)  #利用CoSaMP算法计算稀疏系数
            else:
                column_rec=self.cs_sp(img_cs_1d[:,i],Theta_1d)
            #print(column_rec.shape)
            sparse_rec_1d[:,i]=column_rec;        
        img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #稀疏系数乘上基矩阵
        return img_rec
        
        
        
#读取图像，并变成numpy类型的 array
#im = np.array(Image.open('lena.bmp'))#图片大小256*256
im = np.fromfile("out.bin",dtype=np.float)
im.shape = 601,626
#im = mpimg.imread('zs.jpg').astype(np.float)
print(im.shape)
l,r = im.shape
print(l,r)

"""
#生成高斯随机测量矩阵
sampleRate=0.7  #采样率
Phi=np.random.randn(int(r*sampleRate),l)
"""
sampleRate=1
Phi=np.eye(l)
Phi1=np.eye(l)
sam_n = int(l*(1-sampleRate))
fl = np.ones(l)
print(fl.shape)
for i in range(sam_n):
    Phi = np.delete(Phi,random.randint(0,Phi.shape[0]-1),0)
    while True:
        t = random.randint(0,Phi1.shape[0]-1)
        if fl[t] == 0:
            continue
        else:
            Phi1[t] = np.zeros(l)
            fl[t] = 0
            break;
        

"""
#生成稀疏基DCT矩阵
mat_dct_1d=np.zeros((l,l))
v=range(l)
for k in range(0,l):  
    dct_1d=np.cos(np.dot(v,k*math.pi/r))
    if k>0:
        dct_1d=dct_1d-np.mean(dct_1d)
    mat_dct_1d[:,k]=dct_1d/np.linalg.norm(dct_1d)
"""
mat_dct_1d = np.fromfile("dictionary1.bin",dtype=np.float)
mat_dct_1d.shape = 601,601

#随机测量
img_cs_1d=np.dot(Phi,im)

cs = CS()
img_rec = cs.train(img_cs_1d,mat_dct_1d,Phi,"SP")

#显示重建后的图片
image2=Image.fromarray(img_rec)
image2.show()
plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(img_rec)
plt.subplot(2, 2, 2)
plt.imshow(im)
plt.subplot(2, 2, 3)
plt.imshow(mat_dct_1d)
plt.subplot(2, 2, 4)
plt.imshow(np.dot(Phi1,im))
plt.show()
print(mat_dct_1d.shape)
scipy.misc.imsave('SP.jpg', img_rec)
scipy.misc.imsave('im.jpg', im)
scipy.misc.imsave('mat_dct_1d.jpg', mat_dct_1d)
scipy.misc.imsave('img_cs_1d.jpg', np.dot(Phi,im))
scipy.misc.imsave('sparse_rec_1d.jpg', sparse_rec_1d)