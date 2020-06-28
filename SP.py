# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:02:53 2018

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
import datetime



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
sampleRate=0.5
Phi=np.eye(l)
Phi1=np.eye(l)
sam_n = int(l*(1-sampleRate))
fl = np.ones(l)
print(fl.shape)
t=0
for i in range(sam_n):
    k = random.randint(0,Phi.shape[0]-1)
    Phi = np.delete(Phi,k,0)
    #print(k)
    for j in range(Phi1.shape[0]):
        if fl[j] == 0:
            continue
        k -= 1
        if k<0:
            Phi1[j] = np.zeros(l)
            fl[j] = 0
            t+=1
            break;
print(t,sam_n)

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
mat_dct_1d.shape = 601,1202
"""


#随机测量
img_cs_1d=np.dot(Phi,im)


def cs_samp2(y,D,theta,s=1):
    (M,N) = D.shape
    K = (int)(np.count_nonzero(theta)/2)
    K = max(K,s)
    pos_num = np.array([],dtype=np.int64)
    pos_num = np.argsort(np.fabs(theta))
    pos_num = pos_num[::-1]
    pos_num = pos_num[0:K]
    theta_ls_last = theta[pos_num]
    res = y - np.dot(D[:,pos_num],theta_ls_last)
    theta = theta * 0.
    i=0
    while i<50:
        product = np.fabs(np.dot(D.T,res))
        pos_temp = np.argsort(product)
        pos_temp = pos_temp[::-1]
        Is = np.union1d(pos_num,pos_temp[0:K])
        #if Is.shape[0] > M:
        #    break
        At = D[:,Is]
        theta_ls = np.dot(np.dot(np.linalg.inv(np.dot(At.T,At)),At.T),y)
        pos = np.argsort(np.fabs(theta_ls))
        pos = pos[::-1]
        pos_num_cur = Is[pos[0:K]]
        theta_ls = theta_ls[pos[0:K]]
        res_cur = y - np.dot(At[:,pos[0:K]],theta_ls)
        nres_cur = np.linalg.norm(res_cur)
        nres = np.linalg.norm(res)
        if nres_cur < 1e-1: #对应步骤5  
            #print(nres_cur)
            res = res_cur
            pos_num = pos_num_cur
            theta_ls_last = theta_ls
            break
        if nres_cur >= nres:
            K+=s
        else:
            res = res_cur
            pos_num = pos_num_cur
            theta_ls_last = theta_ls
        i+=1
    theta[pos_num] = theta_ls_last
    return theta

def cs_samp(y,D,s = 5):
    (M,N) = D.shape
    theta = np.zeros((N))
    pos_num = np.array([],dtype=np.int64)
    res = y
    K=s
    i=0
    while i<500:
        product = np.fabs(np.dot(D.T,res))
        pos_temp = np.argsort(product)
        pos_temp = pos_temp[::-1]
        Is = np.union1d(pos_num,pos_temp[0:K])
        #if Is.shape[0] > M:
        #    break
        At = D[:,Is]
        theta_ls = np.dot(np.dot(np.linalg.inv(np.dot(At.T,At)),At.T),y)
        pos = np.argsort(np.fabs(theta_ls))
        pos = pos[::-1]
        pos_num_cur = Is[pos[0:K]]
        theta_ls = theta_ls[pos[0:K]]
        res_cur = y - np.dot(At[:,pos[0:K]],theta_ls)
        nres_cur = np.linalg.norm(res_cur)
        nres = np.linalg.norm(res)
        if nres_cur < 1e-1: #对应步骤5  
            #print(nres_cur)
            res = res_cur
            pos_num = pos_num_cur
            theta_ls_last = theta_ls
            break
        if nres_cur >= nres:
            K+=s
        else:
            res = res_cur
            pos_num = pos_num_cur
            theta_ls_last = theta_ls
        i+=1
    theta[pos_num] = theta_ls_last
    return theta

#SP算法函数
def cs_sp(y,D,K=50):     
    #K=math.floor(y.shape[0]/3)  
    l=D.shape[1]
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

def cs_omp(y,D,L):    
    #L=math.floor(3*(y.shape[0])/4)
    #L=r;
    residual=y  #初始化残差
    index=np.zeros((D.shape[1]),dtype=int)
    for i in range(l):
        index[i]= -1
    result=np.zeros((D.shape[1]))
    for j in range(L):  #迭代次数
        product=np.fabs(np.dot(D.T,residual))
        pos=np.argmax(product)  #最大投影系数对应的位置        
        index[j]=pos
        my=np.linalg.pinv(D[:,index>=0]) #最小二乘,看参考文献1           
        a=np.dot(my,y) #最小二乘,看参考文献1     
        residual=y-np.dot(D[:,index>=0],a)
    result[index>=0]=a
    return  result

def cs_IHT(y,D,K):    
    #K=math.floor(y.shape[0]/3)  #稀疏度    
    result_temp=np.zeros((D.shape[1]))  #初始化重建信号   
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
def cs_irls(y,T_Mat,L):   
    #L=math.floor((y.shape[0])/4)
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

#CoSaMP算法函数
def cs_CoSaMP(y,D,K):     
    #K=math.floor(y.shape[0]/3)  
    pos_last=np.array([],dtype=np.int64)
    result=np.zeros((l))

    product=np.fabs(np.dot(D.T,y))
    pos_temp=product.argsort() 
    pos_temp=pos_temp[::-1]#反向，得到前面L个大的位置
    pos_current=pos_temp[0:2*K]#初始化索引集 对应初始化步骤1
    residual_current=y-np.dot(D[:,pos_current],np.dot(np.linalg.pinv(D[:,pos_current]),y))#初始化残差 对应初始化步骤2

    while True:  #迭代次数
        product=np.fabs(np.dot(D.T,residual_current))       
        pos_temp=np.argsort(product)
        pos_temp=pos_temp[::-1]#反向，得到前面L个大的位置
        pos=np.union1d(pos_current,pos_temp[0:2*K])#对应步骤1     
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

#重建
sparse_rec_1d=np.zeros((mat_dct_1d.shape[1],r))   # 初始化稀疏系数矩阵    
Theta_1d=np.dot(Phi,mat_dct_1d)   #测量矩阵乘上基矩阵
column_rec_temp = np.zeros((Theta_1d.shape[1]))

starttime = datetime.datetime.now()
# for i in range(r):
for i in range(1):
    print('正在重建第',i,'列。。。')
    #print(img_cs_1d[:,i].shape)
    #print(Theta_1d.shape)
    #column_rec=cs_omp(img_cs_1d[:,i],Theta_1d,50) 
    #column_rec=cs_irls(img_cs_1d[:,i],Theta_1d,6)  #利用SP算法计算稀疏系数
    column_rec=cs_samp(img_cs_1d[:,i],Theta_1d,1)
    #print(column_rec.shape)
    sparse_rec_1d[:,i]=column_rec;
    column_rec_temp = column_rec        
    
endtime = datetime.datetime.now()
print((endtime - starttime))

img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #稀疏系数乘上基矩阵
img_rec.tofile("result.bin")

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

sample=np.dot(Phi1,im)
sample.tofile("sample.bin")

plt.show()
print(mat_dct_1d.shape)
print(sparse_rec_1d.shape)
scipy.misc.imsave('SP.jpg', img_rec)
scipy.misc.imsave('im.jpg', im)
scipy.misc.imsave('mat_dct_1d.jpg', mat_dct_1d)
scipy.misc.imsave('img_cs_1d.jpg', np.dot(Phi,im))
scipy.misc.imsave('img_cs_1d_2.jpg', np.dot(Phi1,im))
scipy.misc.imsave('sparse_rec_1d.jpg', sparse_rec_1d)
