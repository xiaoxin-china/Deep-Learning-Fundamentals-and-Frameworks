import cv2#默认读取是一个BGR
import matplotlib.pyplot as plt
import numpy as np
"""
#读取展示：

img = cv2.imread('cat.jpg')#读取成一个numpy三维数组
img2 = cv2.imread('cat.jpg',cv2.IMREAD_GRAYSCALE)#读取灰度图
img3 = cv2.imread('cat.jpg',cv2.IMREAD_COLOR)#默认读取彩色图

cv2.imshow('cat', img)#创建一个窗口，用来展示图片
cv2.waitKey(0)#等待时间，毫秒级，0表示任意键终止
cv2.destroyAllWindows()#销毁残留窗口，释放内存

#保存：
cv2.imwrite('mycat.png',img2)

#属性：
print(img.shape)#彩色图：h,w,t灰度图：h,w
print(img.size)#像素点个数


#视频操作
vc = cv2.VideoCapture('test.mp4')#打开，也可以捕获摄像头，0，1
#检查是否打开正确
if vc.isOpened():
    open,frame = vc.read()#open：是否读取成功（布尔值），frame：当前帧的数据（numpy）
else:
    open = False

while open:
    ret,frame = vc.read()#
    if frame is None:#检查是否到视频结尾
        break
    if ret == True:#是读取成功
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#彩色图转为灰度图，存储在gray里
        cv2.imshow('result',gray)#创建一个名字叫result的窗口，里面展示gray
        if cv2.waitKey(10) & 0xFF == 27:#每一帧等10毫秒播放完或者按任意键退出
            break
vc.release()#释放视频捕获占用的内存
cv2.destroyAllWindows()#关闭所有窗口


#ROI:截取部分图像数据
img = cv2.imread('cat.jpg')
cat = img[0:200,0:200]
cv2.imshow('cat',cat)
cv2.waitKey(0)
cv2.destroyAllWindows()

#颜色通道提取
b,g,r = cv2.split(img)

#只保留R
cur_img = img.copy()
cur_img[:,:,0] = 0
cur_img[:,:,1] = 0

#只保留G
cur_img = img.copy()
cur_img[:,:,0] = 0
cur_img[:,:,2] = 0

#只保留B
cur_img = img.copy()
cur_img[:,:,1] = 0
cur_img[:,:,2] = 0

#属性
print(r)
print(b.shape)

#切完之后分别处理，再组合在一起
img = cv2.merge((b,g,r))
print(img.shape)





#边界填充
top_size,bottom_size,left_size,right_size = (50,50,50,50)
replicate = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_REPLICATE)#复制法：把边界复制下去填充
reflect = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_REFLECT)#反射法：镜像反射过去填充
reflect101 = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_REFLECT101)#反射法：也是镜像过去，不过边界会弄得更好，去掉了中间重复的轴
wrap = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_WRAP)#外包装法：四面都是相同的一张图片赋值拼过去
constant = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_CONSTANT,value=0)#常量法：常数值填充




#数值计算
img_cat = cv2.imread('cat.jpg')
img_dog = cv2.imread('dog.jpg')

img_cat2 = img_cat + 10#在每个像素点上都加10
cv2.imshow('cat',img_cat2)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_cat_plus1 = img_cat+img_cat2#相当于每个点都%256
img_cat_plus2 = cv2.add(img_cat,img_cat2)#加起来如果不越界，就是这个值，越界就是取最大255

#图像融合
#融合前先确保图像大小尺寸是一样，不一样就resize
img_dog = cv2.resize(img_dog,(500,414))#W,H
img_dog_temp2 = cv2.resize(img_dog,(0,0),fx=2,fy=2)#按比例放大，x轴放大两倍，y轴放大2倍

res = cv2.addWeighted(img_cat,0.4,img_dog,0.6,0)#第一个图象，权重1，第二个图象，权重2，偏置（提亮）
cv2.imshow('cat&dog',res)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

#图像阈值
img_cat = cv2.imread('cat.jpg')
ret,thresh1 = cv2.threshold(img_cat,127,255,cv2.THRESH_BINARY)#输入图，阈值，赋予值（当满足有关阈值条件后所赋予的值），二值化操作的类型，这里是二分超过阈值赋为255
ret,thresh2 = cv2.threshold(img_cat,127,255,cv2.THRESH_BINARY_INV)#输入图，阈值，赋予值（当满足有关阈值条件后所赋予的值），二值化操作的类型，这里是THRESH_BINARY的反转，小于127的去255
ret,thresh3 = cv2.threshold(img_cat,127,255,cv2.THRESH_TRUNC)#输入图，阈值，赋予值（当满足有关阈值条件后所赋予的值），二值化操作的类型，这里是THRESH_TRUNC的截断操作，大于阈值取阈值
ret,thresh4 = cv2.threshold(img_cat,127,255,cv2.THRESH_TOZERO)#输入图，阈值，赋予值（当满足有关阈值条件后所赋予的值），二值化操作的类型，这里是THRESH_TOZERO，大于阈值全为0，小于阈值不变
ret,thresh5 = cv2.threshold(img_cat,127,255,cv2.THRESH_TOZERO_INV)#输入图，阈值，赋予值（当满足有关阈值条件后所赋予的值），二值化操作的类型，这里是THRESH_TOZERO_INV，小于阈值全为0，大于阈值不变

