import cv2
import numpy as np



"""
#Canny边缘检测

    1）使用高斯滤波器，以平滑图像，滤除噪声
    2）计算图像中每个像素点的提督强度和方向
    3）应用非极大值抑制，以消除边缘检测带来的杂散相应
    4）应用双阈值检测来确定真实和潜在的边缘
    5）通过抑制孤立的弱边缘最终完成边缘检测

img = cv2.imread('/Users/app/Desktop/Deep-Learning-Fundamentals-and-Frameworks-main/model/OpenCV/图像操作/car.png', cv2.IMREAD_GRAYSCALE)
v1 = cv2.Canny(img,120,150)#minvalue,maxvalue,双阈值，如果min指定的比较小,边界比较宽，如果min指定的比较大，边界比较窄,max同理
v2 = cv2.Canny(img,50,100)#如果整体比较高，边界将会明显，清晰，但是得到的边界信息会相对少

res = np.hstack((v1,v2))
cv2.imshow('img',res)
cv2.waitKey(0)
cv2.destroyAllWindows()


#梯度计算--Sobel算子
img = cv2.imread('/Users/app/Desktop/Deep-Learning-Fundamentals-and-Frameworks-main/model/OpenCV/图像操作/car.png', cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)#img:图片,cv2.CV_64F:输出带负数的值，若这里填-1:默认负数截断为0，dx，dy谁为1则最后计算哪个方向的梯度，ksize：sobel算子的大小
cv2.imshow('sobelx',sobelx)
cv2.waitKey(0)
cv2.destroyAllWindows()
#问题：由于永远都是右边减左边，白到黑能显示，黑到白会显示负数，不显示，所以应该取绝对值
sobelx = cv2.convertScaleAbs(sobelx)
cv2.imshow('sobelx',sobelx)
cv2.waitKey(0)
cv2.destroyAllWindows()

#分别计算x和y的梯度权重，然后再求和。不建议直接一起，因为会重影
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)#img:图片,cv2.CV_64F:输出带负数的值，若这里填-1:默认负数截断为0，dx，dy谁为1则最后计算哪个方向的梯度，ksize：sobel算子的大小
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)#img:图片,cv2.CV_64F:输出带负数的值，若这里填-1:默认负数截断为0，dx，dy谁为1则最后计算哪个方向的梯度，ksize：sobel算子的大小
sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)#前两项为权重，后一项为偏置
cv2.imshow('sobelxy',sobelxy)
cv2.waitKey(0)


#scharr算子，和sobel算子思想一样，但是每一项数值都变大
#laplacian算子，对噪音点敏感，所以不会单独用



#高斯金字塔
#向上采样：放大；向下采样：缩小
img = cv2.imread('/Users/app/Desktop/Deep-Learning-Fundamentals-and-Frameworks-main/model/OpenCV/图像操作/AM.png')
cv2.imshow('img',img)
print(img.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()

#上采样
up = cv2.pyrUp(img)
cv2.imshow('up',up)
print(up.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()

#下采样
down = cv2.pyrDown(img)
cv2.imshow('down',down)
print(down.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()



#图像轮廓
#cv2.findContours(img,model,method)     img:传入图像，model：轮廓检索模式（通常用最后一个，默认检测所有，并按照嵌套形式保存，以后用哪个就调用哪个），method:轮廓逼近方法（）
#第一个返回值contours返回轮廓点集,第二个返回值hierarchy返回层级
#model:RETR_EXTERNAL:只检测最外面的轮廓      RETR_LIST:检测所有轮廓并将其保存到一个链表中     RETR_CCOMP:检测所有的轮廓，并将它们组织为两层，顶层是各部分的外部边界，第二层是空洞的边界      RETR_TREE:检索所有的轮廓，并重构嵌套轮廓的整个层次
#method:CHAIN_APPROX_NONE:以Freeman链码的方式输出轮廓，所有其他方法输出多边形（顶点的序列）      CHAIN_APPROX_SIMPLE:压缩水平的，垂直的和斜的部分，也就是函数只保留他们的终点部分

#1、读数据  2、转换成灰度图    3、用阈值转换成二值图
img = cv2.imread('/Users/app/Desktop/Deep-Learning-Fundamentals-and-Frameworks-main/model/OpenCV/图像操作/contours2.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
cv2.imshow('thresh',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)#contours
#传入绘制图像，轮廓，轮廓索引(-1是把所有轮廓都画进来)，颜色模式，线条厚度
#注意需要copy不然原图会变
draw_img = thresh.copy()
draw_img = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
res = cv2.drawContours(draw_img,contours,-1,(0,0,255),2)
cv2.imshow('draw_img',draw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
ret = np.hstack((img,draw_img))
cv2.imshow('ret',ret)
cv2.waitKey(0)
cv2.destroyAllWindows()


#轮廓特征
cnt = contours[0]
#面积
print(cv2.contourArea(cnt))
#周长,True表示闭合的
print(cv2.arcLength(cnt,True))

"""
"""
#轮廓近似
img = cv2.imread('/Users/app/Desktop/Deep-Learning-Fundamentals-and-Frameworks-main/model/OpenCV/图像操作/contours.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret ,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
res = thresh.copy()
contours,hierarchy = cv2.findContours(res,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
"""
"""
res = cv2.drawContours(img,contours,0,(0,0,255),2)
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
cnt = contours[4]#指定外界轮廓
epsilon = 0.02 * cv2.arcLength(cnt,True)#指定比较值为0.1倍的周长
approx = cv2.approxPolyDP(cnt,epsilon,True)#做完这一步还只是一个轮廓
draw_img = img.copy()
res = cv2.drawContours(draw_img,[approx],-1,(0,0,255),2)#把轮廓放到RGB三通道图上
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()

#边界矩形
img = cv2.imread('/Users/app/Desktop/Deep-Learning-Fundamentals-and-Frameworks-main/model/OpenCV/图像操作/contours.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转成灰度图
ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)#转成二值图
res = thresh.copy()#拷贝一份
contours,hierarchy = cv2.findContours(res,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)#返回轮廓点集和层集
cnt = contours[0]#选取一个轮廓
epsilon = 0.1 * cv2.arcLength(cnt,True)

x,y,w,h = cv2.boundingRect(cnt)#求边界矩形
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)#画到原图像上，而非二值图，（x,y）是左上角的点，（x+w，y+h）是右下角的点
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

area = cv2.contourArea(cnt)
x,y,w,h = cv2.boundingRect(cnt)
rect_area = w * h#边界矩形的面积
extent = float(area)/rect_area#轮廓面积和边界矩形比
print("轮廓面积和边界矩形比：",extent)

"""
#模版匹配




