import cv2
import numpy as np
import matplotlib.pyplot as plt


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


#模版匹配
img = cv2.imread('/Users/app/Desktop/Deep-Learning-Fundamentals-and-Frameworks-main/model/OpenCV/图像操作/lena.jpg')
face = cv2.imread('/Users/app/Desktop/Deep-Learning-Fundamentals-and-Frameworks-main/model/OpenCV/图像操作/截屏2026-03-17 20.08.51.png')
face_down = cv2.pyrDown(face)
face_down = cv2.pyrDown(face_down)
h,w = face_down.shape[:2]
print(img.shape)
print('-----')
print(face_down.shape)
print('-----')
print(h,w)

#进行模版匹配
method = ['cv2.TM_SQDIFF_NORMED','cv2.CCORR_NORMED','cv2.CCOEFF_NORMED']#以上三者都是归一化之后的，加了归一化更可靠一些
#第一个是平常差匹配，越接近0越好，第三个是消算系数，越接近1越好
res = cv2.matchTemplate(img,face_down,cv2.TM_SQDIFF_NORMED)#平方差匹配
print(res.shape)

#找最优匹配位置
min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
print(min_val,max_val,min_loc,max_loc)#对于平方差匹配来说，找的是min_loc的点，这个点是匹配上的那个板块的左上角
ans = img[107:107+92,90:90+77]
ans_hstack = np.hstack((ans,face_down))
cv2.imshow('img',ans_hstack)
cv2.waitKey(0)
cv2.destroyAllWindows()


#匹配多个对象：找同一个小块
img_rgb = cv2.imread('mario.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('mario_coin.jpg', 0)
h, w = template.shape[:2]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
# 取匹配程度大于80%的坐标
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):  # *号表示可选参数
    bottom_right = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 2)

cv2.imshow('img_rgb', img_rgb)
cv2.waitKey(0)


#直方图
#cv2.calcHist(img,channels,mask,histSize,ranges)
#img:原图像格式为uint8或者float32，当传入函数时应该用中括号[img]括起来
#channels:如果没有进行一个灰度的转换，可以指定一下，BGR，分别用[0][1][2]来指定哪个通道
#mask:掩膜图像，统计整幅图像直方图就设置为None，如果只想统计某一部分，就用掩码
#histSize:BIN的数目，也应用中括号括来，原先是256，可以指定0-10为一个BIN，11-20是一个BIN，这样压缩范围
#ranges:像素值范围通常：0-256

img = cv2.imread('/Users/app/Desktop/Deep-Learning-Fundamentals-and-Frameworks-main/model/OpenCV/图像操作/cat.jpg')
color = ['b','g','r']
for i,col in enumerate(color):#枚举的格式
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    print(histr.shape)
    plt.plot(histr,color = col)#用对应颜色画出对应的直方图
    plt.xlim([0,256])#设置x轴坐标为0到256
plt.show()

#掩码mask操作
#掩码本质就是一个黑白图，需要与原图片大小一致，黑色区域（像素为0）不统计，白色区域（像素255）统计，
mask = np.zeros(img.shape[:2],dtype = np.uint8)#img.shape[:2]取的是图像的长宽
mask[100:300,100:300] = 255#把中间这一块设置为白色，表示只看这一块
#掩码作用到原图上
masked_img = cv2.bitwise_and(img,img,mask = mask)#img与img做按位与运算，但是只在mask区域上
res = cv2.calcHist([masked_img],[0],mask,[256],[0,256])
cv2.imshow('masked_img',masked_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.plot(res,color = color[0])
plt.xlim([0,256])
plt.show()


#均衡化：对比度拉高，让暗的地方更暗，亮的地方更亮，适合偏暗偏灰，雾蒙蒙，亮度分布太集中的图像，原来不太能注意到的细节都能看到
img = cv2.imread('/Users/app/Desktop/Deep-Learning-Fundamentals-and-Frameworks-main/model/OpenCV/图像操作/lena.jpg',0)

plt.hist(img.ravel(),256)
plt.show()

equ = cv2.equalizeHist(img)
plt.hist(equ.ravel(),256)
plt.show()

res = np.hstack((img,equ))
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()

#自适应均衡化
#创建CLAHE对象
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#第一个参数：对比度限制参数，值越大对比度增强越明显，值太大噪声也可能更明显，常用2.0
#第二个参数：把图像分成多少小块来处理，分块越小局部增强越强，分块太小图像可能不自然，分块太大又会接近全局均衡化
#应用CLAHE
restemp = clahe.apply(img)#一般用于单通道图像，也就是灰度图
cv2.imshow('res',restemp)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


#傅立叶变换


