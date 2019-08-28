import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform


frame = cv2.imread("cut3.JPG")
frame = imutils.resize(frame, width=320)    #调整图片大小

edges = cv2.Canny(frame,100,300)   #边缘检测，只是为了显示

#提取白色区域
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, np.array((0., 0., 160)), np.array((180., 15.,255.)))
mask = cv2.medianBlur(mask,3)           #去椒盐噪

#提取白色区域轮廓点
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
aimcnt = None


if len(cnts) > 0:
    #根据面积大小，从大到小排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    #找到目标轮廓四个角点
    for i in cnts:
        arc = cv2.arcLength(i, True)    #计算周长，下面函数一般要以周长为参数
        approx = cv2.approxPolyDP(i, 0.1 * arc, True)  #提取近似轮廓点
        if len(approx) == 4:
            aimcnt = approx
            break

#rect = cv2.minAreaRect(cnts[0])
#box = np.int0(cv2.boxPoints(rect))     #这里也能得到四个轮廓点，只是在这幅图中效果没有上面方法理想


#绘制四个目标轮廓点
newFrame=frame.copy()
for i in aimcnt:
    cv2.circle(newFrame, (i[0][0],i[0][1]),4, (0, 0, 255), -1)


#提取目标区域
aimFrame = four_point_transform(frame, aimcnt.reshape(4, 2))
aimMask = four_point_transform(mask, aimcnt.reshape(4, 2))
aimEdges =  four_point_transform(edges, aimcnt.reshape(4, 2))

aimGray = cv2.cvtColor(aimFrame,cv2.COLOR_BGR2GRAY)   #转化为灰度图，进行霍夫圆检测

#检测圆
circles = cv2.HoughCircles(aimGray,cv2.HOUGH_GRADIENT,1,100,param1=100,param2=20,minRadius=1,maxRadius=20)

#绘出圆
if circles is not None:
        for i in circles[0,:]:
            cv2.circle(aimFrame,(i[0],i[1]),i[2],(255,0,0),2)
            Circles_x = int(i[0])
            Circles_y = int(i[1])
        cv2.line(aimFrame, (circles[0][0][0], circles[0][0][1]), (circles[0][1][0], circles[0][1][1]), (255, 0, 0), 2)
else:
    Circles_x = 160  #开启摄像头时的图像中点
    Circles_y = 120


cv2.imshow('newFrame',newFrame)
cv2.imshow('mask',mask)
cv2.imshow('edges',edges)

cv2.imshow('aimFrame',aimFrame)
cv2.imshow('aimMask',aimMask)
cv2.imshow('aimEdges',aimEdges)


cv2.waitKey(0)
