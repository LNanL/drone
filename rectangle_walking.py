# -*- coding:utf-8 -*-
import numpy as np
import imutils
import serial
import time
import cv2


#Position
Postion_x = 80
Postion_y = 60

land = 0
turn = False
num = 0

cap = cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,240)
print("start reading video...")
time.sleep(2.0)
print("start working")


if __name__ == '__main__':
    while(True):
        
        ret,frame = cap.read()
        frame = imutils.resize(frame, width=160) #调整图片大小
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(gray,80, 255, cv2.THRESH_BINARY_INV)#转化为二值图
        mask = cv2.dilate(img, None, iterations=1)  #膨胀一下，补足一些残缺部分
        
        #寻找x坐标
        x_acc = np.sum(mask,axis=0)
        x_diff = np.diff(x_acc)
        x_index1 = np.argmax(x_diff)
        x_index2 = np.argmin(x_diff)+1
        x_max = max(x_diff)
        x_min = min(x_diff)

        #寻找y坐标
        y_acc = np.sum(mask,axis =1)
        y_diff = np.diff(y_acc)
        y_index1 = np.argmax(y_diff)
        y_index2 = np.argmin(y_diff)+1
        y_max = max(y_diff)
        y_min = min(y_diff)
        
        #判断是否为拐角
        if(x_max <2000 and x_min>-2000 and y_max < 2000 and y_min > -2000):
            if turn == False:
                num++
                turn = True
        else:
            turn = False
        
        print('num:',num)
        
        #矩形框右下角拐角为第一个点，无人机顺时针飞行
        if(num==1):
            Postion_x = 82       #控制pitch正向条件
            Postion_y = (y_index1+y_index2)//2
        elif(num==2):
            Postion_x = (x_index1+x_index2)//2
            Postion_y = 62       #控制roll正向条件
        elif(num==3):
            Postion_x = 75       #控制pitch反向条件
            Postion_y = (y_index1+y_index2)//2
        elif(num==4):
            Postion_x = (x_index1+x_index2)//2
            Postion_y = 55       #控制pitch反向条件
        else:
            land = 0x10          #降落标志位
        
        cv2.circle(frame,(Postion_x,Postion_y ),4,(0,0,255),-1)

        print('Postion_x,Postion_y',Postion_x,Postion_y)

        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)

        k  = cv2.waitKey(2)
        if k==27:
            break

