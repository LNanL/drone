# -*- coding:utf-8 -*-
import numpy as np
import imutils
import time
import cv2

# Position
Postion_x = 80
Postion_y = 60

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)
time.sleep(2.0)
print("ready...")


if __name__ == '__main__':
    while (True):
        distance = 0
        index = 0
        ok = True
        
        ret, frame = cap.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, th1 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.dilate(th1, None, iterations=1)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        
        if len(cnts) > 0:
            j=0
            for i in cnts:
                x_max = max(i[:,:,0])
                x_min = min(i[:,:,0])
                if(x_max-x_min > distance):
                    distance = x_max-x_min
                    index = j
                j += 1
        
            c = cnts[index]
            
            mid = np.median(c[:,:,1])
            for i in c:
                if i[0][1] - mid > 20 or i[0][1] - mid <-20:
                    ok = False
                    i[0][1] = mid

            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
            Postion_x = int(rect[0][0]//2)
            Postion_y = int(rect[0][1]//2)
        else:
            Postion_x = 80
            Postion_y = 60


        cv2.circle(frame,(Postion_x*2,Postion_y*2),4,(0,0,255),-1)

        print('ok:',ok)
        print('Postion_x,Postion_y', Postion_x, Postion_y)

        cv2.imshow('frame',frame)
        cv2.waitKey(2)




