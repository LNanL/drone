import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
time.sleep(1.0)

OK = True

print('ready...')

while (True):
    ret, frame = cap.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_yellow = cv2.inRange(hsv, np.array((26., 43., 46)), np.array((34., 255., 255.)))
    mask_yellow = cv2.medianBlur(mask_yellow, 3)
    cnts_yellow = cv2.findContours(mask_yellow.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_yellow = cnts_yellow[0] if imutils.is_cv2() else cnts_yellow[1]

    if (len(cnts_yellow) > 0):
        area = [cv2.contourArea(i) for i in cnts_yellow]
        index = np.argmax(area)
        maxarea = max(area)
        rect_yellow = cv2.minAreaRect(cnts_yellow[index])
        box_yellow = np.int0(cv2.boxPoints(rect_yellow))
        cv2.drawContours(frame, [box_yellow], 0, (0, 0, 255), 2)

        if OK == True and maxarea > 20:
            cv2.imwrite('yellow.jpg',frame)
            OK = False

    cv2.imshow('frame', frame)
    cv2.imshow('mask_yellow', mask_yellow)

    cv2.waitKey(2)
