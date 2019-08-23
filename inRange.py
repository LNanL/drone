import cv2
import numpy as np
import imutils


frame = cv2.imread("UAV.jpg")
frame = imutils.resize(frame, width=320)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

mask_red = cv2.inRange(hsv, np.array((110., 50., 50)), np.array((170., 255., 255.)))
mask_red = cv2.medianBlur(mask_red, 3)
cnts_red = cv2.findContours(mask_red.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts_red = cnts_red[0] if imutils.is_cv2() else cnts_red[1]

mask_blue = cv2.inRange(hsv, np.array((40., 50., 50.)), np.array((140., 255., 255.)))
mask_blue = cv2.medianBlur(mask_blue, 3)
cnts_blue = cv2.findContours(mask_blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts_blue = cnts_blue[0] if imutils.is_cv2() else cnts_blue[1]

if (len(cnts_red) > 0):
    area = [cv2.contourArea(i) for i in cnts_red]
    index = np.argmax(area)
    rect_red = cv2.minAreaRect(cnts_red[index])
    box_red = np.int0(cv2.boxPoints(rect_red))
    cv2.drawContours(frame, [box_red], 0, (0, 0, 255), 2)
    cv2.putText(frame,"head",(int(rect_red[0][0]),int(rect_red[0][1])),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)


if (len(cnts_blue) > 0):
    area = [cv2.contourArea(i) for i in cnts_blue]
    index = np.argmax(area)
    rect_blue = cv2.minAreaRect(cnts_blue[index])
    box_blue = np.int0(cv2.boxPoints(rect_blue))
    cv2.drawContours(frame, [box_blue], 0, (0, 0, 255), 2)
    cv2.putText(frame, "tail", (int(rect_blue[0][0]), int(rect_blue[0][1])),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

cv2.line(frame,(int(rect_red[0][0]),int(rect_red[0][1])), (int(rect_blue[0][0]), int(rect_blue[0][1])),(0, 255, 0),2)

cv2.imshow('frame', frame)
cv2.imshow('mask_red', mask_red)
cv2.imshow('mask_blue', mask_blue)

cv2.waitKey(0)




