import cv2
import numpy as np
import imutils
import time

cap = cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,240)
time.sleep(2.0)
print("ready...")


if __name__ == '__main__':
    while(True):

        ret,frame = cap.read()
        frame = imutils.resize(frame, width=160)     #调整图片大小

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, th1 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)   #转化为二值图
        mask = cv2.dilate(th1, None, iterations=1)   #膨胀一下，补足一些残缺部分

        ROImask = mask[0:120, 50:110]
        cnts2 = cv2.findContours(ROImask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts2 = cnts2[0] if imutils.is_cv2() else cnts2[1]

        if (len(cnts2) > 0):
            #取得面积最大的轮廓并重绘（减少噪音干扰）
            area = [cv2.contourArea(i) for i in cnts2]
            area = np.array(area)
            index = np.argmax(area)
            img = np.zeros(ROImask.shape)
            im = cv2.drawContours(img, cnts2, index, 255, -1)

            #求交点X坐标 （可以封装成函数）
            x_acc = np.sum(im, axis=0)
            x_diff = np.diff(x_acc)
            x_index1 = np.argmax(x_diff)
            x_index2 = np.argmin(x_diff) + 1
            x_max = max(x_diff)
            x_min = min(x_diff)

            if (x_max < 1000 and x_min > -1000):
                Postion_x = 80
            else:
                Postion_x = (x_index1 + x_index2) // 2 + 50    #ROImask取的范围在[50,110]，所以+50

            # 求交点Y坐标
            y_acc = np.sum(im, axis=1)
            y_diff = np.diff(y_acc)
            y_index1 = np.argmax(y_diff)
            y_index2 = np.argmin(y_diff) + 1
            y_max = max(y_diff)
            y_min = min(y_diff)

            if (y_max < 1000 and y_min > -1000):
                Postion_y = 60
            else:
                Postion_y = (y_index1 + y_index2) // 2

        else:
            Postion_x = 80
            Postion_y = 60

        cv2.circle(frame, (Postion_x, Postion_y), 4, (0, 0, 255), -1)

        print('Postion_x,Postion_y', Postion_x, Postion_y)

        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)

        k  = cv2.waitKey(2)
        if k==27:
            break




