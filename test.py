# -*- coding: utf-8 -*-
import numpy as np
import cv2
import sys, os 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

print("Hi")

cap = cv2.VideoCapture("00006.mp4")
frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
buf = np.empty((frame_cnt, frame_height, frame_width, 3), np.dtype('uint8'))

fc = 0
ret = True

while (fc < frame_cnt and ret):
    ret, buf[fc] = cap.read()
    fc += 1
gi = cv2.cvtColor(buf[4], cv2.COLOR_BGR2GRAY)
bi = cv2.GaussianBlur(gi, (7,7,), .2)
med_val = np.median(bi)
icannybelieveit = cv2.Canny(bi, 0, 255, apertureSize=3)

dave = cv2.imread('Untitled.jpg')
gray = cv2.cvtColor(dave,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,0,255,apertureSize = 3)
cap.release()
#cv2.namedWindow('frame 10')
#cv2.imshow('frame 10', icannybelieveit)
#input("Press enter to fuck off") 

#lines = cv2.HoughLines(icannybelieveit,100,np.pi/180,200)

#cv2.imshow('corso is a bad person, of corso', edges)
lines = cv2.HoughLines(icannybelieveit, 1, np.pi/180, 160) 
cnt = 0
for line in lines:
    # for x1, y1, x2, y2, in line:
    #     cv2.line(dave, (x1,y1), (x2,y2), (0,0,255),2)
    for rho, theta in line:
        delta_v = np.pi/float(32)
        delta_h = np.pi/float(128)
        diff_h = abs(theta - 3*np.pi/2) if theta > np.pi else abs(theta - np.pi/2)
        diff_v = abs(theta - np.pi) if theta > np.pi/2 else abs(theta)
        if diff_h > delta_h and diff_v > delta_v:
            continue
        print(theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 3000*(-b))
        y1 = int(y0 + 3000*(a))
        x2 = int(x0 - 3000*(-b))
        y2 = int(y0 - 3000*(a))
        cv2.line(buf[4], (x1,y1), (x2,y2), (0,0,255),2)
    cnt += 1

cv2.imshow("fuck this", buf[4])

# cv2.waitKey(0)
# while (1) :
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q') :
#         cv2.destroyAllWindows()
#         exit()

#cv2.imwrite('houghlines.jpg',img)


def hough_line(im, edge_im, num_rhos, num_thetas, t_count):
    N, M = edge_im.shape[:2]
    Nhalf, Mhalf = N/2 , M/2
    d = np.sqrt(np.square(N) + np.square(M))
    dtheta = 180 / num_thetas
    drho = (2*d) / num_rhos


#test shit