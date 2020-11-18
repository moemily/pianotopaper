# -*- coding: utf-8 -*-
import numpy as np
import cv2
import sys, os 

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
icannybelieveit = cv2.Canny(bi, 0, 255)


cap.release()
cv2.namedWindow('frame 10')
cv2.imshow('frame 10', gi)
cv2.imshow('frame 10', icannybelieveit)
#input("Press enter to fuck off")