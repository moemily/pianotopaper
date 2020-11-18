# -*- coding: utf-8 -*-
import numpy as np
import cv2
import sys, os 
import matplotlip.pyplot as plt
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
icannybelieveit = cv2.Canny(bi, 128, int(max(0, 0.7*med_val)), int(min(255, 1.3*med_val)))


cap.release()
cv2.namedWindow('frame 10')
cv2.imshow('frame 10', icannybelieveit)
input("Press enter to fuck off")

def hough_line(im, edge_im, num_rhos, num_thetas, t_count):
    N, M = edge_im.shape[:2]
    Nhalf, Mhalf = N/2 , M/2
    d = np.sqrt(np.square(N) + np.square(M))
    dtheta = 180 / num_thetas
    drho = (2*d) / num_rhos
