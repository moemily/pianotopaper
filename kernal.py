# -*- coding: utf-8 -*-
import numpy as np
import cv2
import sys, os 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import itertools
import math
import copy

def black_rect(img):
    print("in black_rect")
    key_len = 200 #200 is magic number
    img_pad = np.ones((HEIGHT, WIDTH+key_len))
    img_pad[:,0:WIDTH] = cont
    for i in range(0,HEIGHT):
        for j in range(0,WIDTH):
            black = False
            if img_pad[i,j] == 0:
                pp = j 
                for p in range(j,j+key_len):
                    if img_pad[i,p] == 0:
                        black = True
                        pp = p
                if black:
                    img_pad[i,j:pp] = 0
    rect_b = img_pad[:, 0:WIDTH]
    return(rect_b)

def white_rect(img):
    print("in white_rect")
    key_len = 200 #500 is magic number
    img_pad = np.ones((HEIGHT, WIDTH+key_len))
    img_pad[:,0:WIDTH] = cont
    for i in range(0,HEIGHT):
        for j in range(0,WIDTH):
            white = False
            if img_pad[i,j] == 1:
                pp = j
                for p in range(j,j+key_len):
                    if img_pad[i,p] == 1:
                        white = True
                        pp = p
                if white:
                    img_pad[i,j:pp] = 1
    rect_w = img_pad[:, 0:WIDTH]
    print("here")
    return(rect_w)
                    
def penalize(gimg_p):
    print("in penalize")
    ker = np.ones((7,7))
    bw = np.zeros((HEIGHT, WIDTH))
    for i in range(3,HEIGHT+3):
        for j in range(3,WIDTH+3):
            diff = ker - gimg_p[i-3:i+4, j-3:j+4]
            pen = np.sum(np.sum(diff))
            pen_norm = pen / 49
            if pen_norm > .3: #.2 is magic number
                pen_norm = 1
            else:
                pen_norm = 0
            bw[i-3,j-3] = pen_norm
    return bw

def get_edges(img, mini, maxi):
    print("in get_edges")
    #bi = cv2.GaussianBlur(rect_p, (7,7,), .2)
    edge = np.zeros((HEIGHT, WIDTH))
    cv2.Canny(img, mini, maxi, apertureSize = 3)
    return edge


def do_stuff():
    #STRAIGHT PIANO VIDEO
    
    cap = cv2.VideoCapture("EveryNoteOneFinger.mp4")
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    global WIDTH
    global HEIGHT 
    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frame_cnt, HEIGHT, WIDTH, 3), np.dtype('uint8'))
    print(WIDTH, HEIGHT)
    fc = 0
    ret = True
    while (fc < frame_cnt and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    #0 is magic number, frame without hands in it
    gimg = cv2.cvtColor(buf[0], cv2.COLOR_BGR2GRAY)
    cap.release()
    
    # #SKEWED PIANO JPEG
    
    # pic = cv2.imread("skew.jpg")
    # global WIDTH
    # global HEIGHT 
    # WIDTH = pic.shape[1]
    # HEIGHT = pic.shape[0]
    # print(WIDTH, HEIGHT)
    # gimg = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    
    gimg_n = gimg / 255
    gimg_p = np.zeros((HEIGHT + 6, WIDTH + 6))
    gimg_p[3:HEIGHT+3, 3:WIDTH+3] = gimg_n
    
    bw = penalize(gimg_p)
    #rect_b = black_rect(bw)
    #rect_w = white_rect(rect_b)
    edge = get_edges(bw,0,255)
    
    
    #print("superimposing")
    #for i in range(0,HEIGHT):
        #for j in range(0,WIDTH):
            #if cont[i,j] == 0:
                #gi_norm[i,j] = 0
    
    cv2.imshow("window", edge)


if __name__ == "__main__":
    do_stuff()
