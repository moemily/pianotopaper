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
    img_pad[:,0:WIDTH] = img
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
    img_pad[:,0:WIDTH] = img
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
            if pen_norm > .3: #.3 is magic number
                pen_norm = 1
            else:
                pen_norm = 0
            bw[i-3,j-3] = pen_norm
    return bw

def get_lines(img, mini, maxi):
    print("in get_lines")
    edge = cv2.Canny(img, mini, maxi, apertureSize = 3)
    lines = cv2.HoughLinesP(edge, 1, np.pi/180, 100, minLineLength = 10, maxLineGap = 250)
    return lines

def get_intersections(lines):
    print("in get_intersections")
    lines_c = lines[1:np.size(lines,0), :, :]
    intersections = []
    lines1 = []
    lines2 = []
    num_inter = 0
    for line in lines:
        for line_c in lines_c:
            x11, y11, x12, y12 = line[0]
            x21, y21, x22, y22 = line_c[0]
            
            y11f = -y11
            y12f = -y12
            y21f = -y21
            y22f = -y22
    
            A1 = y12f - y11f
            B1 = x11 - x12
            C1 = A1 * x11 + B1 * y11f
            A2 = y22f - y21f
            B2 = x21 - x22
            C2 = A2 * x21 + B2 * y21f
            det = A1*B2 - A2*B1
            if det == 0:
                #lines parallel
                continue
            else:
                x = (B2*C1 - B1*C2) / det
                xp = round(x,0)
                yf = (A1*C2 - A2*C1) / det
                y = -yf
                yp = round(y,0)
                xp = yp
                yp = round(x,0)
                #320 is magic number to crop out all the extra lines xp
                if ((xp >= 0) and (xp < HEIGHT) and (yp >= 0) and (yp < WIDTH)) :
                    num_inter = num_inter + 1
                    intersections.append([xp,yp])
                    lines1.append(line[0])
                    lines2.append(line_c[0])
    return (lines1, lines2, intersections, num_inter)

def find_boxes(line1, line2, intersect) :

    # want map of intersections -> line and line -> intersection 
    print("in find_boxes")

    same_line = [] # intersections on the same line, each entry i holds ((intersections), line of intersection, (non intersecting lines))
    boxes = []
    iteration = 0
    for i in range(len(intersect)) :
        for j in range(i+1, len(intersect)) :
            dist = math.dist(intersect[i], intersect[j])
            if  (dist > (1/16 * HEIGHT)):
                if   equalLines(line2[i], line2[j]) and not equalLines(line1[i],line1[j]) : 
                    same_line.append(((i,j), line2[i], (line1[i], line1[j])))  
                elif equalLines(line1[i], line1[j]) and not equalLines(line2[i],line2[j]) : 
                    same_line.append(((i,j), line1[i], (line2[i], line2[j])))
                elif equalLines(line1[i], line2[j]) and not equalLines(line2[i],line1[j]) : 
                    same_line.append(((i,j), line1[i], (line2[i], line1[j])))
                elif equalLines(line2[i], line1[j]) and not equalLines(line1[i],line2[j]) : 
                    same_line.append(((i,j), line2[i], (line1[i], line2[j])))

            # if  (dist > (1/16 * HEIGHT)  and equalLines(line2[i], line2[j]) and not equalLines(line1[i],line1[j])) : 
            #     same_line.append(((i,j), line2[i], (line1[i], line1[j])))  
            # elif (dist > (1/16 * HEIGHT) and equalLines(line1[i], line1[j]) and not equalLines(line2[i],line2[j])) : 
            #     same_line.append(((i,j), line1[i], (line2[i], line2[j])))
            # elif (dist > (1/16 * HEIGHT) and equalLines(line1[i], line2[j]) and not equalLines(line2[i],line1[j])) : 
            #     same_line.append(((i,j), line1[i], (line2[i], line1[j])))
            # elif (dist > (1/16 * HEIGHT) and equalLines(line2[i], line1[j]) and not equalLines(line1[i],line2[j])) : 
            #     same_line.append(((i,j), line2[i], (line1[i], line2[j])))
            if (iteration % 1000 == 0):
                print("Iteration: ", iteration)
            iteration+=1
    print("fuckshit stack")     
    iteration = 0       
    # print(same_line)            
    # find boxes :)
    for l1 in range(len(same_line)) :
        for l2 in range(l1+1,len(same_line)) :
            i1, j1 = same_line[l1][0]
            i2, j2 = same_line[l2][0]
            dist1 = math.dist(intersect[i1], intersect[i2]) # assumes dist(j1,j2) appromately equals dist(i1,i2)
            dist2 = math.dist(intersect[i1], intersect[j2]) # assumes dist(j1,i2) appromately equals dist(i1,j2)
            if dist1 > (1/16 * HEIGHT) and dist2 > (1/16 * HEIGHT) :
                if not equalLines(same_line[l1][1],same_line[l2][1]) : # not the same line of intersection
                    if i1 != i2 and j1 != j2 and i1 != j2 and j1 != j2 : # no shared intersections
                        linei1, linej1 = same_line[l1][2]
                        linei2, linej2 = same_line[l2][2]
                        if equalLines(linei1, linei2) and equalLines(linej1, linej2) : 
                            boxes.append((intersect[i1], intersect[j1], intersect[i2], intersect[j2]))
                        elif equalLines(linei1,linej2) and equalLines(linej1,linei2) : 
                            boxes.append((intersect[i1], intersect[j1], intersect[i2], intersect[j2]))
                # if i1 != i2 and j1 != j2 and i1 != j2 and j1 != j2 : # no shared intersections
                #     linei1, linej1 = same_line[l1][2]
                #     linei2, linej2 = same_line[l2][2]
                #     if equalLines(linei1, linei2) and equalLines(linej1, linej2) : 
                #         dist1 = math.dist(intersect[i1], intersect[i2])
                #         dist2 = math.dist(intersect[j1], intersect[j2])
                #         if dist1 > (1/16 * HEIGHT) and dist2 > (1/16 * HEIGHT) :
                #             boxes.append((intersect[i1], intersect[j1], intersect[i2], intersect[j2]))
                #     elif equalLines(linei1,linej2) and equalLines(linej1,linei2) : 
                #         dist1 = math.dist(intersect[i1], intersect[j2])
                #         dist2 = math.dist(intersect[j1], intersect[i2])
                #         if dist1 > (1/16 * HEIGHT) and dist2 > (1/16 * HEIGHT) :
                #             boxes.append((intersect[i1], intersect[j1], intersect[i2], intersect[j2]))
            if (iteration % 1000 == 0):
                print("Iteration: ", iteration)
            iteration+=1
    return np.asarray(boxes)



def equalLines(line1, line2) :
    x1,y1,x2,y2 = line1
    x3,y3,x4,y4 = line2

    if x1 == x2 :
        slope1 = 9999 # vertical line
    else :
        slope1 = (y1 - y2) / (x1 - x2)
    
    if x3 == x4 :
        slope2 = 9999 # vertical line
    else :
        slope2 = (y3 - y4) / (x3 - x4)
    
    if slope1 == slope2 and slope1 == 9999 and x1 == x3 :
        return True

    b1 = y1 - slope1*x1
    b2 = y3 - slope2*x3

    if slope1 == slope2 and b1 == b2 :
        return True
    else :
        return False


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
    pic = buf[0]
    gimg = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    cap.release()

    
    #SKEWED PIANO JPEG
    
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
    bw_8 =  bw.astype('uint8')
    #rect_b = black_rect(bw)
    #rect_w = white_rect(rect_b)
    
    #p = (WIDTH * HEIGHT - np.sum(np.sum(rect_w))) / (WIDTH * HEIGHT)
    #print(p)
    
    lines = get_lines(bw_8,.1,.9)
    lines1, lines2, intersections, N = get_intersections(lines)
    boxes = find_boxes(lines1, lines2, intersections)
    
    print(N)
    print("plotting results")
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(pic, (x1,y1), (x2,y2), (255, 0, 0), 3)
    for intersection in intersections:
        pic[int(intersection[0]), int(intersection[1]), 0] = 0
        pic[int(intersection[0]-1), int(intersection[1]), 0] = 0
        pic[int(intersection[0]+1), int(intersection[1]), 0] = 0
        pic[int(intersection[0]), int(intersection[1]-1), 0] = 0
        pic[int(intersection[0]), int(intersection[1]+1), 0] = 0
        
        pic[int(intersection[0]), int(intersection[1]), 1] = 0
        pic[int(intersection[0]-1), int(intersection[1]), 1] = 0
        pic[int(intersection[0]+1), int(intersection[1]), 1] = 0
        pic[int(intersection[0]), int(intersection[1]-1), 1] = 0
        pic[int(intersection[0]), int(intersection[1]+1), 1] = 0
        
        pic[int(intersection[0]), int(intersection[1]), 2] = 255
        pic[int(intersection[0]-1), int(intersection[1]), 2] = 255
        pic[int(intersection[0]+1), int(intersection[1]), 2] = 255
        pic[int(intersection[0]), int(intersection[1]-1), 2] = 255
        pic[int(intersection[0]), int(intersection[1]+1), 2] = 255
        
    
    #print("superimposing")
    #for i in range(0,HEIGHT):
        #for j in range(0,WIDTH):
            #if cont[i,j] == 0:
                #gi_norm[i,j] = 0
    
    cv2.imshow("RESULT", pic)


if __name__ == "__main__":
    do_stuff()
