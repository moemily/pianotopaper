# -*- coding: utf-8 -*-
import numpy as np
import cv2
import sys, os 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import itertools
import math
import copy

WIDTH = 1280
HEIGHT = 720
BG_WIDTH = 1342
BG_HEIGHT = 270

#in:
#lines - N x 1 X 2 list of lines in our image, (rho, theta)
#out: 
#line1 - N x 4 numpyarray that is line 1 to make the intersetction, (x1,y1,x2,y2)
#line2 - N X 4 numpyarray that is line 2 to make the intersetction, (x1,y1,x2,y2)
#intersect - N x 2 numpyarry that is the set of unique intersection points of line 1 and line 2 (x, y)
def find_intersections(lines):
    cnt = 0
    N = np.size(lines, 0)
    hvlines = np.zeros((N, 4))
    for line in lines:
        for rho, theta in line:
            i = 0
            delta_v = np.pi/float(32)
            delta_h = np.pi/float(128)
            diff_h = abs(theta - 3*np.pi/2) if theta > np.pi else abs(theta - np.pi/2)
            diff_v = abs(theta - np.pi) if theta > np.pi/2 else abs(theta)
            if diff_h > delta_h and diff_v > delta_v:
                i = i + 1
                continue
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            m = max(WIDTH, HEIGHT)
            x1 = int(x0 + m*(-b))
            y1 = int(y0 + m*(a))
            x2 = int(x0 - m*(-b))
            y2 = int(y0 - m*(a))
            hvlines[i-1, 0] = x1
            hvlines[i-1, 1] = y1
            hvlines[i-1, 2] = x2
            hvlines[i-1, 3] = y2
    pairs = itertools.combinations(list(range(0,i-1)), 2)
    print("here")
    intersections_pad = np.zeros((i,2))
    lines1_pad = np.zeros((i,4))
    lines2_pad = np.zeros((i,4))
    j = 0
    for pair in pairs:
        A1 = hvlines[pair[0],3] - hvlines[pair[0],1]
        B1 = hvlines[pair[0],2] - hvlines[pair[0],0]
        #print(A1)
        A2 = hvlines[pair[1],3] - hvlines[pair[1],1]
        B2 = hvlines[pair[1],2] - hvlines[pair[1],0]
        C2 = A1 * hvlines[pair[1],0] + B1 * hvlines[pair[1],1]
        det = A1*B2 - A2*B1
        if det == 0:
            #lines parallel
            continue
        else:
            x = (B2*C1 - B1*C2) / det
            xp = round(x,0)
            y = (A1*C2 - A2*C1) / det
            yp = round(y,0)
            if xp >= 0 and xp <= WIDTH and yp >= 0 and yp <= HEIGHT :
                j = j + 1
                intersections_pad[j-1,0] = x
                intersections_pad[j-1,1] = y
                lines1_pad[j-1,0] = hvlines[pair[0], 0]
                lines1_pad[j-1,1] = hvlines[pair[0], 1]
                lines1_pad[j-1,2] = hvlines[pair[0], 2]
                lines1_pad[j-1,3] = hvlines[pair[0], 3]
                lines2_pad[j-1,0] = hvlines[pair[1], 0]
                lines2_pad[j-1,1] = hvlines[pair[1], 1]
                lines2_pad[j-1,2] = hvlines[pair[1], 2]
                lines2_pad[j-1,3] = hvlines[pair[1], 3]
    lines1 = lines1_pad[0:j, :]     
    lines2 = lines2_pad[0:j, :]  
    intersections = intersections_pad[0:j,:]  
    return (lines1, lines2, intersections)
    
# in:
#   line1 - 1 x 4 numpyarray (x1,y1,x2,y2)
#   line2 - 1 x 4 numpyarray (x1,y1,x2,y2)
# in:
#   line1 - N x 4 numpyarray that is line 1 to make the intersetction, (x1,y1,x2,y2)
#   line2 - N x 4 numpyarray that is line 2 to make the intersetction, (x1,y1,x2,y2)
#   intersect - N x 2 numpyarry that is the intersetion point of line 1 and line 2 (x, y)
# out:
#   boxes - M x 4 X 2  numpyarray that is a list of M sets of 4 lines that intersect to form a box, (rho,theta)
def find_boxes(line1, line2, intersect) :
    sameLine = []
    boxes = []
    for i in range(len(intersect)) :
        for j in range(i, len(intersect)) :
            dist = math.dist(intersect[i], intersect[j])
            # finds intersections on the same line
            if (dist > (1/16 * HEIGHT) and not equalLines(line1[i],line1[j]) and not equalLines(line2[i],line2[j])) : 
                sameLine.append(((i,j), line2[i], line1[i], line1[j]))  # intesects, line of intersection, non intersecting lines
            elif (dist > (1/16 * HEIGHT) and equalLines(line1[i], line1[j]) and not equalLines(line2[i],line2[j])) : 
                sameLine.append(((i,j), line1[i], line2[i], line2[j]))
                
    print(sameLine)            
    # find boxes :)
    for l1 in range(len(sameLine)) :
        for l2 in range(l1,len(sameLine)) :
            if not equalLines(sameLine[l1][1],sameLine[l2][1]) : # not the same line of intersection
                i1, j1 = sameLine[l1][0]
                i2, j2 = sameLine[l2][0]
                if i1 != i2 and j1 != j2 and i1 != j2 and j1 != j2 : # no shared intersections
                    if equalLines(sameLine[l1][2],sameLine[l2][2]) and equalLines(sameLine[l1][3],sameLine[l2][3]) : 
                        dist1 = math.dist(intersect[i1], intersect[i2])
                        dist2 = math.dist(intersect[j1], intersect[j2])
                        if dist1 > (1/16 * HEIGHT) and dist2 > (1/16 * HEIGHT) :
                            boxes.append((intersect[i1], intersect[j1], intersect[i2], intersect[j2]))
                    elif equalLines(sameLine[l1][2],sameLine[l2][3]) and equalLines(sameLine[l1][3],sameLine[l2][2]) : 
                        dist1 = math.dist(intersect[i1], intersect[j2])
                        dist2 = math.dist(intersect[j1], intersect[i2])
                        if dist1 > (1/16 * HEIGHT) and dist2 > (1/16 * HEIGHT) :
                            boxes.append((intersect[i1], intersect[j1], intersect[i2], intersect[j2]))
    return boxes# out:
#   equal - bool, whether or not points lie on the same line
def equalLines(line1, line2) :
    def equalLines(line1, line2) :
        l1x1, l1y1, l1x2, l1y2 = line1
        l2x1, l2y1, l2x2, l2y2 = line2

    if l1x1 == l1x2 :
        slope1 = 9999
    else :
        slope1 = (l1y1 - l1y2) / (l1x1 - l1x2)
    
    if l2x1 == l2x2 :
        slope2 = 9999
    else :
        slope2 = (l2y1 - l2y2) / (l2x1 - l2x2)
    
    if slope1 == slope2 and slope1 == 9999 and l1x1 == l2x2 :
        return True

    b1 = l1y1 - slope1*l1x1
    b2 = l2y1 - slope2*l2x1

    if slope1 - slope2 < 0.01 and b1 - b2 < 0.01 :
        return True
    else :
        return False

def test_if_image_is_keyboard(candidate, best_so_far):
    gray_image = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY)
    binary = cv.threshold(gray_image,127,255,cv.THRESH_BINARY)
    c1 = np.mean(binary[0:90,:]) > np.mean(binary[90:,:])
    #c2 = 

# in:
#   candidate background image 
# out:
#   background image 
def get_homography_image(frame, boxes):
    # according to google white keys are 0.875x6 in^2 and there are 61 keys on emily's keyboard
    # a black key is supposedly 3.9375 in long
    # intersections of found keyboard
    #p = np.array([[57,381],[35,639],[1369,644],[1350,378]]) - manually chosen intersection points to test homography
    # corners of desired keyboard background
    background = np.array([[0, 0],[0, BG_HEIGHT],[BG_WIDTH, BG_HEIGHT],[BG_WIDTH, 0]])
    H, status = cv2.findHomography(boxes, background)
    best_bg = None
    for box in boxes:
        best_bg = test_if_image_is_keyboard(cv2.warpPerspective(frame, H, (BG_WIDTH, BG_HEIGHT)), best_bg)
    return cv2.warpPerspective(frame, H, (BG_WIDTH, BG_HEIGHT))

def do_stuff():
    cap = cv2.VideoCapture("00006.mp4")
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
    #4 is magic number, frame without hands in it
    orig = copy.deepcopy(buf[4])
    gi = cv2.cvtColor(buf[4], cv2.COLOR_BGR2GRAY)
    bi = cv2.GaussianBlur(gi, (7,7,), .2)
    
    #0, 255 are magic numbers
    icannybelieveit = cv2.Canny(bi, 0, 255, apertureSize=3)


    cap.release()

    #1 and 160 are magic numbers
    lines = cv2.HoughLines(icannybelieveit, 1, np.pi/180, 160)
    cnt = 0
    for line in lines:
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

    #isabel find_intersections call
    print("area 54")
    lines1, lines2, intersections = find_intersections(lines)
    N = np.size(intersections,0)
    for i in range(0,N):
        orig[intersections[i,0], intersections[i,1], 0] = 255
        orig[intersections[i,0], intersections[i,1], 1] = 0
        orig[intersections[i,0], intersections[i,1], 2] = 0

    cv2.imshow("window", orig)

    #emily stuff
    # cv2.waitKey(0)
    # while (1) :
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord('q') :
    #         cv2.destroyAllWindows()
    #         exit()

    #cv2.imwrite('houghlines.jpg',img)

if __name__ == "__main__":
    do_stuff()
