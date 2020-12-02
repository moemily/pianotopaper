# -*- coding: utf-8 -*-
import numpy as np
import cv2
import sys, os 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import itertools
import math
import copy
from skimage import segmentation
from skimage import io, color
import skimage
from skimage.measure import label

WIDTH = 1280
HEIGHT = 720
BG_WIDTH = 1342
BG_HEIGHT = 270
BG_IMAGE = None
BG_MEAN = 0
BG_KEYS = 0

'''
in:
    lines - N x 1 X 2 list of lines in our image, (rho, theta)
out: 
    line1 - N x 4 numpyarray that is line 1 to make the intersetction, (x1,y1,x2,y2)
    line2 - N X 4 numpyarray that is line 2 to make the intersetction, (x1,y1,x2,y2)
    intersect - N x 2 numpyarry that is the set of unique intersection points of line 1 and line 2 (x, y)
'''
def find_intersections(lines):
    hvlines = []
    num_lines = 0
    for line in lines:
        for rho, theta in line:
            delta_v = np.pi/float(32)
            delta_h = np.pi/float(128)
            diff_h = abs(theta - 3*np.pi/2) if theta > np.pi else abs(theta - np.pi/2)
            diff_v = abs(theta - np.pi) if theta > np.pi/2 else abs(theta)
            if diff_h > delta_h and diff_v > delta_v:
                continue
            num_lines = num_lines + 1
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            m = max(WIDTH, HEIGHT)
            x1 = int(x0 + m*(-b))
            y1 = int(y0 + m*(a))
            x2 = int(x0 - m*(-b))
            y2 = int(y0 - m*(a))
            hvlines.append([x1,y1,x2,y2])
    intersections = []
    lines1 = []
    lines2 = []
    num_inter = 0
    for i in range(0,num_lines):
        for j in range(i+1,num_lines):
            x11 = hvlines[i][0]
            y11f = hvlines[i][1]
            y11 = -y11f
            x12 = hvlines[i][2]
            y12f = hvlines[i][3]
            y12 = -y12f
            
            x21 = hvlines[j][0]
            y21f = hvlines[j][1]
            y21  = -y21f
            x22 = hvlines[j][2]
            y22f = hvlines[j][3]
            y22 = -y22f
    
            A1 = y12 - y11
            B1 = x11 - x12
            C1 = A1 * x11 + B1 * y11
            A2 = y22 - y21
            B2 = x21 - x22
            C2 = A2 * x21 + B2 * y21
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
                    lines1.append([hvlines[i][0],hvlines[i][1],hvlines[i][2],hvlines[i][3]])
                    lines2.append([hvlines[j][0],hvlines[j][1],hvlines[j][2],hvlines[j][3]])
    return (lines1, lines2, intersections, num_inter)

'''    
in:
  line1 - N x 4 numpyarray that is line 1 to make the intersection, (x1,y1,x2,y2)
  line2 - N x 4 numpyarray that is line 2 to make the intersection, (x1,y1,x2,y2)
  intersect - N x 2 numpyarry that is the intersetion point of line 1 and line 2 (x, y)
out:
  boxes - M x 4 X 2  numpyarray that is a list of M sets of 4 lines that intersect to form a box, (x,y)
'''
def find_boxes(line1, line2, intersect) :
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

'''
in:
  line1 - 1 x 4 numpyarray (x1,y1,x2,y2)
  line2 - 1 x 4 numpyarray (x3,y3,x4,y4)
out:
  equal - bool, whether or not all four points lie on the same line
'''
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

'''
in:
    bg: background frame
    frame: current frame (hands)
    step: 0-255, 255 = bg, 0 = frame
out: 
    norm: image normalized to bg or frame lol
'''
def illumination_normalization(bg, frame, step=255) :
    norm = np.zeros(bg.shape)
    norm = frame + np.minimum(abs(bg - frame),step) * np.sign(bg - frame)
    return norm

index = 0
# IN:
#   takes a candidate background image and checks if it's a keyboard
# OUT:
#   returns information for appropriately updating BG image if applicable
#   returns mean, num_keys 
def test_if_image_is_keyboard(candidate):
    gray_image = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (7,7,), .2)
    
    #magic numbers ahoy below
    brightness_bottom = np.mean(blurred_image[int(BG_HEIGHT/3 * 2):,:])
    brightness_top = np.mean(blurred_image[:int(BG_HEIGHT/3 * 2),:])
    cond1 = brightness_bottom > brightness_top
    if not cond1: 
        #debug information
        #print("Lower third was not brighter than top two thirds")
        return 0, 0
    #magic numbers 170/255
    #magic number crop amount 
    CROP_AMOUNT = 5
    _, binary = cv2.threshold(blurred_image[2*CROP_AMOUNT:-2*CROP_AMOUNT,CROP_AMOUNT:-CROP_AMOUNT],170,255,cv2.THRESH_BINARY)
    img = label(binary)
    num_labels = len(np.unique(img))
    for i in range(num_labels):
        #magic number of 3*BG_HEIGHT
        if np.count_nonzero(img == i) < 3*int(BG_HEIGHT):
            img[img == i] = 0
    img = skimage.color.label2rgb(img, bg_label=0)
    img = skimage.color.rgb2gray(img)
    img[img != 0] = 255
    img = (255-img)
    img = label(img)
    num_labels = len(np.unique(img))
    for i in range(num_labels):
        if np.count_nonzero(img == i) < 3*int(BG_HEIGHT):
            img[img == i] = 0

    #magic number kernel width
    kernel_width = 13
    for row in range(img.shape[0]):
        for col in range(int(kernel_width/2), img.shape[1] - int(kernel_width/2)):
            if np.count_nonzero(img[row,col-int(kernel_width/2):col+int(kernel_width/2)] == 0) > int(kernel_width/2):
                img[row,col] = 0

    # img[img != 0] = 255
    img = skimage.img_as_ubyte(img) 
    num_keys = len(np.unique(img)) - 1

    #this shows the keys / regions clearly 
    # global index
    # img[img != 0] = 255
    # cv2.imshow("segmented image" + str(index), img)
    # index += 1

    #hardcoded but doesn't need to be if we're doing iterative updates like the paper
    cond2 = num_keys == 25
    if not cond2:
        #debug information
        #print("Found ", num_keys, " black keys")
        return 0, 0

    if cond1 and cond2: 
        return brightness_bottom, num_keys
    return 0, 0

# IN:
#   takes image and coorindates of a box 
# OUT:
#   returns warped image with dimensions BG_{WIDTH, HEIGHT} from image
def get_homography_image(frame, box):
    background = np.array([[0, 0],[0, BG_HEIGHT],[BG_WIDTH, BG_HEIGHT],[BG_WIDTH, 0]])
    homography, status = cv2.findHomography(box, background)
    return cv2.warpPerspective(frame, homography, (BG_WIDTH, BG_HEIGHT))

# in:
#   candidate background image and array of potential boxes
# out:
#   bool whether it updated BG_IMAGE or not indicating, essentially, whether BG_IMAGE is valid or not
def get_background_image(frame, boxes):
    global BG_IMAGE, BG_MEAN, BG_KEYS
    best_mean = 0
    best_keys = 0
    updated_background = False
    for box in boxes:
        candidate = get_homography_image(frame, box)
        test_mean, test_keys = test_if_image_is_keyboard(candidate)
        if test_mean > BG_MEAN and test_keys >= BG_KEYS:
            BG_MEAN = test_mean
            BG_KEYS = test_keys
            BG_IMAGE = candidate
            updated_background = True
    return updated_background

def find_lines(frame) :
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (7,7,), .2)

    # Canny Edge Detection 
    icannybelieveit = cv2.Canny(blurred, 0, 255, apertureSize=3)
    
    # Hough Line Detection  (1 and 150 are magic numbers)
    lines = cv2.HoughLines(icannybelieveit, 1, np.pi/180, 150)
    return lines

def show_lines_and_intersections(orig, lines, lines1, lines2, intersections, N):
    cnt = 0

    for line in lines:
        for rho, theta in line:
            delta_v = np.pi/float(32)
            delta_h = np.pi/float(128)
            diff_h = abs(theta - 3*np.pi/2) if theta > np.pi else abs(theta - np.pi/2)
            diff_v = abs(theta - np.pi) if theta > np.pi/2 else abs(theta)
            if diff_h > delta_h and diff_v > delta_v:
                continue
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 3000*(-b))
            y1 = int(y0 + 3000*(a))
            x2 = int(x0 - 3000*(-b))
            y2 = int(y0 - 3000*(a))
            cv2.line(orig, (x1,y1), (x2,y2), (0,0,255),2)
        cnt += 1

    for i in range(0, N):
        orig[int(intersections[i][0]), int(intersections[i][1]), 0] = 0
        orig[int(intersections[i][0]), int(intersections[i][1]), 1] = 255
        orig[int(intersections[i][0]), int(intersections[i][1]), 2] = 0
    cv2.imshow("intersections", orig)

def do_stuff():
    #cap = cv2.VideoCapture("00006.mp4")
    cap = cv2.VideoCapture("videos/EveryNoteOneFinger.mp4")
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    global WIDTH
    global HEIGHT 
    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frame_cnt, HEIGHT, WIDTH, 3), np.dtype('uint8'))
    fc = 0
    ret = True
    while (fc < frame_cnt and ret):
        ret, buf[fc] = cap.read()
        fc += 1

    mycopy = copy.deepcopy(buf[0])
    mycopy = mycopy[320:600,:]
    orig = copy.deepcopy(mycopy)
    cv2.imshow("cropped", mycopy)
    print("Getting lines")
    lines = find_lines(orig)
    print("Calculating Intersections")
    lines1, lines2, intersect, N = find_intersections(lines)
    print("intersections is N is ", N)
    show_lines_and_intersections(mycopy, lines, lines1, lines2, intersect, N)
    print("Generating boxes")
    boxes = find_boxes(lines1, lines2, intersect)
    print("Getting Background Image")
    success = get_background_image(orig, boxes)
    print("Was a success: ", success)
    cv2.imshow("background", BG_IMAGE)

    # orig = copy.deepcopy(buf[0])
    # bg = get_background_image(orig, np.array([[[46,374],[49,556],[1245,548],[1243,356]],[[16,3544],[19,526],[1215,518],[1213,326]]]))

    cap.release()

    #emily stuff
    # key = cv2.waitKey(0)
    # while (key != ord('q')) :
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord('q') :
    #         cv2.destroyAllWindows()
    #         continue

    #cv2.imwrite('houghlines.jpg',img)

if __name__ == "__main__":
    do_stuff()
