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

'''
in:
    lines - N x 1 X 2 list of lines in our image, (rho, theta)
out: 
    line1 - N x 4 numpyarray that is line 1 to make the intersetction, (x1,y1,x2,y2)
    line2 - N X 4 numpyarray that is line 2 to make the intersetction, (x1,y1,x2,y2)
    intersect - N x 2 numpyarry that is the set of unique intersection points of line 1 and line 2 (x, y)
'''
def find_intersections(lines):
    N = np.size(lines, 0)
    #hvlines = np.zeros((N, 4))
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
    #pairs = itertools.combinations(list(range(0,i-1)), 2)
    n = num_lines #(list(range(0,num_lines)))
    r = 2
    num_comb = math.factorial(n) / (math.factorial(n-r) * math.factorial(r))
    print(num_comb)
    intersections_pad = np.zeros((num_lines,2))
    lines1_pad = np.zeros((num_lines,4))
    lines2_pad = np.zeros((num_lines,4))
    num_inter = 0
    for i in range(0,num_lines):
        for j in range(i+1,num_lines):
            A1 = hvlines[i][3] - hvlines[i][1]
            #print(A1)
            B1 = hvlines[i][2] - hvlines[i][0]
            C1 = A1 * hvlines[i][0] + B1 * hvlines[i][1]
            A2 = hvlines[j][3] - hvlines[j][1]
            B2 = hvlines[j][2] - hvlines[j][0]
            C2 = A2 * hvlines[j][0] + B2 * hvlines[j][1]
            det = A1*B2 - A2*B1
            if det == 0:
                #lines parallel
                continue
            else:
                x = (B2*C1 - B1*C2) / det
                xp = round(x,0)
                y = (A1*C2 - A2*C1) / det
                yp = round(y,0)
                if xp >= 0 and xp < WIDTH and yp >= 0 and yp < HEIGHT :
                    num_inter = num_inter + 1
                    intersections_pad[j-1,0] = x
                    intersections_pad[j-1,1] = y
                    lines1_pad[j-1,0] = hvlines[i][0]
                    lines1_pad[j-1,1] = hvlines[i][1]
                    lines1_pad[j-1,2] = hvlines[i][2]
                    lines1_pad[j-1,3] = hvlines[i][3]
                    lines2_pad[j-1,0] = hvlines[j][0]
                    lines2_pad[j-1,1] = hvlines[j][1]
                    lines2_pad[j-1,2] = hvlines[j][2]
                    lines2_pad[j-1,3] = hvlines[j][3]
    print("shitbag")
    lines1 = lines1_pad[0:num_inter, :]     
    lines2 = lines2_pad[0:num_inter, :]  
    intersections = intersections_pad[0:num_inter,:]  
    return (lines1, lines2, intersections)

'''    
in:
  line1 - N x 4 numpyarray that is line 1 to make the intersetction, (x1,y1,x2,y2)
  line2 - N x 4 numpyarray that is line 2 to make the intersetction, (x1,y1,x2,y2)
  intersect - N x 2 numpyarry that is the intersetion point of line 1 and line 2 (x, y)
out:
  boxes - M x 4 X 2  numpyarray that is a list of M sets of 4 lines that intersect to form a box, (rho,theta)
'''
def find_boxes(line1, line2, intersect) :
    same_line = [] # intersections on the same line, each entry i holds ((intesections), line of intersection, (non intersecting lines))
    boxes = []
    for i in range(len(intersect)) :
        for j in range(i+1, len(intersect)) :
            dist = math.dist(intersect[i], intersect[j])
            if  (dist > (1/16 * HEIGHT)  and equalLines(line2[i], line2[j]) and not equalLines(line1[i],line1[j])) : 
                same_line.append(((i,j), line2[i], (line1[i], line1[j])))  
            elif (dist > (1/16 * HEIGHT) and equalLines(line1[i], line1[j]) and not equalLines(line2[i],line2[j])) : 
                same_line.append(((i,j), line1[i], (line2[i], line2[j])))
            elif (dist > (1/16 * HEIGHT) and equalLines(line1[i], line2[j]) and not equalLines(line2[i],line1[j])) : 
                same_line.append(((i,j), line1[i], (line2[i], line1[j])))
            elif (dist > (1/16 * HEIGHT) and equalLines(line2[i], line1[j]) and not equalLines(line1[i],line2[j])) : 
                same_line.append(((i,j), line2[i], (line1[i], line2[j])))
                
    # print(same_line)            
    # find boxes :)
    for l1 in range(len(same_line)) :
        for l2 in range(l1+1,len(same_line)) :
            if not equalLines(same_line[l1][1],same_line[l2][1]) : # not the same line of intersection
                i1, j1 = same_line[l1][0]
                i2, j2 = same_line[l2][0]
                if i1 != i2 and j1 != j2 and i1 != j2 and j1 != j2 : # no shared intersections
                    linei1, linej1 = same_line[l1][2]
                    linei2, linej2 = same_line[l2][2]
                    if equalLines(linei1, linei2) and equalLines(linej1, linej2) : 
                        dist1 = math.dist(intersect[i1], intersect[i2])
                        dist2 = math.dist(intersect[j1], intersect[j2])
                        if dist1 > (1/16 * HEIGHT) and dist2 > (1/16 * HEIGHT) :
                            boxes.append((intersect[i1], intersect[j1], intersect[i2], intersect[j2]))
                    elif equalLines(linei1,linej2) and equalLines(linej1,linei2) : 
                        dist1 = math.dist(intersect[i1], intersect[j2])
                        dist2 = math.dist(intersect[j1], intersect[i2])
                        if dist1 > (1/16 * HEIGHT) and dist2 > (1/16 * HEIGHT) :
                            boxes.append((intersect[i1], intersect[j1], intersect[i2], intersect[j2]))
    return boxes

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
        slope1 = 9999
    else :
        slope1 = (y1 - y2) / (x1 - x2)
    
    if x3 == x4 :
        slope2 = 9999
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

def test_if_image_is_keyboard(candidate, best_so_far):
    gray_image = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY)
    #blurred_image = cv2.GaussianBlur(gray_image, (7,7,), .2)
    #magic numbers for lower threshold
    _, binary = cv2.threshold(gray_image[20:-20,5:-5],170,255,cv2.THRESH_BINARY)
    # binary = (255-binary)
    # num_labels, labels_im = cv2.connectedComponents(binary, connectivity=4)
    # for i in range(num_labels):
    #     if np.sum(labels_im[labels_im == i]) < 4000:
    #         labels_im[labels_im == i] = 0
    # label_hue = np.uint8(179*labels_im/np.max(labels_im))
    # blank_ch = 255*np.ones_like(label_hue)
    # labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2RGB)
    # labeled_img[label_hue==0] = 0
    # cv2.imshow("w1", labeled_img)
    img = label(binary)
    num_labels = len(np.unique(img))
    for i in range(num_labels):
        if np.count_nonzero(img == i) < 3*270:
            img[img == i] = 0
    img = skimage.color.label2rgb(img, bg_label=0)
    img = skimage.color.rgb2gray(img)
    img[img != 0] = 255
    img = (255-img)
    img = label(img)
    num_labels = len(np.unique(img))
    for i in range(num_labels):
        if np.count_nonzero(img == i) < 3*270:
            img[img == i] = 0

    #magic number
    kernel_width = 13
    for row in range(img.shape[0]):
        for col in range(int(kernel_width/2), img.shape[1] - int(kernel_width/2)):
            if np.count_nonzero(img[row,col-int(kernel_width/2):col+int(kernel_width/2)] == 0) > 6:
                img[row,col] = 0

    # img[img != 0] = 255
    img = skimage.img_as_ubyte(img) 
    cv2.imshow("w1", img)
    c1 = np.mean(binary[0:90,:]) > np.mean(binary[90:,:])
    #c2 = 
    return c1
# in:
#   candidate background image 
# out:
#   background image 
def get_homography_image(frame, boxes):
    # according to google white keys are 0.875x6 in^2 and there are 61 keys on emily's keyboard
    # a black key is supposedly 3.9375 in long
    # intersections of found keyboard
    # for everynoteonefinger frame 0 np.array([[46,374],[49,556],[1245,548],[1243,356]]) - manually chosen intersection points to test homography
    # corners of desired keyboard background
    background = np.array([[0, 0],[0, BG_HEIGHT],[BG_WIDTH, BG_HEIGHT],[BG_WIDTH, 0]])
    H, status = cv2.findHomography(boxes, background)
    best_bg = None
    # for box in boxes:
    #     best_bg = test_if_image_is_keyboard(cv2.warpPerspective(frame, H, (BG_WIDTH, BG_HEIGHT)), best_bg)
    return cv2.warpPerspective(frame, H, (BG_WIDTH, BG_HEIGHT))

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

    
    #4 is magic number, frame without hands in it
    orig = copy.deepcopy(buf[0])
    gi = cv2.cvtColor(buf[0], cv2.COLOR_BGR2GRAY)
    bi = cv2.GaussianBlur(gi, (7,7,), .2)
    #candidate = get_homography_image(orig, np.array([[46,374],[49,556],[1245,548],[1243,356]]))
    candidate = get_homography_image(orig, np.array([[46,374],[49,556],[1245,548],[1243,356]]))
    cv2.imshow("w0.jpg", candidate)
    print(test_if_image_is_keyboard(candidate, None))
    #0, 255 are magic numbers
    icannybelieveit = cv2.Canny(bi, 0, 255, apertureSize=3)

    cap.release()

    #1 and 160 are magic numbers
    lines = cv2.HoughLines(icannybelieveit, 1, np.pi/180, 150)
    cnt = 0
    for line in lines:
        for rho, theta in line:
            delta_v = np.pi/float(32)
            delta_h = np.pi/float(128)
            diff_h = abs(theta - 3*np.pi/2) if theta > np.pi else abs(theta - np.pi/2)
            diff_v = abs(theta - np.pi) if theta > np.pi/2 else abs(theta)
            if diff_h > delta_h and diff_v > delta_v:
                continue
            #print(theta)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 3000*(-b))
            y1 = int(y0 + 3000*(a))
            x2 = int(x0 - 3000*(-b))
            y2 = int(y0 - 3000*(a))
            cv2.line(buf[0], (x1,y1), (x2,y2), (0,0,255),2)
        cnt += 1

    #isabel find_intersections call
    print("area 54")
    lines1, lines2, intersections = find_intersections(lines)
    N = np.size(intersections,0)
    for i in range(0,N):
        orig[intersections[i,0], intersections[i,1], 0] = 255
        orig[intersections[i,0], intersections[i,1], 1] = 0
        orig[intersections[i,0], intersections[i,1], 2] = 0

    cv2.imshow("window", buf[0])

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
