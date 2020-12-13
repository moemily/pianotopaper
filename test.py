#-*- coding: utf-8 -*-
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
from PIL import Image

WIDTH = 1280
HEIGHT = 720
BG_WIDTH = 1368
BG_HEIGHT = 270
BG_IMAGE = None
BG_SEG = None
BG_TEMP = None
BG_MEAN = 0
BG_KEYS = 0
CROP_AMOUNT = 5


def show_lines(img, lines):
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
            x1 = int(x0 + 200*(-b))
            y1 = int(y0 + 200*(a))
            x2 = int(x0 - 200*(-b))
            y2 = int(y0 - 200*(a))
            cv2.line(img, (x1,y1), (x2,y2), (0,0,255),2)

def show_intersections(img, intersections, N):
    for i in range(0, N):
        img[int(intersections[i][0]), int(intersections[i][1]), 0] = 0
        img[int(intersections[i][0]), int(intersections[i][1]), 1] = 255
        img[int(intersections[i][0]), int(intersections[i][1]), 2] = 0
    cv2.imshow("intersections", img)

'''
in:
    lines - N x 1 X 2 list of lines in our image, (rho, theta)
out: 
    line1 - N x 4 list that is line 1 to make the intersetction, (x1,y1,x2,y2)
    line2 - N X 4 list that is line 2 to make the intersetction, (x1,y1,x2,y2)
    intersect - N x 2 list that is the set of unique intersection points of line 1 and line 2 (x, y)
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

    # want map of intersections -> line and line -> intersection 


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

'''
in:
    bg: background frame
    frame: current frame (hands)
out: 
    posbin: binary image of positive parts of norm - bg
    negbin: binary image of negative parts of norm - bg
'''
def bg_subtraction(bg, frame) :
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    norm = illumination_normalization(bg_gray, frame_gray, 100) #100 is magic number
    
    pos = norm - bg_gray #black keys
    pos = cv2.GaussianBlur(pos, (15,15,), .2) #magic number kernals
    thresh = 200 #also magic number thresholds
    posbin = pos 
    posbin[pos>=thresh] = 255
    posbin[pos<thresh] = 0

    neg = bg_gray - norm #white keys
    neg = cv2.GaussianBlur(neg, (15,15,), .2)
    thresh = 70
    negbin = neg
    negbin[neg >= thresh] = 255
    negbin[neg < thresh] = 0

    return posbin,negbin

    
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
        return 0, 0, 0
    #magic numbers 170/255
    #magic number crop amount 
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
        return 0, 0, 0

    if cond1 and cond2: 
        seg_img = np.zeros((img.shape[0]+4*CROP_AMOUNT,img.shape[1]+2*CROP_AMOUNT)).astype('uint8')
        seg_img[2*CROP_AMOUNT:-2*CROP_AMOUNT,CROP_AMOUNT:-CROP_AMOUNT] = img
        return brightness_bottom, num_keys, seg_img
    return 0, 0, 0

# IN:
#   takes image and coorindates of a box 
# OUT:
#   returns warped image with dimensions BG_{WIDTH, HEIGHT} from image
#currently depends on TL - BL - BR - TR order of points
def get_homography_image(frame, box):
    background = np.array([[0, 0],[0, BG_HEIGHT],[BG_WIDTH, BG_HEIGHT],[BG_WIDTH, 0]])
    homography, status = cv2.findHomography(box, background)
    return cv2.warpPerspective(frame, homography, (BG_WIDTH, BG_HEIGHT))

def sort_coords(sub_list):
    sub_list.sort(key = lambda x: x[1])

#sorts into tl bl br tr
def sort_box_coordinates(box):   
    x1 = None
    x2 = None
    cur_min_x = BG_WIDTH
    sec_min_x = BG_WIDTH
    for i in range(4):
        if box[i][0] < cur_min_x:
            sec_min_x = cur_min_x
            x1 = x2
            cur_min_x = box[i][0]
            x2 = i
        elif box[i][0] < sec_min_x:
            sec_min_x = box[i][0]
            x1 = i
    one = None
    two = None
    three = None
    four = None
    indicies = [0, 1, 2, 3]
    print(indicies)
    if (box[x1][1] < box[x2][1]):
        one = x1
        two = x2
    else:
        one = x2
        two = x1
    indicies.remove(one)
    indicies.remove(two)
    print(box.shape)
    if box[indicies[0]][1] > box[indicies[1]][1]:
        three = indicies[0]
        four = indicies[1]
    else:
        three = indicies[1]
        four = indicies[0]
    
    sorted_box = np.zeros_like(box)
    sorted_box[0] = box[one]
    sorted_box[1] = box[two]
    sorted_box[2] = box[three]
    sorted_box[3] = box[four]
    return sorted_box

# in:
#   candidate background image and array of potential boxes
# out:
#   bool whether it updated BG_IMAGE or not indicating, essentially, whether BG_IMAGE is valid or not
def get_background_image(frame, boxes):
    global BG_IMAGE, BG_MEAN, BG_KEYS, BG_SEG
    updated_background = False
    for box in boxes:
        sorted_box = sort_box_coordinates(box)
        print("box:", box)
        print("sorted_box: ", sorted_box)
        candidate = get_homography_image(frame, sort_box_coordinates(box))
        test_mean, test_keys, test_seg = test_if_image_is_keyboard(candidate)
        if test_mean > BG_MEAN and test_keys >= BG_KEYS:
            BG_MEAN = test_mean
            BG_KEYS = test_keys
            BG_IMAGE = candidate
            BG_SEG = test_seg
            updated_background = True
            print(box)
    return updated_background

def find_lines(frame) :
    grey = cv2.cvtColor(frame[300:,:], cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (7,7,), .2) #magic number

    # Canny Edge Detection 
    icannybelieveit = cv2.Canny(blurred, 0, 255, apertureSize=3) #magic number
    # Hough Line Detection  (1 and 150 are magic numbers)
    lines = cv2.HoughLines(icannybelieveit, 1, 1*np.pi/180, 100) #magic number
    # show_lines(frame, lines)

    # #magic numbers galore
    # blur = cv2.GaussianBlur(grey, (1,1),0)
    # _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # mask = np.ones(grey.shape[:2], dtype="uint8")*255
    # for c in contours:
    #     x, y, w, h = cv2.boundingRect(c)
    #     if w*h > 4000:
    #         cv2.rectangle(mask, (x,y), (x+w, y+h), (0,0,255), -1)
    # cv2.imshow("gray", grey)
    # cv2.imshow("boxes", mask)

    return lines

def clean_segmentation(img):
    start = None
    end = None
    for i in range(1,img.shape[1]):
        if not img[2*CROP_AMOUNT,i-1] and img[2*CROP_AMOUNT,i]:
            start = i
        if img[2*CROP_AMOUNT,i-1] and not img[2*CROP_AMOUNT,i]:
            end = i
        if start and end:
            img[:2*CROP_AMOUNT,start:end] = img[2*CROP_AMOUNT+1, start+1]
            start = None
            end = None
    width = int(img.shape[1] / 36)
    for i in range(36):
        key = np.array(img[:,i*width:i*width + width])
        key[img[:,i*width:i*width + width] == 0] = BG_KEYS + i + 1
        img[:,i*width:i*width + width] = key
        # show lines for white keys
        # img[:,i*width+width:i*width+width+1] = 255
    return img

def get_keys(coords):
    global BG_SEG
    key_list = []
    for i in coords:
        key_list.append(BG_SEG[0,i])
    return key_list

def do_stuff():
    cap = cv2.VideoCapture("videos/mov.mp4")
    #cap = cv2.VideoCapture("videos/EveryNoteOneFinger.mp4")
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

    # mycopy = copy.deepcopy(buf[0]) #magic number
    # mycopy = mycopy[320:600,:] #magic number (cropped example)
    # orig = copy.deepcopy(mycopy)
    # cv2.imshow("cropped", mycopy)
    # print("Getting lines")
    # lines = find_lines(orig)
    # print("Calculating Intersections")
    # lines1, lines2, intersect, N = find_intersections(lines)
    # print("intersections is N is ", N)
    # show_lines_and_intersections(mycopy, lines, lines1, lines2, intersect, N)
    # print("Generating boxes")
    # boxes = find_boxes(lines1, lines2, intersect)
    # print("Getting Background Image")
    # success = get_background_image(orig, boxes)
    # print("Was a success: ", success)
    # cv2.imshow("background", BG_IMAGE) 

    orig = copy.deepcopy(buf[0])
    cv2.imwrite("bgcheck.jpg", orig)
    #this one is for the everynote finger video
    #boxes = np.array([[[46,374],[49,556],[1245,548],[1243,356]],[[16,3544],[19,526],[1215,518],[1213,326]]])


    # this one works
    # boxes = np.array([[[130,257],[70,437],[1240,646],[1196,497]],[[16,3544],[19,526],[1215,518],[1213,326]]])
    boxes1 = np.array([[[130,257],[70,437],[1240,646],[1196,497]],[[16,3544],[19,526],[1215,518],[1213,326]]])
    print(boxes1[0])
    print("^ correct")
    boxes = np.array([[[70,437],[1240,646],[130,257],[1196,497]],[[16,3544],[19,526],[1215,518],[1213,326]]])
    print(sort_box_coordinates(boxes[0]))
    print("^ sorted")
    cv2.imshow("homography", get_homography_image(orig, sort_box_coordinates(boxes[0])))
    bg = get_background_image(orig, boxes)
    if (bg == False):
        print("Was not able to find background from the boxes provided")
    #bg = get_background_image(orig, boxes)
    #box = np.array([[[46,374],[49,556],[1245,548],[1243,356]],[[16,3544],[19,526],[1215,518],[1213,326]]])
    #boxes = np.array([[497,1196],[646,1240],[437,70],[257,130]])
    #bg = get_background_image(orig, box)
    #sort_box_coordinates(boxes[0])
    print("we here")
    lines = find_lines(orig)
    show_lines(orig, lines)
    _, _, intersections, N = find_intersections(lines)
    show_intersections(orig, intersections, N)
    cv2.imshow("background", BG_IMAGE)
    global BG_SEG
    BG_SEG = clean_segmentation(BG_SEG)
    #BG_SEG[BG_SEG == 26] = 255
    cv2.imshow("segmentation", BG_SEG)
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
