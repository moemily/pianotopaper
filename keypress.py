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
BG_WIDTH = 1368
BG_HEIGHT = 270

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
    
    norm = illumination_normalization(bg_gray, frame_gray, 50) # magic number
    pos = norm - bg_gray #black keys
    pos = cv2.GaussianBlur(pos, (15,15,), .2) #magic number kernals
    thresh = 200 #also magic number thresholds
    posbin = pos 
    posbin[pos>=thresh] = 255
    posbin[pos<thresh] = 0

    norm = illumination_normalization(bg_gray, frame_gray, 120) # magic number
    neg = bg_gray - norm #white keys
    neg = cv2.GaussianBlur(neg, (15,15,), .2)
    thresh = 128
    negbin = neg
    negbin[neg >= thresh] = 255
    negbin[neg < thresh] = 0

    return posbin,negbin

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


'''
in:
    img: binary image of hands and key press (positive difference image), black and white image
out: 
    blob: image filled with 1s where kernel sized white blobs were found
'''
def blobs(img):
    print("in blobs")
 
    k_size = 5 # magic number kernel, minimum acceptable finger blob size
    k = int(k_size/2)
    img_pad = np.zeros((BG_HEIGHT + k_size - 1, BG_WIDTH + k_size - 1), dtype = np.uint8)
    img_pad[k:BG_HEIGHT+k, k:BG_WIDTH+k] = img

    blob = np.zeros((BG_HEIGHT, BG_WIDTH), dtype = np.uint8)
    for i in range(k,BG_HEIGHT+k) :
        for j in range(k,BG_WIDTH+k) :
            kernel = img_pad[i-k:i+k+1, j-k:j+k+1]
            value = np.sum(np.sum(kernel)) / (k_size*k_size)
            if value == 255: # if entire kernel is white
                white_blob = 1
            else:
                white_blob = 0
            blob[i-k,j-k] = white_blob
    return blob


'''
in:
    blob: image filled with 1s where kernel sized white blobs were found
    k_size: size of kernel used in blobs()
out:
    left: x coordinate of left-most hand location 
    right: x coordinate of right-most hand location
'''
def hand_edges(blobs, k_size) :
    print("in hand_edges")
    left = BG_WIDTH
    right = 0
    for i in range(BG_HEIGHT) :
        for j in range(BG_WIDTH) :
            if blobs[i,j] == 1 : #img[height,width]
                if j < left :
                    left = j
                if j > right :
                    right = j

    left = left - int(k_size/2)
    right = right + int(k_size/2)
    return left, right


def skinny_boi(img, left, right) :
    print("in skinny_boi")
    pressed_j = np.zeros(BG_WIDTH)
    press_coord = []

    height_crop = int(BG_HEIGHT*3/5) # magic number of how much to crop
    pressarea = img[0:height_crop,left:right] 
    
    kern_h = 11 # height of line
    k_h = int((kern_h) / 2)
    
    # tall_kern_w = 3 
    # tall_k = int(tall_kern_w/2)
    
    fat_kern_w = 69 
    fat_k = int((fat_kern_w) / 2)
    
    search_press = np.zeros((height_crop + kern_h - 1, (right-left) + fat_kern_w - 1), dtype = np.uint8)
    search_press[k_h:height_crop+k_h, fat_k:(right-left)+fat_k] = pressarea
    
    for j in range(0, right-left) :
        for i in range(height_crop) :
            # tall_kernel = pressarea[i-k_h:i+k_h, j-tall_k:j+tall_k]
            tall_kernel = pressarea[i-k_h:i+k_h, j]
            tallness = np.sum(np.sum(tall_kernel)) 

            fat_kernel_left = pressarea[i-k_h:i+k_h, j-fat_k:j]
            fat_kernel_right = pressarea[i-k_h:i+k_h, j:j+fat_k]
            fatness_left = np.sum(np.sum(fat_kernel_left)) 
            fatness_right = np.sum(np.sum(fat_kernel_right))

            if tallness > (kern_h*255)*0.5 :  #tallness > (tall_kern_w*kern_h*255)*0.5 :
                # print("j ", j, "fatty: ", fatness)
                if fatness_left < (fat_k*kern_h*255)*0.1 and fatness_right < (fat_k*kern_h*255)*0.1: # tall kernel > 50% white and fat kernel < 10 white
                # then this is a good x coordinate aka j, skinny and tall line
                    pressed_j[j] = 1
                    # print(j)
                    break
            # if fatness < (fat_kern_w*kern_h*255)*0.1:
            #     print("j ", j, "tall: ", tallness)
            #     if tallness > (tall_kern_w*kern_h*255)*0.5 : # tall kernel > 50% white and fat kernel < 10 white
            #     # then this is a good x coordinate aka j, skinny and tall line
            #         pressed_j[j] = 1
            #         # print(j)
            #         break

    # find center x coordinate of each press
    j = left
    while j < right :
        if pressed_j[j] == 1 :
            i = j + 1
            while pressed_j[i] == 1 and i < right :
                i = i + 1
            press_center = int((i-j)/2) + j + left - 10
            press_coord.append(press_center) 
            j = i 
        j = j + 1  

    return press_coord

def do_stuff():

    cap = cv2.VideoCapture("IMG_8040.MOV")
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

    boxes = np.array([[190,390],[97,650],[1865,975],[1800,750]]) #for 8040
    bg = get_homography_image(buf[0], boxes)
    f = 330
    frame = get_homography_image(buf[f], boxes)
    # cv2.imshow("frame", frame)
    # cv2.imwrite("frame.jpg", frame)

    posbin, negbin = bg_subtraction(bg, frame)

    # cv2.imshow("positive", posbin)
    # cv2.imshow("negative", negbin)

    blobs_found = blobs(posbin)
    left, right = hand_edges(blobs_found, k_size=5)
    
    # print("left: ", left)
    # print("right: ", right)
    # left = 216; right = 920
    height_crop = int(BG_HEIGHT*3/5)
    pressarea = posbin[0:height_crop,left:right] 
    cv2.imshow("cropped", pressarea)

    coord = skinny_boi(posbin, left, right)
    print("coords: ", coord)

    for x in coord :
        # Draw a diagonal blue line with thickness of 5 px
        posbin = cv2.line(posbin,(x,0),(x,BG_HEIGHT),(255,0,0),1)
        frame = cv2.line(frame,(x,0),(x,BG_HEIGHT),(255,0,0),1)
    
    cv2.imshow("posbin", posbin)
    cv2.imshow("frame", frame)

    key = cv2.waitKey(0)
    while (key != ord('q')) :
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') :
            cv2.destroyAllWindows()
            continue


if __name__ == "__main__":
    do_stuff()


