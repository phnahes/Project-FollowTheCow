#!/usr/bin/env python
#-*- encoding: utf-8 -*-

##http://weirdinventionoftheday.blogspot.com.br/2012/10/opencv-background-subtraction-in-python.html

import cv2
import numpy as np

#bgs = cv2.BackgroundSubtractorMOG(24*60, 1, 0.9, 0.01)
bgs = cv2.BackgroundSubtractorMOG(24*60, 9, 0.1, 0.001) # Adjust this values to work better (may put it on a parameter) 

capture = cv2.VideoCapture(0) 

# Windows declarations
cv2.namedWindow("mask")
cv2.namedWindow("cam")
#cv2.namedWindow("blur")
cv2.namedWindow("thr")

while(True):
    
    f, img = capture.read()

    im_blur = cv2.GaussianBlur(img,(9,9),0)
    im_gray = cv2.cvtColor(im_blur,cv2.COLOR_BGR2GRAY) # Convert to Gray
    (thresh1, im_bin) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    fgmask = bgs.apply(im_bin) # Computes a foreground mask.

    # Find and Draw Contours
    ret,thresh2 = cv2.threshold(im_bin,127,255,1)
    contours, hierarchy = cv2.findContours(im_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#    cv2.drawContours(img,contours,-1,(255,0,0),1)

    #len(contours)
    print ("Number of Contours: ", len(contours))

    for h,cnt in enumerate(contours):
      mask = np.zeros(im_gray.shape,np.uint8)
      cv2.drawContours(mask,[cnt],0,255,-1)
      #pixelpoints = np.transpose(np.nonzero(mask))
      mean = cv2.mean(img,mask = mask)

    approx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
    hull = cv2.convexHull(cnt)

    (x,y),radius = cv2.minEnclosingCircle(approx)
    center = (int(x),int(y))
    radius = int(radius)
    cv2.circle(im_gray,center,radius,(0,255,0),2)

    cv2.moveWindow("mask", 1,800)    
    cv2.imshow("mask", fgmask)

    cv2.moveWindow("cam", 1,1)    
    cv2.imshow("cam", img)
    
    cv2.moveWindow("thr", 600,1)    
    cv2.imshow("thr", im_gray)

#    cv2.imshow("blur", im_blur)

    cv2.waitKey(1)
#    cv2.imwrite("./pngs/image-"+str(a).zfill(5)+".png", fgmask)

