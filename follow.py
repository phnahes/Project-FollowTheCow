#!/usr/bin/env python
#-*- encoding: utf-8 -*-

##http://weirdinventionoftheday.blogspot.com.br/2012/10/opencv-background-subtraction-in-python.html

import cv2
import numpy

#bgs = cv2.BackgroundSubtractorMOG(24*60, 1, 0.9, 0.01)
bgs = cv2.BackgroundSubtractorMOG(24*60, 2, 0.3, 0.01)

capture = cv2.VideoCapture(0)
cv2.namedWindow("mask")
cv2.namedWindow("cam")
#cv2.namedWindow("blur")

while(True):
    
    f, img = capture.read()

    im_blur = cv2.GaussianBlur(img,(9,9),0)
    im_gray = cv2.cvtColor(im_blur,cv2.COLOR_BGR2GRAY) # Convert to Gray
    (thresh, im_bin) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    fgmask = bgs.apply(im_bin) # Computes a foreground mask.
    
    cv2.imshow("mask", fgmask)
    cv2.imshow("cam", img)
#    cv2.imshow("blur", im_blur)
    cv2.waitKey(1)
#    cv2.imwrite("./pngs/image-"+str(a).zfill(5)+".png", fgmask)

