#!/usr/bin/env python
#-*- encoding: utf-8 -*-

#http://weirdinventionoftheday.blogspot.com.br/2012/10/opencv-background-subtraction-in-python.html

import cv2
import numpy as np

def CannyThreshold(lowThreshold):
	detected_edges = cv2.GaussianBlur(gray,(3,3),0)
	detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
	dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
	cv2.imshow('canny demo',dst)

bgs = cv2.BackgroundSubtractorMOG(24*60, 2, 0.5, 0.01)

capture = cv2.VideoCapture(0)

#cv2.namedWindow("FGMask")
cv2.namedWindow("Capture")
#cv2.namedWindow("Capture2")
#cv2.namedWindow("ThsContour")

a=0
while(True):

	f, img = capture.read() # Imagem Capturada
#	f, contdraw = capture.read() # Clonando Imagem Capturada -- Deve existir forma melhor de fazer isso
#	fgmask = bgs.apply(img) # Mascara de Plano de frente
#	cont = bgs.apply(contdraw) #Imagem para ser usada com contornos

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	lowThreshold = 0
	max_lowThreshold = 100
	ratio = 3
	kernel_size = 3

	CannyThreshold(80)  # initialization

## Desenha contorno nos objetos encontrados --- Deve ser melhorado
#	ret,thresh = cv2.threshold(cont,127,255,1)
#	contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
#	cv2.drawContours(contdraw,contours,-1,(0,255,0),3)


## Este bloco conta o numero de contornos
##
#	len(contours)
#	cnt = contours[0]
#	print(len(cnt))
#	
#	cv2.drawContours(contdraw,[cnt],0,(255,0,0),3)
#	for h,cnt in enumerate(contours):
#	    mask = np.zeros(cont.shape,np.uint8)
#	    cv2.drawContours(mask,[cnt],0,255,3)
#	    mean = cv2.mean(contdraw,mask = mask)

	#Exibe imagens
	#cv2.imshow("ThsContour", thresh)
#	cv2.imshow("FGMask", fgmask)
	cv2.imshow("Capture", img)
#	cv2.imshow("Capture2", contdraw)

#	cv2.imwrite("./pngs/image-"+str(a).zfill(5)+".png", fgmask)

	if cv2.waitKey(0) == 27:
	    cv2.destroyAllWindows()


