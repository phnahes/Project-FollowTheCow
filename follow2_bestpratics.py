import cv2
import numpy as np


if __name__ == '__main__': 
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("input")
    cv2.namedWindow("sig2")
    cv2.namedWindow("detect")
    BGsample = 20 # number of frames to gather BG samples from at start of capture
    success, img = cap.read()
    width = cap.get(3)
    height = cap.get(4)
    if success:
        acc = np.zeros((height, width), np.float32) # 32 bit accumulator
        sqacc = np.zeros((height, width), np.float32) # 32 bit accumulator
        for i in range(20): a = cap.read() # dummy to warm up sensor
        # gather BG samples
        for i in range(BGsample):
            success, img = cap.read()
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.accumulate(frame, acc)
            cv2.accumulateSquare(frame, sqacc)
        #
        M = acc/float(BGsample)
        sqaccM = sqacc/float(BGsample)
        M2 = M*M
        sig2 = sqaccM-M2
        # have BG samples now
        # calculate upper and lower bounds of detection window around mean.
        # coerce into 8bit image space for cv2.inRange compare
        detectmin = cv2.convertScaleAbs(M-sig2)
        detectmax = cv2.convertScaleAbs(M+sig2)
        # start FG detection
        key = -1
        while(key < 0):
            success, img = cap.read()
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            level = cv2.inRange(frame, detectmin, detectmax)

            # Find and Draw Contours
            ret,thresh2 = cv2.threshold(level,127,255,1)
            contours, hierarchy = cv2.findContours(level,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            print ("Number of Contours: ", len(contours))

            for h,cnt in enumerate(contours):
                 mask = np.zeros(level.shape,np.uint8)
                 cv2.drawContours(mask,[cnt],0,255,-1)
                 mean = cv2.mean(level,mask = mask)

            approx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
            hull = cv2.convexHull(cnt)

            (x,y),radius = cv2.minEnclosingCircle(approx)
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(frame,center,radius,(0,255,0),2)


            cv2.imshow("input", frame)
            cv2.imshow("sig2", M/200)
            cv2.imshow("detect", level)
            key = cv2.waitKey(1)
    cv2.destroyAllWindows()
