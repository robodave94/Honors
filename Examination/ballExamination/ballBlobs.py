import numpy as np
import cv2
import scipy.ndimage
import timeit
#region Goal Blobs

def GetBallAreaLaplacian(ballCan):
    if type(ballCan[0,0]) is not np.uint8:
        ballCan=cv2.cvtColor(ballCan,cv2.COLOR_BGR2GRAY)
    start_time = timeit.default_timer()
    lap = scipy.ndimage.filters.gaussian_laplace(ballCan, [5, 4])
    returnTime = (timeit.default_timer() - start_time)
    returnVal = genBallAreas(lap)
    return returnVal, len(returnVal), returnTime

def GetBallAreaDoG(ballCan):
    if type(ballCan[0,0]) is not np.uint8:
        ballCan=cv2.cvtColor(ballCan,cv2.COLOR_BGR2GRAY)
    start_time = timeit.default_timer()
    g1 = cv2.GaussianBlur(ballCan, (1, 1), 0, 0, 0)
    g2 = cv2.GaussianBlur(ballCan, (15, 15), 0, 0, 0)
    diff = g2 - g1
    returnTime=(timeit.default_timer() - start_time)
    returnVal=genBallAreas(diff)
    return returnVal,len(returnVal),returnTime

def genBallAreas(mask):
    thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea,reverse=True)
    #todo fixe to return a series of bounding boxes
    ary=np.array([list(cv2.boundingRect(cnt)) for cnt in contours[0:8]])

    #print ary
    #return loopFunc(ary)
    return repBox(list(ary))

def repBox(arry):
    for c in range(0,len(arry)):
        box=arry[c]
        #expand to square
        if float(box[2])/box[3]>1:
            #increase height
            box[1] = box[1]-(box[2]-box[3])/2
            box[3]=box[2]
        elif float(box[2])/box[3]<1:
            box[0] = box[0]-(box[3]-box[2])/2
            box[2]=box[3]
        if box[0]<1:
            box[0]=1
        if box[1] < 1:
            box[1] = 1
        if box[3]>100:
            arry.__delitem__(c)
            return repBox(arry)

    return arry