import numpy as np
import cv2
import scipy.ndimage
import timeit
#region Goal Blobs

def PlayerLaplacianOfGaussians(image):
    start_time = timeit.default_timer()
    lap = scipy.ndimage.filters.gaussian_laplace(image,[5,4])
    timing=(timeit.default_timer() - start_time)
    rslt = getPlayAreas(lap)
    #dispGoalPosts(image,rslt)
    return rslt,timing

def PlayerFieldSupressionDiff(im):
    start_time = timeit.default_timer()
    g1 = cv2.GaussianBlur(im, (1, 3), 0, 0, 0)
    g2 = cv2.GaussianBlur(im, (1, 9), 0, 0, 0)
    diff = g2 - g1
    timing = (timeit.default_timer() - start_time)
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('',thresh)
    cv2.waitKey(2020202)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea, reverse=True)
    rslt=[list(cv2.boundingRect(cnt)) for cnt in contours]
    return rslt,timing

def PlayerDifferenceOfGaussians(image):
    start_time = timeit.default_timer()
    g1=cv2.GaussianBlur(image, (1,1),0, 0, 0)
    g2=cv2.GaussianBlur(image, (9,9), 0, 0, 0)
    diff = g2-g1
    timing = (timeit.default_timer() - start_time)
    thresh=cv2.threshold(diff,100,255,cv2.THRESH_BINARY)[1]
    #cv2.imshow('', thresh)
    #cv2.waitKey(1010110)
    rslt = getPlayAreas(thresh)
    #dispGoalPosts(image,g2)
    return rslt,timing

def getPlayAreas(inVal):
    contours, _ = cv2.findContours(inVal, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea,reverse=True)

    #print len(contours)
    ##for x in contours:
    #    print cv2.contourArea(x)
    return recursiveOverLapping([list(cv2.boundingRect(cnt)) for cnt in contours[0:200]])
    #sort rects into regions


def recursiveOverLapping(rect):
    def area(a, b):  # returns None if rectangles don't intersect
        dx = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
        dy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
        if (dx >= 0) and (dy >= 0):
            return True
        else:
            return False
    for b in range(0,len(rect)):
        x=rect[b]
        for h in range(b+1, len(rect)):
            y = rect[h]
            if not np.array_equal(x,y) and area(x, y):
                #examine for distance between middle of contours
                #dist=math.sqrt(math.pow(x[0]+int(x[2]/2)-(y[0]+int(y[2]/2)),2)+math.pow(x[1]+int(x[3]/2)-(y[1]+int(y[3]/2)),2))
                #if max(x[2],x[3])+max(y[2],y[3])>=dist:
                    #print x,y
                origin=min(x[0],y[0]),min(x[1],y[1])
                if h < b:
                    rect.__delitem__(b)
                    rect.__delitem__(h)
                else:
                    rect.__delitem__(h)
                    rect.__delitem__(b)
                '''rect.append(np.array([origin[0], origin[1], max(y[0] + x[2], y[0] + y[2]) - origin[0],
                max(x[1] + x[3], y[1] + y[3]) - origin[1]]))
                print [origin[0], origin[1], max(y[0] + x[2], y[0] + y[2]) - origin[0],
                max(x[1] + x[3], y[1] + y[3]) - origin[1]]'''
                leftPt = min(x[0], y[0])
                heightpt = min(x[1], y[1])
                maxwidth = max(x[0] + x[2], y[0] + y[2]) - leftPt
                maxHeight = max(x[1] + x[3], y[1] + y[3]) - heightpt
                newBx = np.array([leftPt, heightpt, maxwidth, maxHeight])
                rect.append(newBx)
                return recursiveOverLapping(rect)
    return np.array(rect)