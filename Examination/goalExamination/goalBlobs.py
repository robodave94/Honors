import numpy as np
import cv2
import scipy.ndimage
import timeit
#region Goal Blobs

def GoalLaplacianOfGaussians(image,RGBimage):
    start_time = timeit.default_timer()
    lap = scipy.ndimage.filters.gaussian_laplace(image,[5,4])
    timing=(timeit.default_timer() - start_time)
    rslt = getPostAreas(lap,RGBimage)
    #dispGoalPosts(image,rslt)
    return rslt,timing

def GoalDifferenceOfGaussians(image,RGBimage):
    start_time = timeit.default_timer()
    g1=cv2.GaussianBlur(image, (1,1),0, 0, 0)
    g2=cv2.GaussianBlur(image, (9,9), 0, 0, 0)
    diff = g2-g1
    timing = (timeit.default_timer() - start_time)
    #cv2.imshow('', diff)
    #cv2.waitKey(1010110)
    rslt = getPostAreas(diff,RGBimage)
    #dispGoalPosts(image,g2)
    return rslt,timing

'''def dispGoalPosts(im,res):
    if len(res)==4:
        cv2.rectangle(im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (255, 255, 0), 1)
    else:
        cv2.rectangle(im, (res[0][0], res[0][1]), (res[0][0] + res[0][2], res[0][1] + res[0][3]), (255, 255, 0), 1)
        cv2.rectangle(im, (res[1][0], res[1][1]), (res[1][0] + res[1][2], res[1][1] + res[1][3]), (255, 255, 0), 1)
    cv2.imshow('',im)
    cv2.waitKey(1010110)
    return
'''
def getPostAreas(inVal,rgb):
    thresh = cv2.threshold(inVal, 50, 255, cv2.THRESH_BINARY)[1]
    #cv2.imshow('', thresh)
    #cv2.waitKey(202020202)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea,reverse=True)
    contours=contours[0:10]
    rect= np.array([list(cv2.boundingRect(cnt)) for cnt in contours])
    return recursiveElimination(list(rect),rgb)


def recursiveElimination(rec,im):
    for b in range(0,len(rec)):
        v=rec[b]
        classifcation=cv2.inRange(im[v[1]:v[1] + v[3], v[0]:v[0] + v[2]], np.array([0, 150, 0],dtype="uint16"),
                    np.array([40, 230, 230], dtype="uint16"))
        whitepx = len(np.nonzero(classifcation)[0])
        if whitepx<300:
            rec.__delitem__(b)
            return recursiveElimination(rec,im)
    return rec