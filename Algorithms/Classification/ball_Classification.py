import numpy as np
import cv2
import matplotlib.pyplot as plt
#import CandidateExamination.ScanLines_Candidates as SL
from skimage.feature import hog
from skimage import data, color, exposure
import math
import timeit

#region HiContrastAlgorithms

def HAAR_Classifier(ROI):
    start_time = timeit.default_timer()
    ballfinder = cv2.CascadeClassifier('ballClassifier/cascade.xml')
    gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    faces = ballfinder.detectMultiScale(ROI)
    if len(faces) > 0:
        print faces
    for (x, y, w, h) in faces:
        cv2.rectangle(ROI, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = ROI[y:y + h, x:x + w]
    cv2.imshow('img', ROI)
    cv2.waitKey(23232321)
    return (timeit.default_timer() - start_time)

def ContrastThresholdEx(ROIinput):
    start_time = timeit.default_timer()
    # extract centripital circular blob
    ballCandidate = GetCircle(ROIinput)
    # get validation pixels
    if ballCandidate is None:
        return None
    # region generate evaluation
    spe = ROIinput.shape
    grayscale = cv2.cvtColor(ROIinput, cv2.COLOR_BGR2GRAY)
    # supress the backgroundcv2.cv.CV_FILLED
    mask = np.zeros((spe[0], spe[1]), dtype=np.uint8)
    cv2.circle(mask, (ballCandidate[0][0], ballCandidate[0][1]), ballCandidate[1] - 1, 255, cv2.cv.CV_FILLED)
    cv2.circle(ROIinput, (ballCandidate[0][0], ballCandidate[0][1]), ballCandidate[1], 255, 1)
    nzrs = cv2.findNonZero(mask)
    left_right = np.split(nzrs[:-1], 2)
    examPxls = MeasureContrast(grayscale, [t for t in left_right[0] if (t[0][0] > ballCandidate[0][0])])
    examPxls.append(MeasureContrast(grayscale, [t for t in left_right[0] if (t[0][0] < ballCandidate[0][0])]))
    examPxls.append(MeasureContrast(grayscale, [t for t in left_right[1] if (t[0][0] > ballCandidate[0][0])]))
    examPxls.append(MeasureContrast(grayscale, [t for t in left_right[1] if (t[0][0] < ballCandidate[0][0])]))
    # endregion


    return (timeit.default_timer() - start_time)

def MeasureContrast(gry,ary):
    sum=[]
    for g in ary:
        sum.append( gry[g[0,1],g[0,0]])
    av = np.mean(sum)
    sz=len(ary)
    altVal=2
    returnVal=[]
    for x in range(0,sz-2):
        if  gry[ary[x][0,1],ary[x][0,0]+altVal] < gry[ary[x][0,1],ary[x][0,0]]-av or \
                        gry[ary[x][0, 1]+altVal, ary[x][0, 0]] < gry[ary[x][0,1],ary[x][0,0]]-av or \
                        gry[ary[x][0, 1], ary[x][0, 0]-altVal] < gry[ary[x][0,1],ary[x][0,0]]-av or \
                        gry[ary[x][0, 1]-altVal, ary[x][0, 0]] < gry[ary[x][0,1],ary[x][0,0]]-av:
            returnVal.append([ary[x][0, 1], ary[x][0, 0]])
    return returnVal


def bhumanInteriorExamination(ROIinput):
    start_time = timeit.default_timer()

    def PointsInCircum(z, y, r, n):
        return [(int(math.sin(2 * math.pi / n * x) * r) + y, int(math.cos(2 * math.pi / n * x) * r) + z) for x in
                xrange(0, n + 1)]

    # extract centripital circular blob
    ballCandidate = GetCircle(ROIinput)
    # get validation pixels
    if ballCandidate is None:
        return None
    cv2.circle(ROIinput, (ballCandidate[0][0], ballCandidate[0][1]), ballCandidate[1], (255, 0, 255))
    validationpts = np.array([[ballCandidate[0][0], ballCandidate[0][1]]])
    radSeg = int(ballCandidate[1] / 3)
    for i in range(1, 4):
        validationpts = np.concatenate([validationpts, np.array(
            PointsInCircum(ballCandidate[0][0], ballCandidate[0][1], int(radSeg * i * .9), 4 + i * 4))])

    # classify validation points
    blackpx = [];
    whtpx = []
    for i in validationpts:
        # classify the point as black or white
        if np.mean(ROIinput[i[0], i[1]]) > 37:
            ROIinput[i[0], i[1]] = (0, 0, 255)
            whtpx.append(i)
        else:
            ROIinput[i[0], i[1]] = (0, 255, 0)
            blackpx.append(i)

    ratio = float(len(blackpx)) / len(whtpx)
    print ratio
    cv2.imshow('', ROIinput)
    cv2.waitKey(202020202)
    return False, None,(timeit.default_timer() - start_time)




def HoG_Classifier(ROI):
    #reshape image/make grayscale
    start_time = timeit.default_timer()
    fd, hog_image = hog(ROI, orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(1, 1), block_norm='L2')

    print fd

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(ROI, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()
    return (timeit.default_timer() - start_time)

def GetCircle(ROI):
    shapeX=ROI.shape[0]
    pointOrig=np.array((int(shapeX / 2), int(shapeX / 2)))
    #if cv2.inRange(ROI[pointOrig[0],pointOrig[1]],np.array([2, 53, 19], dtype="uint16"),np.array([48, 180, 92], dtype="uint16"))==255:
    #print ROI[pointOrig[0], pointOrig[1]][1] , ROI[pointOrig[0], pointOrig[1]][2] + 30
    if ROI[pointOrig[0], pointOrig[1]][1] < ROI[pointOrig[0], pointOrig[1]][2]+20:
        c = regionGrow(ROI,pointOrig)
        if not c is None:
            circle =cv2.minEnclosingCircle(c)
            if circle[1]<30 and shapeX>circle[1]*2 and circle[0][0]-circle[1]>0 and circle[0][1]-circle[1]>0\
                    and circle[0][0]+circle[1]<shapeX and circle[0][1]+circle[1]<shapeX:
                return [[int(circle[0][0]),int(circle[0][1])],int(circle[1])]
    return None

def regionGrow(ROI,pntOrig):
    indexA = 0
    indexB = 1
    areapts =[]
    try:
        #region regiongrowing
        exampnt = [pntOrig[0],pntOrig[1]]
        while ROI[exampnt[indexB], exampnt[indexA]][2]+20>ROI[exampnt[indexB], exampnt[indexA]][0]>ROI[exampnt[indexB], exampnt[indexA]][2]-20\
            and ROI[exampnt[indexB], exampnt[indexA]][1]+20>ROI[exampnt[indexB], exampnt[indexA]][2]>ROI[exampnt[indexB], exampnt[indexA]][1]-20:
            exampnt[indexA] = exampnt[indexA] - 1
        areapts.append(np.array(exampnt))
        exampnt = [pntOrig[0], pntOrig[1]]
        while ROI[exampnt[indexB], exampnt[indexA]][2]+20>ROI[exampnt[indexB], exampnt[indexA]][0]>ROI[exampnt[indexB], exampnt[indexA]][2]-20\
            and ROI[exampnt[indexB], exampnt[indexA]][1]+20>ROI[exampnt[indexB], exampnt[indexA]][2]>ROI[exampnt[indexB], exampnt[indexA]][1]-20:
            exampnt[indexA] = exampnt[indexA] + 1
        areapts.append(np.array(exampnt))
        exampnt = [pntOrig[0], pntOrig[1]]
        while ROI[exampnt[indexB], exampnt[indexA]][2]+20>ROI[exampnt[indexB], exampnt[indexA]][0]>ROI[exampnt[indexB], exampnt[indexA]][2]-20\
            and ROI[exampnt[indexB], exampnt[indexA]][1]+20>ROI[exampnt[indexB], exampnt[indexA]][2]>ROI[exampnt[indexB], exampnt[indexA]][1]-20:
            exampnt[indexB] = exampnt[indexB] - 1
        areapts.append(np.array(exampnt))
        exampnt = [pntOrig[0], pntOrig[1]]
        while ROI[exampnt[indexB], exampnt[indexA]][2]+20>ROI[exampnt[indexB], exampnt[indexA]][0]>ROI[exampnt[indexB], exampnt[indexA]][2]-20\
            and ROI[exampnt[indexB], exampnt[indexA]][1]+20>ROI[exampnt[indexB], exampnt[indexA]][2]>ROI[exampnt[indexB], exampnt[indexA]][1]-20:
            exampnt[indexB] = exampnt[indexB] + 1
        areapts.append(np.array(exampnt))

        exampnt = [pntOrig[0], pntOrig[1]]
        while ROI[exampnt[indexB], exampnt[indexA]][2]+20>ROI[exampnt[indexB], exampnt[indexA]][0]>ROI[exampnt[indexB], exampnt[indexA]][2]-20\
            and ROI[exampnt[indexB], exampnt[indexA]][1]+20>ROI[exampnt[indexB], exampnt[indexA]][2]>ROI[exampnt[indexB], exampnt[indexA]][1]-20:
            exampnt[indexB] = exampnt[indexB] + 1
            exampnt[indexA] = exampnt[indexA] + 1
        areapts.append(np.array(exampnt))
        exampnt = [pntOrig[0], pntOrig[1]]
        while ROI[exampnt[indexB], exampnt[indexA]][2]+20>ROI[exampnt[indexB], exampnt[indexA]][0]>ROI[exampnt[indexB], exampnt[indexA]][2]-20\
            and ROI[exampnt[indexB], exampnt[indexA]][1]+20>ROI[exampnt[indexB], exampnt[indexA]][2]>ROI[exampnt[indexB], exampnt[indexA]][1]-20:
            exampnt[indexB] = exampnt[indexB] - 1
            exampnt[indexA] = exampnt[indexA] - 1
        areapts.append(np.array(exampnt))
        exampnt = [pntOrig[0], pntOrig[1]]
        while ROI[exampnt[indexB], exampnt[indexA]][2]+20>ROI[exampnt[indexB], exampnt[indexA]][0]>ROI[exampnt[indexB], exampnt[indexA]][2]-20\
            and ROI[exampnt[indexB], exampnt[indexA]][1]+20>ROI[exampnt[indexB], exampnt[indexA]][2]>ROI[exampnt[indexB], exampnt[indexA]][1]-20:
            exampnt[indexB] = exampnt[indexB]+1
            exampnt[indexA] = exampnt[indexA]-1
        areapts.append(np.array(exampnt))
        exampnt = [pntOrig[0], pntOrig[1]]
        while ROI[exampnt[indexB], exampnt[indexA]][2]+20>ROI[exampnt[indexB], exampnt[indexA]][0]>ROI[exampnt[indexB], exampnt[indexA]][2]-20\
            and ROI[exampnt[indexB], exampnt[indexA]][1]+20>ROI[exampnt[indexB], exampnt[indexA]][2]>ROI[exampnt[indexB], exampnt[indexA]][1]-20:
            exampnt[indexB] = exampnt[indexB] - 1
            exampnt[indexA] = exampnt[indexA] + 1
        areapts.append(np.array(exampnt))
        #endregion
        return np.array(areapts)
    except:
        return None

import glob
import os

#cv2.imshow('',cv2.imread('testBallObj/ball00000.png'))
#cv2.waitKey(20202020)

dataset = glob.glob(os.path.join('testBallObj/','*.png'))
for i in dataset:
    ContrastThresholdEx(cv2.imread(i))

