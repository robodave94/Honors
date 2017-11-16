import numpy as np
import cv2
import matplotlib.pyplot as plt
#import CandidateExamination.ScanLines_Candidates as SL
from skimage.feature import hog as HOGOP
from sklearn.externals import joblib
from skimage import data, color, exposure
import math
import timeit
import tensorflow as tf

#region HiContrastAlgorithms


#region standard anaylsis
def ContrastThresholdEx(ROIinput):
    start_time = timeit.default_timer()
    # extract centripital circular blob
    ballCandidate = GetCircle(ROIinput)
    # get validation pixels
    if ballCandidate is None:
        return (timeit.default_timer() - start_time), False
    # region generate evaluation
    spe = ROIinput.shape
    grayscale = cv2.cvtColor(ROIinput, cv2.COLOR_BGR2GRAY)
    # supress the backgroundcv2.cv.CV_FILLED
    mask = np.zeros((spe[0], spe[1]), dtype=np.uint8)
    cv2.circle(mask, (ballCandidate[0][0], ballCandidate[0][1]), ballCandidate[1] - 1, 255, cv2.cv.CV_FILLED)
    #cv2.circle(ROIinput, (ballCandidate[0][0], ballCandidate[0][1]), ballCandidate[1], 255, 1)
    nzrs = cv2.findNonZero(mask)
    if len(nzrs)%2!=0:
        left_right = np.split(nzrs[:-1], 2)
    else:
        left_right = np.split(nzrs, 2)
    examPxls = MeasureContrast(grayscale, [t for t in left_right[0] if (t[0][0] > ballCandidate[0][0])])
    examPxls.append(MeasureContrast(grayscale, [t for t in left_right[0] if (t[0][0] < ballCandidate[0][0])]))
    examPxls.append(MeasureContrast(grayscale, [t for t in left_right[1] if (t[0][0] > ballCandidate[0][0])]))
    examPxls.append(MeasureContrast(grayscale, [t for t in left_right[1] if (t[0][0] < ballCandidate[0][0])]))
    # endregion
    leng=len(examPxls)
    if 0<leng<200 and leng/float(2*math.pi*ballCandidate[1])>0.15:
        return (timeit.default_timer() - start_time),True
    return (timeit.default_timer() - start_time), False

def MeasureContrast(gry,ary):
    sum=[]
    for g in ary:
        sum.append( gry[g[0,1],g[0,0]])
    av = np.mean(sum)
    #print av
    if av>160:
        av=80
    elif av>90:
        av=60
    elif av>50:
        av=40
    else:
        av=20
    sz=len(ary)
    altVal=2
    returnVal=[]

    for x in range(0,sz-3):
        try:
            if gry[ary[x][0,1],ary[x][0,0]+altVal] < gry[ary[x][0,1],ary[x][0,0]]-av or \
                            gry[ary[x][0, 1]+altVal, ary[x][0, 0]] < gry[ary[x][0,1],ary[x][0,0]]-av or \
                            gry[ary[x][0, 1], ary[x][0, 0]-altVal] < gry[ary[x][0,1],ary[x][0,0]]-av or \
                            gry[ary[x][0, 1]-altVal, ary[x][0, 0]] < gry[ary[x][0,1],ary[x][0,0]]-av:
                returnVal.append([ary[x][0, 1], ary[x][0, 0]])
        except IndexError:
            continue
    return returnVal


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
    #cv2.imshow('None',ROI)
    #cv2.waitKey(22020202)
    return None

def regionGrow(ROI,pntOrig):
    indexA = 0
    indexB = 1
    areapts =[]
    #cv2.imshow('',ROI)
    #cv2.waitKey(202020202)
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
        '''
        exampnt = [pntOrig[0], pntOrig[1]]
        #cv2.inRange(, np.array([42,130,50], dtype="uint16"),np.array([58,256,180], dtype="uint16"))
        while cv2.inRange(ROI[exampnt[indexB]-1:exampnt[indexB],exampnt[indexA]-1:exampnt[indexA]], np.array([42,130,50], dtype="uint16"),np.array([58,256,180], dtype="uint16"))==0:
            exampnt[indexA] = exampnt[indexA] - 1
        areapts.append(np.array(exampnt))
        exampnt = [pntOrig[0], pntOrig[1]]
        while cv2.inRange(ROI[exampnt[indexB], exampnt[indexA]], np.array([42,130,50], dtype="uint16"),np.array([58,256,180], dtype="uint16"))==0:
            exampnt[indexA] = exampnt[indexA] + 1
        areapts.append(np.array(exampnt))
        exampnt = [pntOrig[0], pntOrig[1]]
        while cv2.inRange(ROI[exampnt[indexB], exampnt[indexA]], np.array([42,130,50], dtype="uint16"),np.array([58,256,180], dtype="uint16"))==0:
            exampnt[indexB] = exampnt[indexB] - 1
        areapts.append(np.array(exampnt))
        exampnt = [pntOrig[0], pntOrig[1]]
        while cv2.inRange(ROI[exampnt[indexB], exampnt[indexA]], np.array([42,130,50], dtype="uint16"),np.array([58,256,180], dtype="uint16"))==0:
            exampnt[indexB] = exampnt[indexB] + 1
        areapts.append(np.array(exampnt))

        exampnt = [pntOrig[0], pntOrig[1]]
        while cv2.inRange(ROI[exampnt[indexB], exampnt[indexA]], np.array([42,130,50], dtype="uint16"),np.array([58,256,180], dtype="uint16"))==0:
            exampnt[indexB] = exampnt[indexB] + 1
            exampnt[indexA] = exampnt[indexA] + 1
        areapts.append(np.array(exampnt))
        exampnt = [pntOrig[0], pntOrig[1]]
        while cv2.inRange(ROI[exampnt[indexB], exampnt[indexA]], np.array([42,130,50], dtype="uint16"),np.array([58,256,180], dtype="uint16"))==0:
            exampnt[indexB] = exampnt[indexB] - 1
            exampnt[indexA] = exampnt[indexA] - 1
        areapts.append(np.array(exampnt))
        exampnt = [pntOrig[0], pntOrig[1]]
        while cv2.inRange(ROI[exampnt[indexB], exampnt[indexA]], np.array([42,130,50], dtype="uint16"),np.array([58,256,180], dtype="uint16"))==0:
            exampnt[indexB] = exampnt[indexB] + 1
            exampnt[indexA] = exampnt[indexA] - 1
        areapts.append(np.array(exampnt))
        exampnt = [pntOrig[0], pntOrig[1]]
        while cv2.inRange(ROI[exampnt[indexB], exampnt[indexA]], np.array([42,130,50], dtype="uint16"),np.array([58,256,180], dtype="uint16"))==0:
            exampnt[indexB] = exampnt[indexB] - 1
            exampnt[indexA] = exampnt[indexA] + 1
        areapts.append(np.array(exampnt))'''
        #endregion
        return np.array(areapts)
    except:
        return None

def bhumanInteriorExamination(ROIinput):
    start_time = timeit.default_timer()

    def PointsInCircum(z, y, r, n):
        return [(int(math.sin(2 * math.pi / n * x) * r) + y, int(math.cos(2 * math.pi / n * x) * r) + z) for x in
                xrange(0, n + 1)]

    # extract centripital circular blob
    ballCandidate = GetCircle(ROIinput)
    # get validation pixels
    if ballCandidate is None:
        return (timeit.default_timer() - start_time), False
    #cv2.circle(ROIinput, (ballCandidate[0][0], ballCandidate[0][1]), ballCandidate[1], (255, 0, 255))
    #cv2.imshow('',ROIinput)
    #cv2.waitKey(2020222)
    #cv2.cvtColor(ROIinput, cv2.COLOR_BGR2HSV)
    validationpts = np.array([[ballCandidate[0][0], ballCandidate[0][1]]])
    radSeg = int(ballCandidate[1] / 3)
    for i in range(1, 4):
        validationpts = np.concatenate([validationpts, np.array(
            PointsInCircum(ballCandidate[0][0], ballCandidate[0][1], int(radSeg * i * .7), 4 + i * 4))])

    # classify validation points
    blackpx = []
    whtpx = []
    outliers=0
    for i in validationpts:
        #print ROIinput[i[0], i[1]]
        # check if point is approximatly grayscale
        if ROIinput[i[0], i[1]][1]+30>ROIinput[i[0], i[1]][0]>ROIinput[i[0], i[1]][1]-30 and\
                                        ROIinput[i[0], i[1]][2]+30>ROIinput[i[0], i[1]][1]>ROIinput[i[0], i[1]][2]-30 and \
                                        ROIinput[i[0], i[1]][0] + 30 > ROIinput[i[0], i[1]][2] > ROIinput[i[0], i[1]][0] - 30:
        #############
            # classify the point as black or white
            if np.mean(ROIinput[i[0], i[1]]) > 37:
                #ROIinput[i[0], i[1]] = (0, 0, 255)
                whtpx.append(i)
            else:
                #ROIinput[i[0], i[1]] = (0, 255, 0)
                blackpx.append(i)
        else:
            outliers+=1
    '''try:
        ratio = float(len(blackpx)) / len(whtpx)
    except:
        cv2.imshow('ZroErr',ROIinput)
        cv2.waitKey(2020202)'''
    if len(blackpx)>0 and len(whtpx)>0 and outliers<=5:
        return (timeit.default_timer() - start_time),True
    else:
        return(timeit.default_timer() - start_time), False

#endregion


#region learning methods
def HAAR_Classifier(gray):
    ballfinder = cv2.CascadeClassifier('ballClassifier/cascade.xml')
    start_time = timeit.default_timer()
    balls = ballfinder.detectMultiScale(gray)
    if len(balls) > 0:
        return (timeit.default_timer() - start_time),True
    return (timeit.default_timer() - start_time), False
    #for (x, y, w, h) in balls:
    #    cv2.rectangle(, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #roi_gray = gray[y:y + h, x:x + w]
        #roi_color = ROI[y:y + h, x:x + w]
    #cv2.imshow('img', ROI)
    #cv2.waitKey(23232321)

graph = tf.get_default_graph()
sess = tf.Session()
# First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('Ball_trainedModel/ballmodel-19999.meta')
y_true = graph.get_tensor_by_name("y_output:0")
x = graph.get_tensor_by_name("x_im_input:0")
y_pred = graph.get_tensor_by_name("y_pred:0")
keep_prob = graph.get_tensor_by_name("keepProb:0")
def ConvoloutionalNeuralNetwork(ROI):
    start_time = timeit.default_timer()
    saver.restore(sess, tf.train.latest_checkpoint('Ball_trainedModel/'))
    # reshape image/make grayscale
    x_im = ROI.reshape((-1, 40, 40, 1))
    # ,y_pred:np.zeros((1,2))
    feed_dict_testing = {x: x_im, y_true: np.zeros((1, 2)), keep_prob: 0.5}
    res=sess.run(y_pred, feed_dict=feed_dict_testing)
    if res[0][0]>res[0][1]:
        return (timeit.default_timer() - start_time),True
    return (timeit.default_timer() - start_time),False

clf = joblib.load('svmHoG/svm.model')
def HoG_Classifier(ROI):
    #load classifier, in a real application this would not take time as it will be preloaded
    start_time = timeit.default_timer()
    fd = np.array(HOGOP(ROI, 9, [8,8], [3,3], 'L1', False, False)).reshape((1,-1))
    pred = clf.predict(fd)
    if pred==[1]:
        return (timeit.default_timer() - start_time),True
    return (timeit.default_timer() - start_time), False

#endregion
'''
import glob
import os

#cv2.imshow('',cv2.imread('testBallObj/ball00000.png'))
#cv2.waitKey(20202020)

dataset = glob.glob(os.path.join('testBallObj/','*.png'))
for i in dataset:
    #print bhumanInteriorExamination(cv2.imread(i))
    #print ContrastThresholdEx(cv2.imread(i))
    print ConvoloutionalNeuralNetwork(cv2.resize(cv2.imread(i,cv2.IMREAD_GRAYSCALE),(40,40)))
    print HoG_Classifier(cv2.resize(cv2.imread(i,cv2.IMREAD_GRAYSCALE),(40,40)))
    #print HAAR_Classifier(cv2.imread(i))
'''