import numpy as np
import cv2
import timeit
import matplotlib.pyplot as plt
import scipy.stats as st
import math
import timeit
import tensorflow as tf
from skimage.feature import hog as HOGOP
from sklearn.externals import joblib
from skimage import data, color, exposure

graph = tf.get_default_graph()
sess = tf.Session()
# First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('Player_trainedModel/playermodel-7999.meta')
y_true = graph.get_tensor_by_name("y_output:0")
x = graph.get_tensor_by_name("x_im_input:0")
y_pred = graph.get_tensor_by_name("y_pred:0")
keep_prob = graph.get_tensor_by_name("keepProb:0")
def ConvoloutionalNeuralNetwork(ROI):
    start_time = timeit.default_timer()
    saver.restore(sess, tf.train.latest_checkpoint('Player_trainedModel/'))
    # reshape image/make grayscale
    x_im = ROI.reshape((-1, 40, 64, 1))
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


def FieldSuppFindVal(image,passedVals):

    start_time = timeit.default_timer()
    retVal=[]
    for g in passedVals:
        binImg=cv2.inRange(cv2.cvtColor(image[g[1]:g[1] + g[3], g[0]:g[0] + g[2]],cv2.COLOR_BGR2HSV), np.array([100, 140, 40], dtype="uint16"),np.array([130, 256, 256], dtype="uint16"))
        if np.array(binImg).__contains__(255):
            updatedpos=np.nonzero(binImg)
            minVal=min(updatedpos[1])
            g[0]=g[0]+minVal
            g[2]=max(updatedpos[1])-minVal

            if g[2] / float(g[3]) < 5 and g[2] / float(g[3]) > 0.1 and g[2]>6:
                #run a loop through each found POI
                minpt = 800
                for x in np.arange(g[0]+3, g[0]+g[2], (g[2])/int(5)):
                    differential= int(g[1] - (g[3] * 3))
                    varDiff=max(0,differential)
                    #print varDiff,x
                    gg = np.nonzero(cv2.inRange(image[varDiff:g[1]+g[3], x - 1:x],
                                np.array([200, 0, 0], dtype="uint16"),np.array([256, 150, 150], dtype="uint16")))
                    if len(gg[0])>0:
                        minpt=min(minpt,(min(gg[0])+varDiff))
                        g[3] = g[1]+g[3]-minpt
                        g[1]=minpt
                retVal.append(g)
                #cv2.rectangle(image,(g[0],g[1]),(g[0]+g[2],g[1]+g[3]),(255,255,255),1)

                #cv2.imshow('',image[g[1]:g[1]+g[3],g[0]:g[0]+g[2]])
                #cv2.waitKey(20202020)
    return retVal,(timeit.default_timer() - start_time)

#region clustering
sx=45
minRat=0.25
maxRat=0.85
DistanceValue=50
def Clustering(img,recs):
    def NegativeSetMergeRecursive(rect):

        for b in range(0, len(rect)):
            x = rect[b]
            for h in range(b + 1, len(rect)):
                y = rect[h]
                if y[0] < x[0] < y[0] + y[2] or x[0] < y[0] < x[0] + x[2] or y[0] + y[2] + sx > x[0] or x[0] + x[
                    2] + sx > y[0]:
                    leftPt = min(x[0], y[0])
                    heightpt = min(x[1], y[1])
                    maxwidth = max(x[0] + x[2], y[0] + y[2]) - leftPt
                    maxHeight = max(x[1] + x[3], y[1] + y[3]) - heightpt
                    if h < b:
                        rect.__delitem__(b)
                        rect.__delitem__(h)
                    else:
                        rect.__delitem__(h)
                        rect.__delitem__(b)
                    newBx = np.array([leftPt, heightpt, maxwidth, maxHeight])
                    rect.append(newBx)
                    return NegativeSetMergeRecursive(rect)
        return rect
    def recursiveDistanceMeasurement(rect):

        for b in range(0, len(rect)):
            x = rect[b]
            for h in range(b + 1, len(rect)):
                y = rect[h]
                if not np.array_equal(x, y):
                    if y[0] < x[0] < y[0] + y[2] or x[0] < y[0] < x[0] + x[2] or math.sqrt(
                                    math.pow(x[0] + (x[2] / 2) - (y[0] + (y[2] / 2)), 2) + math.pow(
                                                    x[1] + (x[3] / 2) - (y[1] + (y[3] / 2)), 2)) < DistanceValue:
                        # print math.sqrt(math.pow(x[0]+(x[2]/2)-(y[0]+(y[2]/2)),2)+math.pow(x[1]+(x[3]/2)-(y[1]+(y[3]/2)),2))
                        leftPt = min(x[0], y[0])
                        heightpt = min(x[1], y[1])
                        maxwidth = max(x[0] + x[2], y[0] + y[2]) - leftPt
                        maxHeight = max(x[1] + x[3], y[1] + y[3]) - heightpt
                        if h < b:
                            rect.__delitem__(b)
                            rect.__delitem__(h)
                        else:
                            rect.__delitem__(h)
                            rect.__delitem__(b)
                        newBx = np.array([leftPt, heightpt, maxwidth, maxHeight])
                        rect.append(newBx)
                        return recursiveDistanceMeasurement(rect)
        return rect

    start_time = timeit.default_timer()
    PlayersDetected=[]
    distanceThresh=40
    #find overlapping areas
    #split up rectangle array based on value of distance apart

    recurelim=recursiveDistanceMeasurement(list(recs))
    b=0
    while b <len(recurelim):

        g = recurelim[b]
        #print float(g[2]) / g[3] , g[2]*g[3]
        if g[2]*g[3]<1200:
            recurelim.__delitem__(b)
            if len(recurelim) == 0:
                break
            b -= 1
        elif minRat < float(g[2]) / g[3] < maxRat:
            PlayersDetected.append(g)
            recurelim.__delitem__(b)
            if len(recurelim)==0:
                break
            b-=1
        b+=1
    if len(recurelim)>0:
        #examine negative areas for cross analysis
        PlayersDetected+=NegativeSetMergeRecursive(recurelim)
        for are in recurelim:
            cv2.rectangle(img, (are[0], are[1]), (are[0] + are[2], are[1] + are[3]), (255, 0, 0), 2)
    '''
    for x in PlayersDetected:
        cv2.rectangle(img,(x[0],x[1]),(x[0]+x[2],x[1]+x[3]),(0,255,0),2)
    cv2.imshow('',img)
    cv2.waitKey(202020222)
    '''
    return PlayersDetected, (timeit.default_timer() - start_time)

def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    dy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if (dx >= 0) and (dy >= 0):
        return True
    else:
        return False

#endregion
