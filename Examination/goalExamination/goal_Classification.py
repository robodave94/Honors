import numpy as np
import cv2
import timeit
import matplotlib.pyplot as plt
import scipy.stats as st
import timeit
def FuzzyLogicImplementation(img,PostData):
    start_time = timeit.default_timer()
    if len(PostData) == 2:
        heightGoal = max(PostData[0][3], PostData[1][3])
        # INFO STEPS
        # step 1 establish that field transition line is correct if bottom position 1 is within frame ref to pos 2 then establish field line
        # TODO make sure any constants accomodate differing sizes
        if (PostData[0][1] + PostData[0][3] < PostData[1][1] + PostData[1][3] + (heightGoal * 0.15)) and (
                PostData[0][1] + PostData[0][3] > PostData[1][1] + PostData[1][3] - (heightGoal * 0.15)):
            TrueFieldTransPoint = ((PostData[0][0] + (PostData[0][2] / 2), PostData[0][1] + PostData[0][3]),
                                   (PostData[1][0] + (PostData[1][2] / 2), PostData[1][1] + PostData[1][3]))
        else:
            if PostData[0][3] > PostData[0][3]:
                TrueFieldTransPoint = (PostData[0][0] + (PostData[0][2] / 2), PostData[0][1] + PostData[0][3])
            else:
                TrueFieldTransPoint = (PostData[1][0] + (PostData[1][2] / 2), PostData[1][1] + PostData[0][3])
        # find the true high point of the posts
        if (PostData[0][1] < PostData[1][1] + (heightGoal * 0.15)) and (
            PostData[0][1] > PostData[1][1] - (heightGoal * 0.15)):
            trueTopGoalArea = ((PostData[0][0], PostData[0][1]), (PostData[1][0], PostData[1][1]))
        else:
            if PostData[0][0] < PostData[1][0]:
                trueTopGoalArea = (PostData[0][0], PostData[0][1])
            else:
                trueTopGoalArea = (PostData[1][0], PostData[1][1])
        return TrueFieldTransPoint, trueTopGoalArea
    else:
        # single post
        # because we have so little information in this instance,we get the top and bottom of the goal and depending on its status of left or right,
        # we assume that post is totally accurate
        lower = np.array([15, 90, 110], dtype="uint16")
        upper = np.array([180, 255, 240], dtype="uint16")
        TrueFieldTransPoint = (PostData[0] + (PostData[2] / 2), PostData[1] + PostData[3])
        trueTopGoalArea = (PostData[0], PostData[1])
        # Determine if the Post is a left or right component examine area to top left of image and determine if the post is left or right
        listBin = np.nonzero(cv2.inRange(img[PostData[1] - 10:PostData[1] + 10,
                                         (PostData[0] - int(PostData[2] * 0.5)):PostData[0] + int(
                                             PostData[2] * 1.5)], lower, upper))
        if (PostData[2] * 2) - max(listBin[1]) > min(listBin[1]):
            return TrueFieldTransPoint, trueTopGoalArea, 'right', (timeit.default_timer() - start_time)
        elif (PostData[2] * 2) - max(listBin[1]) < min(listBin[1]):
            return TrueFieldTransPoint, trueTopGoalArea, 'left', (timeit.default_timer() - start_time)
        else:
            return TrueFieldTransPoint, trueTopGoalArea, 'Unknown', (timeit.default_timer() - start_time)

def FieldPostTransition(img,PostData):
    def transitionScanlines_singleLines(im, pstPoint):
        # region function variables
        # cv2.inRange(ROI, np.array([2, 53, 19], dtype="uint16"), np.array([48, 180, 92], dtype="uint16"))
        # lower=np.array([0, 90, 30], dtype="uint16");upper = np.array([110, 230, 220], dtype="uint16")
        lower = np.array([2, 53, 19], dtype="uint16")
        upper = np.array([48, 180, 92], dtype="uint16")
        # post areas to find lower area
        PostSec = im[pstPoint[1] + pstPoint[3] / 2:pstPoint[1] + int(pstPoint[3] * 1.5),
                  pstPoint[0]:pstPoint[0] + pstPoint[2]]
        # cv2.imshow('',PostSec)
        # cv2.waitKey(20000)
        # endregion
        x = 0.1
        Postshpe = PostSec.shape
        # cv2.imshow('',PostSec)
        # cv2.waitKey(202002202)
        arry = np.array([])
        # loop through scanlines
        while x < 1:
            index = int(PostSec.shape[1] * x) + 1
            # r=2,g=1,b=0
            leftLine = PostSec[0:PostSec.shape[0], index - 1:index]
            z = 1
            pt = None
            while cv2.inRange(leftLine[z - 1:z], lower, upper) == 0:
                z += 1
            if z > 2 and z < Postshpe[0]:
                arry = np.append(arry, z)
            x += 0.05
        if arry != []:
            return [pstPoint[0] + (pstPoint[2] / 2),
                    (st.mode(np.around(((arry / 500)), 2) * 500)[0][0] + pstPoint[1] + pstPoint[3] / 2)], Postshpe[0]
        return
    start_time = timeit.default_timer()
    pts=[]
    topheight=0

    #establish points of interest
    #get furthest left and furthest right
    if len(PostData)>1:
        for g in PostData:
            lnsRes = transitionScanlines_singleLines(img, g)
            if lnsRes is not None:
                pts.append(lnsRes[0])
                topheight = max([topheight, lnsRes[1]])
            if pts==[]:
            #changed from none
                return [-1,-1,-1,-1], 'Error in Calculation', (timeit.default_timer() - start_time)
        leftpst = min(pts[:])
        rightpst = max(pts[:])

        btmVal = int(max([leftpst[1], rightpst[1]]))
        toppt = btmVal - topheight
        width = int(rightpst[0]+5) - leftpst[0]-5
        GoalArea = [leftpst[0],toppt,width, topheight]

        if rightpst[0]<leftpst[0]+40:
            Classification = 'Partial Goal'
        else:
            Classification = classifyGoalArea(img,np.array(GoalArea))
        if pts.__contains__(leftpst):
            pts.remove(leftpst)
        if pts.__contains__(rightpst):
            pts.remove(rightpst)

        cv2.rectangle(img, (GoalArea[0], GoalArea[1]), (GoalArea[0] + GoalArea[2], GoalArea[1] + GoalArea[3]), 255, 2)
    else:
        #[846,143,width,height]
        Classification=classifyGoalArea(img,PostData[0])
        GoalArea=PostData[0]
        cv2.rectangle(img, (GoalArea[0], GoalArea[1]), (GoalArea[0] + GoalArea[2], GoalArea[1] + GoalArea[3]), 255, 2)
    #cv2.imshow('', img)
    #cv2.waitKey(20202020)
    return GoalArea,Classification,(timeit.default_timer() - start_time)#leftside, rightside,

def classifyGoalArea(image,BoundingBox):
    try:
        #lower = np.array([0, 90, 110], dtype="uint16")
        #upper = np.array([110, 230, 220], dtype="uint16")
        #bounding box=[846,143,401,257]
        #classifcation = cv2.inRange(cv2.cvtColor(image[BoundingBox[1]:BoundingBox[1] + BoundingBox[3], BoundingBox[0]:BoundingBox[0] + BoundingBox[2]],cv2.COLOR_BGR2HSV),
        #                            np.array([0, 200, 150], dtype="uint16"),
        #                            np.array([40, 230, 230], dtype="uint16"))
        SegLeftSide = cv2.inRange(cv2.cvtColor(image[BoundingBox[1]:BoundingBox[1]+BoundingBox[3],
                                               BoundingBox[0]:BoundingBox[0]+int(BoundingBox[2]*0.25)],cv2.COLOR_BGR2HSV),
                                               np.array([0, 200, 40], dtype="uint16"),
                                               np.array([40, 230, 230], dtype="uint16"))
        SegRightSide = cv2.inRange(cv2.cvtColor(image[BoundingBox[1]:BoundingBox[1]+BoundingBox[3],
                                               BoundingBox[0]+int(BoundingBox[2]*0.75):BoundingBox[0]+int(BoundingBox[2])],cv2.COLOR_BGR2HSV),
                                               np.array([0, 200, 40], dtype="uint16"),
                                               np.array([40, 230, 230], dtype="uint16"))
        lftsz=len(np.nonzero(SegLeftSide)[0])
        rghtsz = len(np.nonzero(SegRightSide)[0])
        #cv2.imshow('',cv2.cvtColor(image,cv2.COLOR_BGR2HSV))
        #cv2.waitKey(20202020)
        #print len(lftsz[0]),len(rghtsz[0])'''
        if lftsz>0 and rghtsz>0:
            return 'Complete Goal'
        elif lftsz > 0 or rghtsz > 0:
            return 'Partial Goal'
        else:
            return 'Error in Detection'
        #else:
        #    return 'Complete Goal'
    except:
        return 'Error in Detection'


def BlobLogicImplementation(img,PostData):

    def recursiveMelding(rects):
        for p in range(0, len(rects)):
            i = rects[p]
            for b in range(p+1, len(rects)):
                u = rects[b]
                if i is not u and area(i, u):
                    leftPt = min(i[0], u[0])
                    heightpt = min(i[1], u[1])
                    maxwidth = max(i[0] + i[2], u[0] + u[2]) - leftPt
                    maxHeight = max(i[1] + i[3], u[1] + u[3]) - heightpt
                    newBx = [leftPt, heightpt, maxwidth, maxHeight]
                    if p<b:
                        rects.__delitem__(b)
                        rects.__delitem__(p)
                    else:
                        rects.__delitem__(p)
                        rects.__delitem__(b)
                    rects.append(newBx)
                    return recursiveMelding(rects)

        return np.array(rects)[0]
    start_time = timeit.default_timer()
    GoalArea=recursiveMelding(list(PostData))
    if GoalArea.shape==(4,):
        #runClassification Algorithm
        Classification = classifyGoalArea(img, GoalArea)
    else:
        #print GoalArea.shape
        #cv2.imshow('Error',img)
        #cv2.waitKey(20202020)
        return [-1,-1,-1,-1], 'Error in Calculation', (timeit.default_timer() - start_time)
    '''else:
        print len(GoalArea)
        #for g in GoalArea:
        #    cv2.rectangle(img, (g[0], g[1]), (g[0] + g[2], g[1] + g[3]), 255, 1)
        #cv2.imshow('',img)
        #cv2.waitKey(202020202)
        Classification='unkown'
        if 
        if GoalArea[0][0]<GoalArea[1][0]:
            i=GoalArea[0]
            u=GoalArea[1]
        else:
            i = GoalArea[1]
            u = GoalArea[0]
        leftPt = min(i[0], u[0])
        heightpt = min(i[1], u[1])
        maxwidth = max(i[0] + i[2], u[0] + u[2]) - leftPt
        maxHeight = max(i[1] + i[3], u[1] + u[3]) - heightpt
        GoalArea=[leftPt,heightpt,maxwidth,maxHeight]
        Classification='Complete Goal'
        #cv2.rectangle(img, (GoalArea[0], GoalArea[1]), (GoalArea[0] + GoalArea[2], GoalArea[1] + GoalArea[3]), 255, 2)
    #cv2.imshow('', img)
    #cv2.waitKey(20202020)
    el'''

    return GoalArea, Classification, (timeit.default_timer() - start_time)

def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0])
    dy = min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1])
    if (dx>=0) and (dy>=0):
        return True
    else:
        return False

def StatisticalRules(img,PostData):
    start_time = timeit.default_timer()
    if len(PostData)>1:
        leftpt=img.shape[1]
        rightpt=0
        lowpt=0
        highpt=img.shape[0]
        for g in PostData:
            leftpt = min(leftpt,g[0])
            rightpt = max(rightpt,g[0]+g[2])
            lowpt = max(lowpt,g[1])
            highpt = min(highpt,g[1]+g[3])
        GoalArea=[leftpt,lowpt,rightpt-leftpt,highpt-lowpt]
        #print GoalArea
        Classification = classifyGoalArea(img, GoalArea)
        '''cv2.rectangle(img, (GoalArea[0], GoalArea[1]), (GoalArea[0] + GoalArea[2], GoalArea[1] + GoalArea[3]), 255, 2)
        cv2.imshow('', img)
        cv2.waitKey(20202020)'''
        return GoalArea, Classification, (timeit.default_timer() - start_time)
    else:
        Classification = classifyGoalArea(img, PostData[0])
        '''cv2.rectangle(img, (PostData[0][0], PostData[0][1]), (PostData[0][0] + PostData[0][2], PostData[0][1] + PostData[0][3]), 255, 2)
        cv2.imshow('', img)
        cv2.waitKey(20202020)'''
        return PostData[0], Classification, (timeit.default_timer() - start_time)








