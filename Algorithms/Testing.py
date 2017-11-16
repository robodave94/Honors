import numpy as np
import cv2
import CandidateExamination.ScanLines_Candidates as SL
import CandidateExamination.buildBlobs as BB
import Classification.goal_Classification as gC
import Classification.player_Classification as pC
import Classification.ball_Classification as bC
import timeit as twrite
import csv


def loopexmination(colorScheme,candidateobj,objType,csvFile,count):
    ary=''
    lapCount = []
    dogCount = []
    if objType=="Player" or objType=="Goal":
        candidate=cv2.cvtColor(candidateobj,cv2.COLOR_BGR2HSV)
    else:
        candidate = candidateobj
    msk,time=SL.binary_colorSegmentation_GenericSingle(candidate,colorScheme[0],colorScheme[1])#Single Channel Segmentation
    ary+=repr(time)+','
    gaussV=ScanFunc(msk,objType+'_BinarySegmentation',count)
    lapCount.append(gaussV[0])
    dogCount.append(gaussV[1])
    msk,time=SL.binary_colorSegGrid(candidate, colorScheme[0], colorScheme[1])#Grid Single Channel Scanline
    ary += repr(time)+','
    gaussV =ScanFunc(msk,objType+'_BinaryGrid',count)
    lapCount.append(gaussV[0])
    dogCount.append(gaussV[1])
    msk,time=SL.binary_getVerticalScanLines_buildMaskSingleImage(candidate, colorScheme[0], colorScheme[1])#Vertical Single Channel Scanline
    ary += repr(time) + ','
    gaussV=ScanFunc(msk,objType+'_BinaryVertical',count)
    lapCount.append(gaussV[0])
    dogCount.append(gaussV[1])
    msk,time=SL.binary_getHorizontalScanLines_buildMaskSingleImage(candidate, colorScheme[0], colorScheme[1])#Horizontal Single Channel Scanline
    ary += repr(time) + ','
    gaussV =ScanFunc(msk,objType+'_BinaryHorizontal',count)
    lapCount.append(gaussV[0])
    dogCount.append(gaussV[1])
    msk,time=SL.colorSegmentation_GenericSingle(candidate, colorScheme[0], colorScheme[1])#RGB Channel Segmentation
    ary += repr(time) + ','
    gaussV =ScanFunc(msk,objType+'_ColorSegmentation',count)
    lapCount.append(gaussV[0])
    dogCount.append(gaussV[1])
    msk,time=SL.colorSegGrid(candidate, colorScheme[0], colorScheme[1])#Grid RGB Channel Scanline
    ary += repr(time) + ','
    gaussV =ScanFunc(msk,objType+'_ColorGrid',count)
    lapCount.append(gaussV[0])
    dogCount.append(gaussV[1])
    msk,time=SL.getVerticalScanLines_buildMaskSingleImage(candidate, colorScheme[0], colorScheme[1])#Vertical RGB Channel Scanline
    ary += repr(time) + ','
    gaussV =ScanFunc(msk,objType+'_ColorVertical',count)
    lapCount.append(gaussV[0])
    dogCount.append(gaussV[1])
    msk,time=SL.getHorizontalScanLines_buildMaskSingleImage(candidate, colorScheme[0], colorScheme[1])#Horizontal RGB Channel Scanline
    ary += repr(time)
    gaussV =ScanFunc(msk,objType+'_ColorHorizontal',count)
    if objType=="Ball":
        msk,time=SL.ball_bhuman_VerticalFieldSupression(cv2.cvtColor(candidate,cv2.COLOR_BGR2HSV))#Field Suppression with Darker Area Vertical Single Channel Scanline
        ary += ','+repr(time) + ','
        gaussV =ScanFunc(msk,objType+'_SuppressionDark',count)
        lapCount.append(gaussV[0])
        dogCount.append(gaussV[1])
        msk,time=SL.ball_bhuman_VerticalFieldSupressionDark(cv2.cvtColor(candidate,cv2.COLOR_BGR2HSV))#Field Suppression Vertical Single Channel Scanline
        gaussV =ScanFunc(msk,objType+'_Suppression',count)
        lapCount.append(gaussV[0])
        ary += repr(time) + ','+ str(np.asarray(lapCount).mean())+ ',' + str(np.asarray(dogCount).mean()) +str(count)+ '\n'
    else:
        ary += ',_,_,'+str(np.asarray(lapCount).mean()) + str(np.asarray(dogCount).mean())+','+str(count)+'\n'
    #write row to ods file
    writeToCSV(ary,csvFile)
    return

def writeToCSV(string,odsFile):
    fd = open(odsFile, 'a')
    #fd.write(string)
    fd.close()
    return

def ScanFunc(mask,Tag,cntr):
    ClassificationCandidates=[]
    blobtm=''
    #
    if Tag.__contains__('Ball'):
        blobs=BB.GetBallAreaDoG(mask)
        blobtm+= repr(blobs[2])+','+blobs[1].__str__() + ','
        dogCount=blobs[1]
        ClassificationCandidates.append(blobs[0])
        blobs=BB.GetBallAreaLaplacian(mask)
        blobtm += repr(blobs[2]) +','+blobs[1].__str__()
        lapCount=blobs[1]
        ClassificationCandidates.append(blobs[0])
        writeToCSV(blobtm+','+Tag +','+str(cntr)+ '\n', 'GaussianOperatorTiming.csv')
        RunBallCandidateExamination(ClassificationCandidates,Tag)
    elif Tag.__contains__('Player'):
        blobs = BB.PlayerDifferenceOfGaussians(mask)
        blobtm += repr(blobs[2]) + ',' + blobs[1].__str__() + ','
        dogCount = blobs[1]
        ClassificationCandidates.append(blobs[0])
        blobs = BB.PlayerLaplacianOfGaussians(mask)
        blobtm += repr(blobs[2]) + ',' + blobs[1].__str__()
        lapCount = blobs[1]
        ClassificationCandidates.append(blobs[0])
        writeToCSV(blobtm + ',' + Tag + ',' + str(cntr) + '\n', 'GaussianOperatorTiming.csv')
        RunPlayerCandidateExamination(ClassificationCandidates, Tag)
    else:
        blobs = BB.GoalDifferenceOfGaussians(mask)
        blobtm += repr(blobs[2]) + ',' + blobs[1].__str__() + ','
        dogCount = blobs[1]
        ClassificationCandidates.append(blobs[0])
        blobs = BB.GoalLaplacianOfGaussians(mask)
        blobtm += repr(blobs[2]) + ',' + blobs[1].__str__()
        lapCount = blobs[1]
        ClassificationCandidates.append(blobs[0])
        writeToCSV(blobtm + ',' + Tag + ',' + str(cntr) + '\n', 'GaussianOperatorTiming.csv')
        RunGoalCandidateExamination(ClassificationCandidates, Tag)
    return lapCount,dogCount


def RunBallCandidateExamination(BlobImgs,Tg):

    return


def RunPlayerCandidateExamination(BlobImgs, Tg):
    return


def RunGoalCandidateExamination(BlobImgs, Tg):

    return

def RunExamination():
    cnt=3125
    black_ball = [0, 0, 0], [30, 30, 30]
    blue_markers = [105, 180, 0], [125, 240, 255]
    yellowgoal = [0, 0, 170],[50,255,256]

    for x in range(113, cnt):
        cndte = cv2.imread("../Data/Frames/frame%d.jpg" % (x))
        '''xx=cv2.cvtColor(cndte,cv2.COLOR_BGR2HSV)
        cv2.imshow('',xx)
        cv2.waitKey(2020202)
        cv2.imshow('',cv2.inRange(xx,np.array(yellowgoal[0], dtype="uint16"),np.array(yellowgoal[1], dtype="uint16")))
        cv2.waitKey(2020202)
        '''#examination for black ball
        loopexmination(black_ball,cndte,"Ball",'PreprocessingSegmentation.csv',x)
        loopexmination(yellowgoal, cndte, "Goal", 'PreprocessingSegmentation.csv',x)
        loopexmination(blue_markers, cndte, "Player", 'PreprocessingSegmentation.csv',x)
    return


RunExamination()