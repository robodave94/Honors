import numpy as np
import cv2
import ballScanLines_Candidates as SL
import ballBlobs as BB
import ball_Classification as bC
import pywt
from pywt import cwt

def loopexmination(colorScheme,candidate,csvFile,count):
    ary=''
    msk,time=SL.binary_colorSegmentation_GenericSingle(candidate,colorScheme[0],colorScheme[1])#Single Channel Segmentation
    gaussV = ScanFunc(msk)
    ary +=repr(time)+','+str(gaussV[0])+','+str(gaussV[1])+','
    msk,time=SL.binary_colorSegGrid(candidate, colorScheme[0], colorScheme[1])#Grid Single Channel Scanline
    gaussV = ScanFunc(msk)
    ary += repr(time) + ',' + str(gaussV[0]) + ',' + str(gaussV[1]) + ','
    msk,time=SL.binary_getVerticalScanLines_buildMaskSingleImage(candidate, colorScheme[0], colorScheme[1])#Vertical Single Channel Scanline
    gaussV = ScanFunc(msk)
    ary += repr(time) + ',' + str(gaussV[0]) + ',' + str(gaussV[1]) + ','
    msk,time=SL.binary_getHorizontalScanLines_buildMaskSingleImage(candidate, colorScheme[0], colorScheme[1])#Horizontal Single Channel Scanline
    gaussV =ScanFunc(msk)
    ary += repr(time) + ',' + str(gaussV[0]) + ',' + str(gaussV[1]) + ','
    msk,time=SL.colorSegmentation_GenericSingle(candidate, colorScheme[0], colorScheme[1])#RGB Channel Segmentation
    gaussV =ScanFunc(msk[:,:,2])
    ary += repr(time) + ',' + str(gaussV[0]) + ',' + str(gaussV[1]) + ','
    msk,time=SL.colorSegGrid(candidate, colorScheme[0], colorScheme[1])#Grid RGB Channel Scanline
    gaussV =ScanFunc(msk[:,:,2])
    ary += repr(time) + ',' + str(gaussV[0]) + ',' + str(gaussV[1]) + ','
    msk,time=SL.getVerticalScanLines_buildMaskSingleImage(candidate, colorScheme[0], colorScheme[1])#Vertical RGB Channel Scanline
    gaussV =ScanFunc(msk[:,:,2])
    ary += repr(time) + ',' + str(gaussV[0]) + ',' + str(gaussV[1]) + ','
    msk,time=SL.getHorizontalScanLines_buildMaskSingleImage(candidate, colorScheme[0], colorScheme[1])#Horizontal RGB Channel Scanline
    gaussV =ScanFunc(msk[:,:,2])
    ary += repr(time) + ',' + str(gaussV[0]) + ',' + str(gaussV[1]) + ','
    msk, time = SL.ball_bhuman_VerticalFieldSupressionDark(cv2.cvtColor(candidate, cv2.COLOR_BGR2HSV))  # Vertical RGB Channel Scanline
    gaussV = ScanFunc(msk)
    ary += repr(time) + ',' + str(gaussV[0]) + ',' + str(gaussV[1]) + ','
    msk, time = SL.ball_bhuman_VerticalFieldSupression(cv2.cvtColor(candidate, cv2.COLOR_BGR2HSV))
    gaussV = ScanFunc(msk)
    ary += repr(time) + ',' + str(gaussV[0]) + ',' + str(gaussV[1]) + ','+str(count)+'\n'
    #write row to ods file

    writeToCSV(ary,csvFile)

    return

def VerificationExamine(im,x):
    q=SL.ball_bhuman_VerticalFieldSupression(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))[0]
    #Area,Classification,Speed,BlobType
    y = BB.GetBallAreaDoG(q)
    g=y[0]
    blobtming=str(y[2])+','
    if len(g) > 0:
        for ballCan in g:
            roi=im[ballCan[1]:ballCan[1]+ballCan[3],ballCan[0]:ballCan[0]+ballCan[2]]
            #cv2.imshow('',roi)
            #cv2.waitKey(2020202)
            '''output = bC.HAAR_Classifier(roi)#output in format of area, result , time, tag,frame
            writeToCSV(str('[' + ballCan[0] + ' ' + ballCan[1] + ' ' + ballCan[2] + ' ' + ballCan[3] + ']') + ',' + str(
                output[0]) + ',' + str(output[1]) + 'DoG_HAAR' + ',' + x + '\n', 'bVerificationExamination.csv')'''
            output = bC.bhumanInteriorExamination(roi)
            writeToCSV('[' + str(ballCan[0]) + ' ' + str(ballCan[1]) + ' ' + str(ballCan[2]) + ' ' + str(ballCan[3]) + ']' + ',' + str(
                output[0]) + ',' + str(output[1]) + ',DoG_bhuman' + ',' + x + '\n', 'bVerificationExamination.csv')

            output = bC.ContrastThresholdEx(roi)
            writeToCSV('[' + str(ballCan[0]) + ' ' + str(ballCan[1]) + ' ' + str(ballCan[2]) + ' ' + str(ballCan[3]) + ']' + ',' + str(
                output[0]) + ',' + str(output[1]) + ',DoG_Constrast' + ',' + x + '\n', 'bVerificationExamination.csv')

            output = bC.HoG_Classifier(cv2.cvtColor(cv2.resize(roi,(40,40)),cv2.COLOR_BGR2GRAY))
            writeToCSV('[' + str(ballCan[0]) + ' ' + str(ballCan[1]) + ' ' + str(ballCan[2]) + ' ' + str(ballCan[3]) + ']' + ',' + str(
                output[0]) + ',' + str(output[1]) + ',DoG_HoG' + ',' + x + '\n', 'bVerificationExamination.csv')

            output = bC.ConvoloutionalNeuralNetwork(cv2.cvtColor(cv2.resize(roi,(40,40)),cv2.COLOR_BGR2GRAY))
            writeToCSV('[' + str(ballCan[0]) + ' ' + str(ballCan[1]) + ' ' + str(ballCan[2]) + ' ' + str(ballCan[3]) + ']' + ',' + str(
                output[0]) + ',' + str(output[1]) + ',DoG_CNN' + ',' + x + '\n', 'bVerificationExamination.csv')

    y = BB.GetBallAreaLaplacian(q)
    g = y[0]
    blobtming += str(y[2])+'\n'
    if len(g) > 0:
        for ballCan in g:
            roi=im[ballCan[1]:ballCan[1]+ballCan[3],ballCan[0]:ballCan[0]+ballCan[2]]
            #cv2.imshow('',roi)
            #cv2.waitKey(2020202)
            '''output = bC.HAAR_Classifier(roi)#output in format of time , True/False
            writeToCSV(str('[' + ballCan[0] + ' ' + ballCan[1] + ' ' + ballCan[2] + ' ' + ballCan[3] + ']') + ',' + str(
                output[0]) + ',' + str(output[1]) + ',Lap_HAAR' + ',' + x + '\n', 'bVerificationExamination.csv')'''
            output = bC.bhumanInteriorExamination(roi)
            writeToCSV('[' + str(ballCan[0]) + ' ' + str(ballCan[1]) + ' ' + str(ballCan[2]) + ' ' + str(ballCan[3]) + ']' + ',' + str(
                output[0]) + ',' + str(output[1]) + ',Lap_bhuman' + ',' + x + '\n', 'bVerificationExamination.csv')

            output = bC.ContrastThresholdEx(roi)
            writeToCSV('[' + str(ballCan[0]) + ' ' + str(ballCan[1]) + ' ' + str(ballCan[2]) + ' ' + str(ballCan[3]) + ']' + ',' + str(
                output[0]) + ',' + str(output[1]) + ',Lap_Constrast' + ',' + x + '\n', 'bVerificationExamination.csv')

            output = bC.HoG_Classifier(cv2.cvtColor(cv2.resize(roi,(40,40)),cv2.COLOR_BGR2GRAY))
            writeToCSV('[' + str(ballCan[0]) + ' ' + str(ballCan[1]) + ' ' + str(ballCan[2]) + ' ' + str(ballCan[3]) + ']' + ',' + str(
                output[0]) + ',' + str(output[1]) + ',Lap_HoG' + ',' + x + '\n', 'bVerificationExamination.csv')

            output = bC.ConvoloutionalNeuralNetwork(cv2.cvtColor(cv2.resize(roi,(40,40)),cv2.COLOR_BGR2GRAY))
            writeToCSV('[' + str(ballCan[0]) + ' ' + str(ballCan[1]) + ' ' + str(ballCan[2]) + ' ' + str(ballCan[3]) + ']' + ',' + str(
                output[0]) + ',' + str(output[1]) + ',Lap_CNN' + ',' + x + '\n', 'bVerificationExamination.csv')

    writeToCSV(blobtming,'bBlobTiming.csv')
    return

def writeToCSV(string,odsFile):
    fd = open(odsFile, 'a')
    fd.write(string)
    fd.close()
    return

def ScanFunc(mask):
    return len(BB.GetBallAreaDoG(mask)[0]),len(BB.GetBallAreaLaplacian(mask)[0])

ballcolor=[0,0,0],[30,30,30]
def RunExamination():
    print pywt.__version__,cwt.__name__
    cnt=0

    #../../Data/Frames/
    import glob,os
    dataset = glob.glob(os.path.join('AnalysisData/', '*.jpg'))
    for x in dataset:
        #print x
        #cv2.imshow('',cv2.inRange(cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2HSV), np.array([42, 130, 50], dtype="uint16"),np.array([58, 256, 180], dtype="uint16")))
        #cv2.waitKey(202020202)
        loopexmination(ballcolor, cv2.imread(x), 'bPreprocessingSegmentation.csv', x)
        VerificationExamine(cv2.imread(x), x)
        print cnt,x
        cnt+=1
    return


RunExamination()
