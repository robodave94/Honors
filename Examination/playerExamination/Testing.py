import numpy as np
import cv2
import playerBlobs as BB
import player_Classification as pC
import playerScanLines_Candidates as SL
import glob,os

def VerificationExamine(scm,im,x):
    w=cv2.inRange(cv2.cvtColor(im,cv2.COLOR_BGR2HSV),np.array(scm[0],dtype="uint16"),np.array(scm[1],dtype="uint16"))
    cv2.imshow('',w)
    cv2.waitKey(20202020)
    q=SL.player_HorizFieldSupression(cv2.cvtColor(im,cv2.COLOR_BGR2HSV))
    y = BB.PlayerFieldSupressionDiff(q[0])
    j = BB.PlayerDifferenceOfGaussians(w)
    g=y[0]
    if len(g) > 0:
        output = pC.Clustering(cv2.cvtColor(im, cv2.COLOR_BGR2HSV), j[0])
        for kk in output[0]:
            inptHoG = pC.HoG_Classifier(
                cv2.cvtColor(cv2.resize(im[kk[1]:kk[1] + kk[3], kk[0]:kk[0] + kk[2]], (40, 80)), cv2.COLOR_BGR2GRAY))
            inptCNN = pC.ConvoloutionalNeuralNetwork(
                cv2.cvtColor(cv2.resize(im[kk[1]:kk[1] + kk[3], kk[0]:kk[0] + kk[2]], (40, 64)), cv2.COLOR_BGR2GRAY))
            writeToCSV('[' + str(kk[0]) + ' ' + str(kk[1]) + ' ' + str(kk[2]) + ' ' + str(kk[3]) + ']' + ',' + str(
                output[1]) + ',' + 'Clustering' + ',' + x + ',' + str(inptHoG[1]) + ',' + \
                       str(inptHoG[0]) + ',' + str(inptCNN[1]) + ',' + str(inptCNN[0])+'\n', 'pVerificationExamination.csv')
        output = pC.FieldSuppFindVal(im, y[0])
        #print 'supp',output
        for kk in output[0]:
            inptHoG=pC.HoG_Classifier(cv2.cvtColor(cv2.resize(im[kk[1]:kk[1] + kk[3], kk[0]:kk[0] + kk[2]],(40,80)),cv2.COLOR_BGR2GRAY))
            inptCNN = pC.ConvoloutionalNeuralNetwork(cv2.cvtColor(cv2.resize(im[kk[1]:kk[1] + kk[3], kk[0]:kk[0] + kk[2]], (40, 64)), cv2.COLOR_BGR2GRAY))
            writeToCSV( '[' + str(kk[0]) + ' ' + str(kk[1]) + ' ' + str(kk[2]) + ' ' + str(kk[3]) + ']' + ',' + str(
                output[1]) + ',' + 'FiedSuppGen' + ',' + x + ',' + str(inptHoG[1]) + ',' + \
                  str(inptHoG[0]) + ',' + str(inptCNN[1]) + ',' + str(inptCNN[0])+'\n','pVerificationExamination.csv')

        '''
        if inpt[1]==True:
                cv2.rectangle(im,(u[0],u[1]),(u[0]+u[2],u[1]+u[3]),255,1)
        cv2.imshow('',im)
            #cv2.imshow('',cv2.cvtColor(cv2.resize(im[u[1]:u[1]+u[3],u[0]:u[0]+u[2]],(80,40)),cv2.COLOR_BGR2GRAY))
        cv2.waitKey(20202020)

        
        #writeToCSV(str('['+output[0][0]+' '+output[0][1]+' '+output[0][2]+' '+output[0][3]+']')+','+str(output[1])+','+str(output[2])+'DoG_StatAreas'+','+x+'\n',
        #           'pVerificationExamination.csv')[u[1]:u[1]+u[3],u[0]:u[0]+u[2]]
        #print 'Clus', output

        #for u in output[0]:
        #    cv2.imshow('',im[u[1]:u[1]+u[3],u[0]:u[0]+u[2]])
        #    cv2.waitKey(2020202020)
        output = pC.CNN(im,g)
        writeToCSV(str('['+output[0][0]+' '+output[0][1]+' '+output[0][2]+' '+output[0][3]+']') + ',' + str(output[1]) + ',' + str(output[2]) + 'DoG_FieldPostTrans' + ',' + x+'\n',
                   'pVerificationExamination.csv')
        output = pC.
        writeToCSV(str('['+output[0][0]+' '+output[0][1]+' '+output[0][2]+' '+output[0][3]+']') + ',' + str(output[1]) + ',' + str(output[2]) + 'DoG_FuzzyLogic' + ',' + x+'\n',
                   'pVerificationExamination.csv')'''
    '''y = BB.PlayerLaplacianOfGaussians(q)
    g = y[0]
    blobtming += str(y[1])+'\n'
    if len(g) > 0:
        output = pC.Clustering((im, g)
        writeToCSV(str('['+output[0][0]+' '+output[0][1]+' '+output[0][2]+' '+output[0][3]+']')+','+str(output[1])+','+str(output[2])+'Lap_StatAreas'+','+x+'\n','pVerificationExamination.csv')
        output = pC.Clustering((im,g)
        writeToCSV(str('['+output[0][0]+' '+output[0][1]+' '+output[0][2]+' '+output[0][3]+']') + ',' + str(output[1]) + ',' + str(output[2]) + 'Lap_FieldPostTrans' + ',' + x+'\n',
                   'pVerificationExamination.csv')
        output = pC.Clustering(im, g)
        writeToCSV(str('['+output[0][0]+' '+output[0][1]+' '+output[0][2]+' '+output[0][3]+']') + ',' + str(output[1]) + ',' + str(output[2]) + 'Lap_FuzzyLogic' + ',' + x+'\n',
                   'pVerificationExamination.csv')
    writeToCSV(blobtming,'pBlobTiming.csv')'''
    return

def ScanFunc(mask):
    return len(BB.PlayerDifferenceOfGaussians(mask)[0]),len(BB.PlayerLaplacianOfGaussians(mask)[0])

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
    msk,time=SL.colorSegmentation_GenericSingle(candidate, colorScheme[0], colorScheme[1])#RpB Channel Segmentation
    gaussV =ScanFunc(msk[:,:,2])
    ary += repr(time) + ',' + str(gaussV[0]) + ',' + str(gaussV[1]) + ','
    msk,time=SL.colorSegGrid(candidate, colorScheme[0], colorScheme[1])#Grid RpB Channel Scanline
    gaussV =ScanFunc(msk[:,:,2])
    ary += repr(time) + ',' + str(gaussV[0]) + ',' + str(gaussV[1]) + ','
    msk,time=SL.getVerticalScanLines_buildMaskSingleImage(candidate, colorScheme[0], colorScheme[1])#Vertical RpB Channel Scanline
    gaussV =ScanFunc(msk[:,:,2])
    ary += repr(time) + ',' + str(gaussV[0]) + ',' + str(gaussV[1]) + ','
    msk,time=SL.getHorizontalScanLines_buildMaskSingleImage(candidate, colorScheme[0], colorScheme[1])#Horizontal RpB Channel Scanline
    gaussV =ScanFunc(msk[:,:,2])
    ary += repr(time) + ',' + str(gaussV[0]) + ',' + str(gaussV[1]) + ','
    msk, time = SL.player_HorizFieldSupression(candidate)  # Horizontal RpB Channel Scanline
    gaussV = BB.PlayerFieldSupressionDiff(msk)
    ary += repr(time) + ','+ str(len(gaussV[0])) +','+str(count)+'\n'
    #write row to ods file
    writeToCSV(ary,csvFile)
    #VerificationExamine(colorScheme,candidate,count)
    return

def writeToCSV(string,odsFile):
    #print string
    fd = open(odsFile, 'a')
    fd.write(string)
    fd.close()
    return

blue_markers = [100, 180, 40], [130, 240, 255]
def RunExamination():
    #cnt=3125
    #../../Data/Frames/
    dataset = glob.glob(os.path.join('AnalysisData/', '*.jpg'))
    c=0
    for x in dataset:
        print c,x
        loopexmination(blue_markers, cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2HSV), 'pPreprocessingSegmentation.csv',x)
        VerificationExamine(blue_markers,cv2.imread(x),x)
        c+=1
        #cv2.imshow('',cv2.imread(x))
        #cv2.waitKey(20202020)
    #return


RunExamination()
'''dataset = glob.glob(os.path.join('../../Data/Frames/', '*.jpg'))
for x in dataset:
        q=cv2.inRange(cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2HSV),np.array(blue_markers[0],dtype="uint16"),np.array(blue_markers[1],dtype="uint16"))
        BB.PlayerDifferenceOfGaussians(q)
        #examine differing method'''