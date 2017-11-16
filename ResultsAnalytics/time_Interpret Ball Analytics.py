import csv
import numpy as np
import cv2
'''
Goal_Area
Time
Classification
Tag
Frame'''
'''
Single Channel Segmentation
Single Channel Segmentation_DoG
Single Channel Segmentation_Lap
Grid Single Channel Scanline
Grid Single Channel Scanline_DoG
Grid Single Channel Scanline_Lap
Vertical Single Channel Scanline
Vertical Single Channel Scanline_DoG
Vertical Single Channel Scanline_Lap
Horizontal Single Channel Scanline
Horizontal Single Channel Scanline_DoG
Horizontal Single Channel Scanline_Lap
RGB Channel Segmentation
RGB Channel Segmentation_DoG
RGB Channel Segmentation_Lap
Grid RGB Channel Scanline
Grid RGB Channel Scanline_DoG
Grid RGB Channel Scanline_Lap
Vertical RGB Channel Scanline
Vertical RGB Channel Scanline_DoG
Vertical RGB Channel Scanline_Lap
Horizontal RGB Channel Scanline
Horizontal RGB Channel Scanline_DoG
Horizontal RGB Channel Scanline_Lap
Vertical Field Gap with Dark Interior
Vertical Field Gap with Dark Interior_DoG
Vertical Field Gap with Dark Interior_Lap
Vertical Field Gaps,Vertical Field Gaps_DoG
Vertical Field Gaps_Lap
Frame'''

class Preprocessingstruct:
    def __init__(self, var1, var2,var3,var4,var5,var6,
                 var7,var8,var9,var10,
                 var11, var12, var13, var14,var15,
                 var16,var17,var18,var19,var20,var21,
                 var22,var23,var24,var25,var26,var27,var28,var29,var30,var31):
        self.SingleChannelSegmentation=var1
        self.SingleChannelSegmentation_DoG=var2
        self.SingleChannelSegmentation_Lap=var3
        self.GridSingleChannelScanline=var4
        self.GridSingleChannelScanline_DoG=var5
        self.GridSingleChannelScanline_Lap=var6
        self.VerticalSingleChannelScanline=var7
        self.VerticalSingleChannelScanline_DoG=var8
        self.VerticalSingleChannelScanline_Lap=var9
        self.HorizontalSingleChannelScanline=var10
        self.HorizontalSingleChannelScanline_DoG=var11
        self.HorizontalSingleChannelScanline_Lap=var12
        self.RGBChannelSegmentation=var13
        self.RGBChannelSegmentation_DoG=var14
        self.RGBChannelSegmentation_Lap=var15
        self.GridRGBChannelScanline=var16
        self.GridRGBChannelScanline_DoG=var17
        self.GridRGBChannelScanline_Lap=var18
        self.VerticalRGBChannelScanline=var19
        self.VerticalRGBChannelScanline_DoG=var20
        self.VerticalRGBChannelScanline_Lap=var21
        self.HorizontalRGBChannelScanline=var22
        self.HorizontalRGBChannelScanline_DoG=var23
        self.HorizontalRGBChannelScanline_Lap=var24
        self.VerticalFieldGapwithDarkInterior=var25
        self.VerticalFieldGapwithDarkInterior_DoG=var26
        self.VerticalFieldGapwithDarkInterior_Lap=var27
        self.VerticalFieldGaps=var28
        self.VerticalFieldGaps_DoG=var29
        self.VerticalFieldGaps_Lap=var30
        self.preFrame=var31
        return


class VerificationAnalysisStruct:
    def __init__(self, var1, var2,var3,var4,var5):
        self.Goal_Area=var1
        self.Time=var2
        self.Classification=var3
        self.Tag=var4
        self.Frame=var5
        return


def gtData():
    with open('Resultsball/bPreprocessingSegmentation.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            #print ', '.join(row)
            try:
                e = Preprocessingstruct(float(row[0]),row[1],int(row[2]),float(row[3]),
                                        int(row[4]),int(row[5]),float(row[6]),int(row[7]),
                                        int(row[8]),float(row[9]),int(row[10]),int(row[11]),float(row[12]),
                                        int(row[13]),int(row[14]),float(row[15]),int(row[16]),int(row[17]),
                                        float(row[18]),int(row[19]),int(row[20]),float(row[21]),int(row[22]),
                                        int(row[23]),float(row[24]),int(row[25]),int(row[26]),float(row[27]),int(row[28]),
                                        int(row[29]),str(row[30]))
                PreprcfArr.append(e)
            except:
                print 'err'
    with open('Resultsball/bVerificationExamination.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            #print ', '.join(row)
            try:
                if str(row[0]).__contains__('['):
                    ara=str(row[0]).split(' ')
                    c = VerificationAnalysisStruct([int(ara[0][1:]),int(ara[1]),int(ara[2]),int(ara[3][:-1])],
                                                   float(row[1]),str(row[2]),str(row[3]),str(row[4]))
                    verifArr.append(c)
            except:
                print 'err'

def dispAv():
    #region preprocessingTiming
    SingleChannelSegmentation=[]
    GridSingleChannelScanline=[]
    VerticalSingleChannelScanline=[]
    HorizontalSingleChannelScanline=[]
    RGBChannelSegmentation=[]
    GridRGBChannelScanline=[]
    VerticalRGBChannelScanline=[]
    HorizontalRGBChannelScanline=[]
    VerticalFieldGapwithDarkInterior=[]
    VerticalFieldGaps=[]

    for c in PreprcfArr:
        SingleChannelSegmentation.append(c.SingleChannelSegmentation)
        GridSingleChannelScanline.append(c.GridSingleChannelScanline)
        VerticalSingleChannelScanline.append(c.VerticalSingleChannelScanline)
        HorizontalSingleChannelScanline.append(c.HorizontalSingleChannelScanline)
        RGBChannelSegmentation.append(c.RGBChannelSegmentation)
        GridRGBChannelScanline.append(c.GridRGBChannelScanline)
        VerticalRGBChannelScanline.append(c.VerticalRGBChannelScanline)
        HorizontalRGBChannelScanline.append(c.HorizontalRGBChannelScanline)
        VerticalFieldGapwithDarkInterior.append(c.VerticalFieldGapwithDarkInterior)
        VerticalFieldGaps.append(c.VerticalFieldGaps)
    print 'SingleChannelSegmentation',np.average(SingleChannelSegmentation)
    print 'GridSingleChannelScanline',np.average(GridSingleChannelScanline)
    print 'VerticalSingleChannelScanline',np.average(VerticalSingleChannelScanline)
    print 'HorizontalSingleChannelScanline',np.average(HorizontalSingleChannelScanline)
    print 'RGBChannelSegmentation',np.average(RGBChannelSegmentation)
    print 'GridRGBChannelScanline',np.average(GridRGBChannelScanline)
    print 'VerticalRGBChannelScanline',np.average(VerticalRGBChannelScanline)
    print 'HorizontalRGBChannelScanline',np.average(HorizontalRGBChannelScanline)
    print 'VerticalFieldGapwithDarkInterior', np.average(VerticalFieldGapwithDarkInterior)
    print 'VerticalFieldGaps', np.average(VerticalFieldGaps)

    HoGTmng=[]
    CNNTmng=[]
    truecontrast=[]
    falsebhuman=[]
    falsecontrast = []
    truebhuman = []

    for c in verifArr:
        if str(c.Tag).__contains__('bhuman'):
            if str(c.Classification)=='True':
                truebhuman.append(c.Time)
            else:
                falsebhuman.append(c.Time)
        elif str(c.Tag).__contains__('Constrast'):
            if str(c.Classification)=='True':
                truecontrast.append(c.Time)
            else:
                falsecontrast.append(c.Time)
        elif str(c.Tag).__contains__('HoG'):
            HoGTmng.append(c.Time)
        else:
            CNNTmng.append(c.Time)

    print 'Truebhuman',np.average(truebhuman)
    print 'Truecontrast',np.average(truecontrast)
    print 'Falsebhuman', np.average(falsebhuman)
    print 'Falsecontrast', np.average(falsecontrast)
    print 'HoGTime',np.average(HoGTmng)
    print 'CNN_Time',np.average(CNNTmng)
    return


def pltFreq():

    return

PreprcfArr=[]
verifArr=[]
gtData()
#dispAv()
#pltFreq()
strlst=[]
truecnt=0
falsecnt=0
cnt = 0
count = 0
import ball_Classification
for x in verifArr:
    if not strlst.__contains__(x.Frame):
        strlst.append(x.Frame)
        recView=cv2.imread(x.Frame)
        test=[]
        for c in verifArr:
            if x.Frame==c.Frame:

                if str(c.Tag).__contains__('DoG_bhuman'):
                    #print c.Goal_Area, c.Classification, c.Tag
                    valud=ball_Classification.bhumanInteriorExamination(recView[c.Goal_Area[1]:c.Goal_Area[1]+c.Goal_Area[3],
                                                                        c.Goal_Area[0]:c.Goal_Area[0] + c.Goal_Area[2]])
                    print valud
                    if valud[1] == True:
                        cv2.rectangle(recView, (c.Goal_Area[0], c.Goal_Area[1]),
                                      (c.Goal_Area[0] + c.Goal_Area[2], c.Goal_Area[1] + c.Goal_Area[3]), (255,255,255), 1)
                        truecnt+=1
                    else:
                        cv2.rectangle(recView, (c.Goal_Area[0], c.Goal_Area[1]),
                                      (c.Goal_Area[0] + c.Goal_Area[2], c.Goal_Area[1] + c.Goal_Area[3]), (100,100,255), 1)
                        falsecnt+=1
                    cnt+=1
                    print cnt,truecnt,falsecnt

        #cv2.imshow('',recView)
        #cv2.waitKey(2020202020)
        count += 1
        if count > 150:
            break
        print cnt,truecnt,falsecnt
print cnt
