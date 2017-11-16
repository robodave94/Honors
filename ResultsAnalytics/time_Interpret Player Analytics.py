import csv
import numpy as np

#region preprocessingvariables
class Preprocessingstruct:
    def __init__(self, var1, var2,var3,var4,var5,var6,
                 var7,var8,var9,var10,
                 var11, var12, var13, var14,var15,
                 var16,var17,var18,var19,var20,var21,
                 var22,var23,var24,var25,var26,var27):
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
        self.HorizontalFieldSupression=var25
        self.HorizontalFieldSupression_DoG=var26
        self.preFrame=var27
        return


class VerificationAnalysisStruct:
    def __init__(self, var1, var2,var3,var4,var5,var6,
                 var7,var8):
        self.Area=var1
        self.Time=var2
        self.Tag=var3
        self.valFrame=var4
        self.HoG_Validation=var5
        self.HoG_Time=var6
        self.CNN_Validation=var7
        self.CNN_Time=var8
        return
#endregion

def gtData():
    with open('resultsplayer/pPreprocessingSegmentation.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            #print ', '.join(row)
            try:
                e = Preprocessingstruct(float(row[0]),row[1],int(row[2]),float(row[3]),
                                        int(row[4]),int(row[5]),float(row[6]),int(row[7]),
                                        int(row[8]),float(row[9]),int(row[10]),int(row[11]),float(row[12]),
                                        int(row[13]),int(row[14]),float(row[15]),int(row[16]),int(row[17]),
                                        float(row[18]),int(row[19]),int(row[20]),float(row[21]),int(row[22]),
                                        int(row[23]),float(row[24]),int(row[25]),str(row[26]))
                PreprcfArr.append(e)
            except:
                print 'err'
    with open('resultsplayer/pVerificationExamination.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            #print ', '.join(row)
            try:
                if str(row[0]).__contains__('['):
                    ara=str(row[0]).split(' ')
                    c = VerificationAnalysisStruct([int(ara[0][1:]),int(ara[1]),int(ara[2]),int(ara[3][:-1])],
                                                   float(row[1]),str(row[2]),str(row[3]),str(row[4]),float(row[5]),
                                                   str(row[6]),float(row[7]))
                    verifArr.append(c)
            except:
                print 'err'

def dispAv():
    #region preprocessingTiming
    SingleChannelSegmentation=[]
    GridSingleChannelScanline=[]
    VerticalSingleChannelScanline=[]
    HorizontalSingleChannelScanline=[]
    HorizontalFieldSupression=[]
    RGBChannelSegmentation=[]
    GridRGBChannelScanline=[]
    VerticalRGBChannelScanline=[]
    HorizontalRGBChannelScanline=[]
    for c in PreprcfArr:
        SingleChannelSegmentation.append(c.SingleChannelSegmentation)
        GridSingleChannelScanline.append(c.GridSingleChannelScanline)
        VerticalSingleChannelScanline.append(c.VerticalSingleChannelScanline)
        HorizontalSingleChannelScanline.append(c.HorizontalSingleChannelScanline)
        HorizontalFieldSupression.append(c.HorizontalFieldSupression)
        RGBChannelSegmentation.append(c.RGBChannelSegmentation)
        GridRGBChannelScanline.append(c.GridRGBChannelScanline)
        VerticalRGBChannelScanline.append(c.VerticalRGBChannelScanline)
        HorizontalRGBChannelScanline.append(c.HorizontalRGBChannelScanline)
    print 'SingleChannelSegmentation',np.average(SingleChannelSegmentation)
    print 'GridSingleChannelScanline',np.average(GridSingleChannelScanline)
    print 'VerticalSingleChannelScanline',np.average(VerticalSingleChannelScanline)
    print 'HorizontalSingleChannelScanline',np.average(HorizontalSingleChannelScanline)
    print 'HorizontalFieldSupression',np.average(HorizontalFieldSupression)
    print 'RGBChannelSegmentation',np.average(RGBChannelSegmentation)
    print 'GridRGBChannelScanline',np.average(GridRGBChannelScanline)
    print 'VerticalRGBChannelScanline',np.average(VerticalRGBChannelScanline)
    print 'HorizontalRGBChannelScanline',np.average(HorizontalRGBChannelScanline)

    HoGTmng=[]
    CNNTmng=[]
    recursiveClustering=[]
    playerSuppMarkers=[]

    for c in verifArr:
        if c.Tag=='Clustering':
            recursiveClustering.append(c.Time)
        else:
            playerSuppMarkers.append(c.Time)
        HoGTmng.append(c.HoG_Time)
        CNNTmng.append(c.CNN_Time)

    print 'ClusteringTime',np.average(recursiveClustering)
    print 'FieldSupression',np.average(playerSuppMarkers)
    print 'HoGTime',np.average(HoGTmng)
    print 'CNN_Time',np.average(CNNTmng)
    return


def pltFreq():
    #for c in verifArr:
    #    if c.Tag=='Clustering':
    return

PreprcfArr=[]
verifArr=[]
gtData()
dispAv()
pltFreq()



import cv2
strlst=[]
truecnt=0
falsecnt=0
cnt = 0
count = 0
for x in verifArr:
    if not strlst.__contains__(x.valFrame):
        strlst.append(x.valFrame)
        recView=cv2.imread(x.valFrame)
        test=[]
        for c in verifArr:
            if x.valFrame==c.valFrame:


                    #print c.Goal_Area, c.Classification, c.Tag
                if str(c.Tag) == 'FiedSuppGen':
                    cv2.rectangle(recView, (c.Area[0], c.Area[1]),
                                 (c.Area[0] + c.Area[2], c.Area[1] + c.Area[3]), (255,255,255), 1)
                    #truecnt+=1
                #else:
                #    cv2.rectangle(recView, (c.Area[0], c.Area[1]),
                #                    (c.Area[0] + c.Area[2], c.Area[1] + c.Area[3]), (100,100,255), 1)
                #    falsecnt+=1
                    cnt+=1
                print cnt,truecnt,falsecnt

        #cv2.imshow('',recView)
        #cv2.waitKey(2020202020)
        count += 1
        if count > 150:
            break
        print cnt,truecnt,falsecnt
print cnt