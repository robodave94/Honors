import csv
import numpy as np
import cv2
class Preprocessingstruct:
    def __init__(self, var1, var2,var3,var4,var5,var6,
                 var7,var8,var9,var10,
                 var11, var12, var13, var14,var15,
                 var16,var17,var18,var19,var20,var21,
                 var22,var23,var24,var25):
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
        self.preFrame=var25
        return

class VerificationAnalysisStruct:
    def __init__(self, var1, var2,var3,var4,var5):
        self.Goal_Area=var1
        self.Classification=var2
        self.Time=var3
        self.Tag=var4
        self.Frame=var5
        return

def gtData():
    with open('V1goalResults/gPreprocessingSegmentation.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            #print ', '.join(row)
            try:
                e = Preprocessingstruct(float(row[0]),row[1],int(row[2]),float(row[3]),
                                        int(row[4]),int(row[5]),float(row[6]),int(row[7]),
                                        int(row[8]),float(row[9]),int(row[10]),int(row[11]),float(row[12]),
                                        int(row[13]),int(row[14]),float(row[15]),int(row[16]),int(row[17]),
                                        float(row[18]),int(row[19]),int(row[20]),float(row[21]),int(row[22]),
                                        int(row[23]),str(row[24]))
                PreprcfArr.append(e)
            except:
                print 'err'
    with open('V1goalResults/gVerificationExamination.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            #print ', '.join(row)
            try:
                if str(row[0]).__contains__('['):
                    ara=str(row[0]).split(' ')
                    c = VerificationAnalysisStruct([int(ara[0][1:]),int(ara[1]),int(ara[2]),int(ara[3][:-1])],
                                                   str(row[1]),float(row[2]),str(row[3]),str(row[4]))
                    verifArr.append(c)
            except:
                print 'err'

    return

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
    for c in PreprcfArr:
        SingleChannelSegmentation.append(c.SingleChannelSegmentation)
        GridSingleChannelScanline.append(c.GridSingleChannelScanline)
        VerticalSingleChannelScanline.append(c.VerticalSingleChannelScanline)
        HorizontalSingleChannelScanline.append(c.HorizontalSingleChannelScanline)
        RGBChannelSegmentation.append(c.RGBChannelSegmentation)
        GridRGBChannelScanline.append(c.GridRGBChannelScanline)
        VerticalRGBChannelScanline.append(c.VerticalRGBChannelScanline)
        HorizontalRGBChannelScanline.append(c.HorizontalRGBChannelScanline)
    print 'SingleChannelSegmentation',np.average(SingleChannelSegmentation)
    print 'GridSingleChannelScanline',np.average(GridSingleChannelScanline)
    print 'VerticalSingleChannelScanline',np.average(VerticalSingleChannelScanline)
    print 'HorizontalSingleChannelScanline',np.average(HorizontalSingleChannelScanline)
    print 'RGBChannelSegmentation',np.average(RGBChannelSegmentation)
    print 'GridRGBChannelScanline',np.average(GridRGBChannelScanline)
    print 'VerticalRGBChannelScanline',np.average(VerticalRGBChannelScanline)
    print 'HorizontalRGBChannelScanline',np.average(HorizontalRGBChannelScanline)

    for c in verifArr:
        if str(c.Tag).__contains__('StatArea'):
            StatArea.append(c.Time)
        elif str(c.Tag).__contains__('PostTrans'):
            FieldTransition.append(c.Time)
        else:
            FuzzyLogic.append(c.Time)


    print 'StatArea', np.average(StatArea)
    print 'TransitionLines', np.average(FieldTransition)
    print 'FuzzyLogic', np.average(FuzzyLogic)
    return



def plotFrequencyOfBlobsDetected():
    def verifCan(input,arr):
        if str(input.Classification).__contains__('Partial'):
            arr.append(1)
        elif str(input.Classification).__contains__('Error'):
            arr.append(0)
        else:
            arr.append(2)
        return
    #region partail args
    partialconflicts=[
        5,9,10,20,21,22,24,25,26,27,34,40,41,42,43,54,63,65,67,70,
        79,80,81,88,91,92,99,100,102,109,116,119,117,118,121,124,129,
        141,142,143,144,145,146,163,164,165,166,168,171,176,180,188,
        189,191,193,197,206,207,209,210,227,228,232,262,265,266,268,
        274,276,277,278,281,286,287,292,295,297,305,306,307,308,311,
        314,329,331,338,339,340,350,358,359,360,361,362,363,366,369,
        374,379,380,382,385,410,418,420,430,435,436,439,440,441,442,
        450,451,453,454,455,456,466,473,474,477,478,481,490,495,495,
        496,498,499,502,505,510,513,518,522,523,526,528,527,529,533,
        533,535,539,540,541,544,545,547,553,554,561,564,565,579,580,
        584,604,608,609,610,615,627,641,645,646,647,649,650,653,654,
        655,656,657,658,659,660,661,662,663,666,667,676,679,680,681,
        682,683,687,689,692,693,694,695,696,704,705,707,708,716,717,
        721,725,726,729,730,735,736,739,740,741,742,743,744,745,751,
        756,761,762,763,764,765,766,769,770,777,782,795,797,800,803,807,808,817,825
        ,826,831,834,838
    ]
    #endregion
    stringList=[]
    Stat=[]
    Field = []
    Fuzzy = []
    conflicting=0
    agreement=0
    trueneg=0
    falsepos=0
    #informat of Stat Fuzzy,Field
    rat=[]#[True,True,True]

    for v in range(0,len(verifArr),3):
        if not verifArr[v].Classification==verifArr[v+1].Classification==verifArr[v+2].Classification:
            conflicting+=1
            #stringList.append(verifArr[v].Frame)
            #print verifArr[v].Classification,verifArr[v+1].Classification,verifArr[v+2].Classification
            #cv2.imshow('',cv2.imread(verifArr[v].Frame))
            #cv2.waitKey(20202020)
            verifCan(verifArr[v],Stat)
            verifCan(verifArr[v+1], Field)
            verifCan(verifArr[v+2], Fuzzy)
            '''print conflicting,verifArr[v].Frame
            trflse = []
            if partialconflicts.__contains__(conflicting):
                if Stat[len(Stat)-1]==1:
                    trflse.append(True)
                else:
                    trflse.append(False)

                if Fuzzy[len(Fuzzy)-1] == 1:
                    trflse.append(True)
                else:
                    trflse.append(False)

                if Field[len(Field)-1]==1:
                    trflse.append(True)
                else:
                    trflse.append(False)
                    falsepos += 1
            else:
                if Stat[len(Stat)-1]==2:
                    trflse.append(True)
                else:
                    trflse.append(False)

                if Fuzzy[len(Fuzzy)-1] == 2:
                    trflse.append(True)
                else:
                    trflse.append(False)

                if Field[len(Field)-1]==2:
                    trflse.append(True)
                else:
                    trflse.append(False)
                    trueneg += 1
            rat.append(trflse)'''
            #cv2.imshow('',cv2.imread(verifArr[v].Frame))
            #cv2.waitKey(2020202020)
        else:
            agreement+=1
            verifCan(verifArr[v], Stat)
            verifCan(verifArr[v + 1], Field)
            verifCan(verifArr[v + 2], Fuzzy)

    print Field.count(2),Field.count(1),Field.count(0)
    print 'conflict',conflicting,'agree',agreement



    import pylab

    counts = np.array(['Error','Partial','Complete'])

    stringList=np.array(stringList)
    Stat = np.array(Stat)
    Field = np.array(Field)
    Fuzzy = np.array(Fuzzy)

    pylab.figure(1)
    x = range(len(stringList))
    y=range(3)
    pylab.xticks(x, stringList)
    pylab.yticks(y, counts)
    pylab.plot(Stat)

    pylab.show()
StatArea = []
FuzzyLogic = []
FieldTransition = []
PreprcfArr=[]
verifArr=[]
gtData()
dispAv()
plotFrequencyOfBlobsDetected()