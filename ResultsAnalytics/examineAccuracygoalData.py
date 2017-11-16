import csv
import numpy as np


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