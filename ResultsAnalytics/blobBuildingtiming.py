import csv
import numpy as np

DoG=[]
Lap=[]

with open('Resultsball/bBlobTiming.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:

        try:
            DoG.append(float(row[0]))
            Lap.append(float(row[1]))
        except:
            print row
with open('V1goalResults/gBlobTiming.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:

        try:
            DoG.append(float(row[0]))
            Lap.append(float(row[1]))
        except:
            print row
print np.mean(DoG)
print np.mean(Lap)