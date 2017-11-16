import numpy as np
import cv2
import ScanLines_Candidates
import scipy.ndimage
import timeit
#region Goal Blobs
def GoalLaplacianOfGaussians(image):
    lap = scipy.ndimage.filters.gaussian_laplace(image,[5,4])
    rslt = getPostAreas(lap)
    #dispGoalPosts(image,rslt)
    return rslt

def PlayerLaplacianOfGaussians(image):
    imshp=image.shape
    lap = scipy.ndimage.filters.gaussian_laplace(image,[5,4])
    return getPlayArea(lap)

def getPlayArea(input):
    nonzrs = np.array(np.nonzero(np.flipud(np.rot90(input))))
    # split array based on horizontal gaps
    subsets = np.hsplit(nonzrs, np.where(np.diff(nonzrs[0].tolist()) > 1)[0]+1)
    #print subsets
    returnVal=[]
    for x in subsets:
            print (min(x[0]),min(x[1]),max(x[0])-min(x[0]),max(x[1])-min(x[1]))
            returnVal.append((min(x[0]),min(x[1]),max(x[0])-min(x[0]),max(x[1])-min(x[1])))
    return returnVal

def PlayerDifferenceOfGaussians(image):
    g1 = cv2.GaussianBlur(image, (1, 1), 0, 0, 0)
    g2 = cv2.GaussianBlur(image, (9, 9), 0, 0, 0)
    diff = g2 - g1
    return getPlayArea(diff)

def GoalDifferenceOfGaussians(image):
    g1=cv2.GaussianBlur(image, (1,1),0, 0, 0)
    g2=cv2.GaussianBlur(image, (9,9), 0, 0, 0)
    diff = g2-g1
    #cv2.imshow('', diff)
    #cv2.waitKey(1010110)
    rslt = getPostAreas(diff)
    #dispGoalPosts(image,g2)
    return rslt

def dispGoalPosts(im,res):
    if len(res)==4:
        cv2.rectangle(im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (255, 255, 0), 1)
    else:
        cv2.rectangle(im, (res[0][0], res[0][1]), (res[0][0] + res[0][2], res[0][1] + res[0][3]), (255, 255, 0), 1)
        cv2.rectangle(im, (res[1][0], res[1][1]), (res[1][0] + res[1][2], res[1][1] + res[1][3]), (255, 255, 0), 1)
    cv2.imshow('',im)
    cv2.waitKey(1010110)
    return

def getPostAreas(inVal):
    thresh = cv2.threshold(inVal, 50, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort( key=cv2.contourArea)
    post1 = cv2.boundingRect(contours[len(contours) - 1])
    post2 = cv2.boundingRect(contours[len(contours) - 2])
    if (post2[2] / float(post2[3])) < 0.35:
        if post1[0] < post2[0]:
            return post1, post2
        else:
            return post2, post1

    return post1

def GetBallAreaLaplacian(ballCan):
    if type(ballCan[0,0]) is not np.uint8:
        ballCan=cv2.cvtColor(ballCan,cv2.COLOR_BGR2GRAY)
    start_time = timeit.default_timer()
    lap = scipy.ndimage.filters.gaussian_laplace(ballCan, [5, 4])
    returnTime = (timeit.default_timer() - start_time)
    returnVal = genBallAreas(lap)
    return returnVal, len(returnVal), returnTime

def GetBallAreaDoG(ballCan):
    if type(ballCan[0,0]) is not np.uint8:
        ballCan=cv2.cvtColor(ballCan,cv2.COLOR_BGR2GRAY)
    start_time = timeit.default_timer()
    g1 = cv2.GaussianBlur(ballCan, (1, 1), 0, 0, 0)
    g2 = cv2.GaussianBlur(ballCan, (15, 15), 0, 0, 0)
    diff = g2 - g1
    returnTime=(timeit.default_timer() - start_time)
    returnVal=genBallAreas(diff)
    return returnVal,len(returnVal),returnTime

def genBallAreas(mask):
    thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea)
    #todo fixe to return a series of bounding boxes
    ary=np.array([list(cv2.boundingRect(cnt)) for cnt in contours])

    #print ary
    #return loopFunc(ary)
    return repBox(ary)

def repBox(arry):
    for box in arry:
        #expand to square
        if float(box[2])/box[3]>1:
            #increase height
            box[1] = box[1]-(box[2]-box[3])/2
            box[3]=box[2]
        elif float(box[2])/box[3]<1:
            box[0] = box[0]-(box[3]-box[2])/2
            box[2]=box[3]
    return arry



#endregion
#input=cv2.imread('../../Data/Frames/frame116.jpg')
#binary = ScanLines_Candidates.binary_colorSegmentation_GenericSingle(input,[60,5,5],[255,50,50])
#cv2.imshow('',binary)
#cv2.waitKey(900000)
#PlayerLaplacianOfGaussians(binary)