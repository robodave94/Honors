import numpy as np
import cv2
import timeit
BALL_MAX = 75
BALL_MIN = 10


#region color based segmentation
def getVerticalScanLines_buildMaskSingleImage(image, lower, upper, percentage_dist=0.008):
    start_time = timeit.default_timer()
    x = 0.001
    mask = np.zeros((image.shape[0],image.shape[1], 3), np.uint8)
    while x < 1:
        index = int(image.shape[1] * x) + 1
        currentLine = cv2.inRange(image[0:image.shape[0], index - 1:index], np.array(lower, dtype="uint16"),
                                  np.array(upper, dtype="uint16"))
        output = cv2.bitwise_and(image[0:image.shape[0], index - 1:index], image[0:image.shape[0], index - 1:index], mask=currentLine)
        mask[0:image.shape[0], index - 1:index] = output
        x += percentage_dist
    return mask,(timeit.default_timer() - start_time)


def getHorizontalScanLines_buildMaskSingleImage(image, lower, upper, percentage_dist=0.008):
    start_time = timeit.default_timer()
    x = 0.001
    mask = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    while x < 1:
        index = int(image.shape[0] * x) + 1
        currentLine = cv2.inRange(image[index - 1:index, 0:image.shape[1]], np.array(lower, dtype="uint16"),
                                  np.array(upper, dtype="uint16"))
        output = cv2.bitwise_and(image[index - 1:index, 0:image.shape[1]], image[index - 1:index, 0:image.shape[1]],
                                 mask=currentLine)
        mask[index - 1:index, 0:image.shape[1]] = output
        x += percentage_dist
    return mask,(timeit.default_timer() - start_time)


def colorSegGrid(img, lwr, uppr,percentage_dist=0.008):
    start_time = timeit.default_timer()
    x = 0.001
    mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    while x < 1:
        index = int(img.shape[0] * x) + 1
        currentLine = cv2.inRange(img[index - 1:index, 0:img.shape[1]], np.array(lwr, dtype="uint16"),
                                  np.array(uppr, dtype="uint16"))
        output = cv2.bitwise_and(img[index - 1:index, 0:img.shape[1]], img[index - 1:index, 0:img.shape[1]],
                                 mask=currentLine)
        mask[index - 1:index, 0:img.shape[1]] = output

        index = int(img.shape[1] * x) + 1
        currentLine = cv2.inRange(img[0:img.shape[0], index - 1:index], np.array(lwr, dtype="uint16"),
                                  np.array(uppr, dtype="uint16"))
        output = cv2.bitwise_and(img[0:img.shape[0], index - 1:index], img[0:img.shape[0], index - 1:index],
                                 mask=currentLine)
        mask[0:img.shape[0], index - 1:index] = output

        x += percentage_dist
    return mask,(timeit.default_timer() - start_time)


# Color segmentation
def colorSegmentation_GenericSingle(image, lower, upper):
    start_time = timeit.default_timer()
    mask = cv2.inRange(image,np.array(lower, dtype="uint8"),np.array(upper, dtype="uint8"))
    output = cv2.bitwise_and(image, image, mask=mask)
    return output,(timeit.default_timer() - start_time)

#endregion

#region binary scanlines
def binary_getVerticalScanLines_buildMaskSingleImage(image, lower, upper, percentage_dist=0.008):
    start_time = timeit.default_timer()
    x = 0.001
    mask = np.zeros((image.shape[0],image.shape[1]), np.uint8)
    while x < 1:
        index = int(image.shape[1] * x) + 1
        currentLine = cv2.inRange(image[0:image.shape[0], index - 1:index], np.array(lower, dtype="uint16"),
                                  np.array(upper, dtype="uint16"))
        mask[0:image.shape[0], index - 1:index] = currentLine
        x += percentage_dist
    return mask,(timeit.default_timer() - start_time)

def binary_getHorizontalScanLines_buildMaskSingleImage(image, lower, upper, percentage_dist=0.008):
    start_time = timeit.default_timer()
    x = 0.001
    mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    while x < 1:
        index = int(image.shape[0] * x) + 1
        currentLine = cv2.inRange(image[index - 1:index, 0:image.shape[1]], np.array(lower, dtype="uint16"),
                                  np.array(upper, dtype="uint16"))
        mask[index - 1:index, 0:image.shape[1]] = currentLine
        x += percentage_dist
    return mask,(timeit.default_timer() - start_time)

def binary_colorSegGrid(img, lwr, uppr,percentage_dist=0.008):
    start_time = timeit.default_timer()
    x = 0.001
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    while x < 1:
        index = int(img.shape[0] * x) + 1
        currentLine = cv2.inRange(img[index - 1:index, 0:img.shape[1]], np.array(lwr, dtype="uint16"),
                                  np.array(uppr, dtype="uint16"))
        mask[index - 1:index, 0:img.shape[1]] = currentLine

        index = int(img.shape[1] * x) + 1
        currentLine = cv2.inRange(img[0:img.shape[0], index - 1:index], np.array(lwr, dtype="uint16"),
                                  np.array(uppr, dtype="uint16"))
        mask[0:img.shape[0], index - 1:index] = currentLine

        x += percentage_dist
    return mask,(timeit.default_timer() - start_time)

def binary_colorSegmentation_GenericSingle(image, lower, upper):
    start_time = timeit.default_timer()
    mask = cv2.inRange(image,np.array(lower, dtype="uint8"),np.array(upper, dtype="uint8"))
    return mask,(timeit.default_timer() - start_time)

#endregion

def ball_bhuman_VerticalFieldSupressionDark(image, percentage_dist=0.008):
    start_time = timeit.default_timer()
    x = 0.001
    mask = np.zeros((image.shape[0],image.shape[1]), np.uint8)
    while x < 1:
        index = int(image.shape[1] * x) + 1
        currentLine = cv2.inRange(image[0:image.shape[0], index - 1:index], np.array([42,130,50], dtype="uint16"),np.array([58,256,180], dtype="uint16"))
        #cv2.bitwise_not(cv2.inRange(image[0:image.shape[0], index - 1:index], np.array([2, 53, 19], dtype="uint16"),np.array([48, 180, 92], dtype="uint16")),None,None)
        itemIndex = np.where(currentLine == 0)
        breaks = list(np.arange(len(itemIndex[0]) - 1)[np.diff(itemIndex[0]) != 1] + 1)
        slices = zip([0] + breaks, breaks + [len(itemIndex[0])])
        gaps= [itemIndex[0][a:b] for a, b in slices if b - a > BALL_MIN and b - a < BALL_MAX]
        #gaps = split_list(itemIndex[0])
        for candidate in gaps:
            if len(np.where(cv2.inRange(image[candidate[0]:candidate[len(candidate)-1], index - 1:index],np.array([0,0,0], dtype="uint16")
                ,np.array([30,30,30], dtype="uint16"))==255)[0]):
                mask[candidate[0]:candidate[len(candidate)-1], index - 1:index].fill(255)
        x += percentage_dist
    return mask,(timeit.default_timer() - start_time)


def ball_bhuman_VerticalFieldSupression(image, percentage_dist=0.008):
    #[42,130,50],[58,256,180]
    start_time = timeit.default_timer()
    x = 0.001
    mask = np.zeros((image.shape[0],image.shape[1]), np.uint8)
    while x < 1:
        index = int(image.shape[1] * x) + 1
        currentLine = cv2.inRange(image[0:image.shape[0], index - 1:index], np.array([42,130,50], dtype="uint16"),np.array([58,256,180], dtype="uint16"))
        #currentLine=cv2.bitwise_not(cv2.inRange(image[0:image.shape[0], index - 1:index], np.array([2, 53, 19], dtype="uint16"),np.array([48, 180, 92], dtype="uint16")),None,None)
        #cv2.imshow('',currentLine)
        #cv2.waitKey(2020202)
        itemIndex = np.where(currentLine == 0)
        breaks = list(np.arange(len(itemIndex[0]) - 1)[np.diff(itemIndex[0]) != 1] + 1)
        slices = zip([0] + breaks, breaks + [len(itemIndex[0])])
        gaps = [itemIndex[0][a:b] for a, b in slices if b - a > BALL_MIN and b - a < BALL_MAX]
        # gaps = split_list(itemIndex[0])
        for candidate in gaps:
            mask[candidate[0]:candidate[len(candidate) - 1], index - 1:index].fill(255)
        x += percentage_dist
    return mask,(timeit.default_timer() - start_time)

#cv2.imshow('',ball_bhuman_VerticalFieldSupression(cv2.imread('_screenshot_03.10.2017.png'),0.03))
#cv2.waitKey(2000000)
#binary_colorSegmentation_GenericSingle(cv2.imread('_screenshot_03.10.2017.png'),[2, 53, 19],[48, 180, 92])