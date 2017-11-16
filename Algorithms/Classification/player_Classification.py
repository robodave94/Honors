import numpy as np
import cv2
import timeit
import matplotlib.pyplot as plt
from skimage.feature import hog
import os, sys
from skimage import data, color, exposure

def kMeansClustering(Blobs):
    start_time = timeit.default_timer()


    return (timeit.default_timer() - start_time)

def player_HoG_Classifier(ROI):
    start_time = timeit.default_timer()
    #reshape the image to a standard size

    #execute HoG function on ROI
    fd = hog(ROI, orientations=8, pixels_per_cell=(6,6),
                        cells_per_block=(1, 1),block_norm='L2')

    #import and run classifier on output array

    print fd.shape
    return (timeit.default_timer() - start_time)
'''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(ROI, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()'''
    #build and run classifier


def Call_HAAR(ROI):
    start_time = timeit.default_timer()


    return (timeit.default_timer() - start_time)


#kMeansClustering(cv2.imread('../../training_DataSets/datasets/player/playerPositive/p70.png',cv2.CV_LOAD_IMAGE_GRAYSCALE))