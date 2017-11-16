from skimage.feature import hog
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import os
import random
import sys
import threading
import numpy as np
import tensorflow as tf
import cv2


dataset = glob.glob(os.path.join('train/player/','*.png'))
for x in dataset:
    fd = hog(cv2.imread(x), orientations=8, pixels_per_cell=(4,4),
                        cells_per_block=(1, 1),block_norm='L2')
    print fd


