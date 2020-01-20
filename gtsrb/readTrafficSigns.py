# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#            
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

import matplotlib.pyplot as plt
import csv
import os
import numpy as np
import cv2

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    imageHeight = 100
    imageWidth = 100
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = os.path.join(rootpath, format(c, '05d')) # subdirectory for class
        with open(os.path.join(prefix, 'GT-' + format(c, '05d') + '.csv')) as gtFile:
            gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
            next(gtReader) # skip header
            # loop over all images in current annotations file
            for row in gtReader:
                image = plt.imread(os.path.join(prefix, row[0]))
                image = cv2.resize(image, dsize=(imageWidth,imageHeight), 
                        interpolation=cv2.INTER_CUBIC)
                image = np.reshape(image, (1,imageWidth,imageHeight,3))
                images.append(image) # the 1th column is the filename
                labels.append(int(row[7])) # the 8th column is the label
    return np.concatenate(images, axis=0), np.array(labels)
