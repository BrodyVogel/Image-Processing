#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 18:57:29 2018

@author: vogebr01
"""

import skimage
import os
from skimage import io
from matplotlib import pyplot as plt
import numpy as np

# read in the images
a = io.imread('/Users/brodyvogel/Desktop/homework3_images/a.tif')
camera = io.imread('/Users/brodyvogel/Desktop/homework3_images/camera.tif')

# function for showing 4 rescaled images accordng to some specified function
def showLoop(img, func):
    for scale in [.5, 1, 2, 4]:
        func(img, scale)

# the function for showing raw images
def showImageWithScale(image, title):
    DPI = 96
    H, W = image.shape
    figSize = H/float(DPI), W/float(DPI)    
    fig = plt.figure(title, figsize = figSize, dpi = DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(image, plt.cm.gray, vmax = 255, vmin = 0)

# basic rescaling by a factor    
def Basic(image, scalar):
    # dimensions of output image; 'ceil()' for the .5s
    new_x_dim = int(np.ceil(scalar * image.shape[0]))
    new_y_dim = int(np.ceil(scalar * image.shape[1]))
    # initialize new image as a matrix of 0s
    new_image = np.zeros((new_x_dim, new_y_dim), dtype = np.uint8)
    # fill in the new image with as many points as we can from the original image
    for x in range(image.shape[0] - 1):
        # again the 'ceil()'s are for the .5s
        x1 = int(np.ceil(scalar * x))
        for y in range(image.shape[1]-1):
            y1 = int(np.ceil(scalar * y))
            # fill in the new image
            new_image[x1, y1] = image[x, y]
    # show the image
    showImageWithScale(new_image, title = "Scaled by Basic Factor of: " + str(scalar))
# show 'em    
showLoop(a, Basic)
showLoop(camera, Basic)   

# single nearest-neighbor scaling
def NearN(image, scalar):
    # dimensions of output image; 'floor()' this time for same reason
    new_x_dim = int(np.floor(scalar * image.shape[0]))
    new_y_dim = int(np.floor(scalar * image.shape[1]))
    # initiate new image
    new_image = np.zeros((new_x_dim, new_y_dim), dtype = np.uint8)
    # now the loop is over the new image
        # it interpolates every pixel from the 
        # (left-justified) nearest neighbor in the original image
    for x in range(new_image.shape[0] - 1):
        # inverse transformation to get the x-coordinate from the source image
        x1 = int(np.floor((1/scalar) * x))
        for y in range(new_image.shape[1] - 1):
            # same inverse transformation to get y
            y1 = int(np.floor((1/scalar) * y))
            # fill in the new image
            new_image[x, y] = image[x1, y1]
    # show the image
    showImageWithScale(new_image, title = "Neared Neighbor Scaled by Factor of: " + str(scalar))
# show 'em
showLoop(a, NearN)
showLoop(camera, NearN)

# bilinear scaling
def Bilinear(image, scalar):
    # dimensions of output image
    new_x_dim = int(np.floor(scalar * image.shape[0]))
    new_y_dim = int(np.floor(scalar * image.shape[1]))
    # initiate new image
    new_image = np.zeros((new_x_dim, new_y_dim), dtype = np.uint8)
    # again, loop through the new image      # this clears some issues at the
                                            # edges of the image
    for x in range(new_image.shape[0] - max([(scalar-1), 1])):
        # inverse transform
        x_1 = (1/scalar) * x
        for y in range(new_image.shape[1] - max([(scalar-1), 1])):
            # the actual interpolation from the pixel's 4-neighbors
            y_1 = (1/scalar) * y
            x0 = int(np.floor(x_1))
            x1 = int(np.ceil(x_1))
            y0 = int(np.floor(y_1))
            y1 = int(np.ceil(y_1))
            A = image[x0, y0]
            B = image[x1, y0]
            C = image[x0, y1]
            D = image[x1, y1]
            alphax = x_1 - x0
            alphay = y_1 - y0
            new_image[x, y] = (1 - alphax) * (1 - alphay) * A + alphax * (1-alphay) * B + (1 - alphax) * alphay * C + alphax * alphay * D
    # show the image
    showImageWithScale(new_image, title = "Bilinear, Scaled by Factor of: " + str(scalar))
# show 'em
showLoop(a, Bilinear)
showLoop(camera, Bilinear)
            









     