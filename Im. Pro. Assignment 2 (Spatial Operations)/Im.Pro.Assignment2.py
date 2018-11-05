#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 10:25:39 2018

@author: vogebr01ith
"""

import skimage
import os
from skimage import io
from matplotlib import pyplot as plt
import numpy as np

os.chdir('/Users/brodyvogel/Desktop/COSC 455/')

from Im_Pro_Assignment_2_Functions import *

# 1
drip = io.imread('/Users/brodyvogel/Desktop/images/drip-bottle.tif')
fig, ax = plt.subplots(4, 2, figsize = (12, 12))
plt.suptitle('Figure 2.24')
for bit in [8, 7, 6, 5, 4, 3, 2, 1]:
    plt.subplot(4, 2, 9-bit)
    plt.text(0, -20, str(2**bit) + 'Intensities')
    drip_to_show = intensityQuan(drip, bit)
    plt.imshow(drip_to_show, cmap = 'gray', vmin = 0, vmax = 255)
    
# 2 
chron = io.imread('/Users/brodyvogel/Desktop/images/Chronometer.tif')
fig, ax = plt.subplots(1, 2, figsize = (10, 10))
plt.suptitle('Figure 2.38')
plt.subplot(1, 2, 1)
plt.text(0, -20, 'Original Image')
plt.imshow(chron, cmap = 'gray', vmin = 0, vmax = 255)
plt.subplot(1, 2, 2)
plt.text(0, -20, 'Negative')
chron_neg = negative(chron)
plt.imshow(chron_neg, cmap = 'gray', vmin = 0, vmax = 255)

    
# 3
# read in images
beach = io.imread('/Users/brodyvogel/Desktop/images/beach.tif')
ship = io.imread('/Users/brodyvogel/Desktop/images/ship.tif')
# create list of blended images
blended_imgs = blend(ship, beach, .7, 10)
# set the plot size to 8 rows of 2; make figure size a little bigger, too
fig, ax = plt.subplots(5, 2, figsize = (10, 10))
# give each image some headspace so they don't overlap
plt.subplots_adjust(hspace = .7)
# super title
plt.suptitle('Ship Blended to Beach')
# set subplot index
where = 1
# plot the iterations
for img in blended_imgs:
    # specify the position
    plt.subplot(5, 2, where)
    # title the subplot
    plt.text(0, -20, 'Iteration: ' + str(where))
    # show the image grayscale
    plt.imshow(img, cmap = 'gray')
    # increment the subplot index
    where += 1
    
# 4a
# The image is far too dark, as most of the intensities cluster around 0.
    # To combat this, I used histogram equalization to get a more dispersed
    # set of intensity values.
horse = io.imread('/Users/brodyvogel/Desktop/images/horse.tif')
fig, ax = plt.subplots(2, 2, figsize = (10, 10))
plt.suptitle('Histogram Equalized Horse')
plt.subplot(2, 2, 1)
plt.text(0, -20, 'Original Image')
plt.imshow(horse, cmap = 'gray', vmin = 0, vmax = 255)
plt.subplot(2, 2, 2)
plt.text(0, 125000, 'Original Histogram')
plt.hist(horse.flatten(), bins = 256, range = (0, 255))
new_horse = histEq(horse)
plt.subplot(2, 2, 3)
plt.text(0, -20, 'Histogram Equalized Image')
plt.imshow(new_horse, cmap = 'gray', vmin = 0, vmax = 255)
plt.subplot(2, 2, 4)
plt.text(0, 125000, 'New Histogram')
plt.hist(new_horse.flatten(), bins = 256, range = (0, 255))

# 4b
# The image is a bit too dark and definitely not spread enough throughout
    # the dynamic range. So, first, I implemented a slight gamma transformation
    # to lighten the image and then used contrast stretching to make better use
    # of the dynamic range.
einstein = io.imread('/Users/brodyvogel/Desktop/images/Einstein.tif')
fig, ax = plt.subplots(2, 2, figsize = (10, 10))
plt.suptitle('Gamma (.7) Transformed and Contrast Stretched Einstein')
plt.subplot(2, 2, 1)
plt.text(0, -20, 'Original Image')
plt.imshow(einstein, cmap = 'gray', vmin = 0, vmax = 255)
plt.subplot(2, 2, 2)
plt.text(0, 46000, 'Original Histogram')
plt.hist(einstein.flatten(), bins = 256, range = (0, 255))
new_einstein = gammaTrans(einstein, .7)
new_einstein = contStretch(new_einstein)
plt.subplot(2, 2, 3)
plt.text(0, -20, 'Gamma Transformed, Contrast Stretched Image')
plt.imshow(new_einstein, cmap = 'gray', vmin = 0, vmax = 255)
plt.subplot(2, 2, 4)
plt.text(0, 200000, 'New Histogram')
plt.hist(new_einstein.flatten(), bins = 256, range = (0, 255))

# 4c
# The image uses far too little of the dynamic range. To fix this,
    # I used contrast stretching with the normal values of min = 0 and max = 255. 
pollen = io.imread('/Users/brodyvogel/Desktop/images/pollen.tif')
fig, ax = plt.subplots(2, 2, figsize = (10, 10))
plt.suptitle('Contrast Stretched Pollen')
plt.subplot(2, 2, 1)
plt.text(0, -20, 'Original Image')
plt.imshow(pollen, cmap = 'gray', vmin = 0, vmax = 255)
plt.subplot(2, 2, 2)
plt.text(0, 21000, 'Original Histogram')
plt.hist(pollen.flatten(), bins = 256, range = (0, 255))
new_pollen = contStretch(pollen)
plt.subplot(2, 2, 3)
plt.text(0, -20, 'Contrast Stretched Image')
plt.imshow(new_pollen, cmap = 'gray', vmin = 0, vmax = 255)
plt.subplot(2, 2, 4)
plt.text(0, 21000, 'New Histogram')
plt.hist(new_pollen.flatten(), bins = 256, range = (0, 255))

# 4d
# The image is too bright and is not well-dispersed in the dynamic range.
    # So, to remedy this, I used a gamma transform to darken the image a bit,
    # and then used histogram equalization to better use the dynamic range.
balcony = io.imread('/Users/brodyvogel/Desktop/images/balcony.jpg')
fig, ax = plt.subplots(2, 2, figsize = (10, 10))
plt.suptitle('Gamma Transformed, Histogram Equalized Balcony')
plt.subplot(2, 2, 1)
plt.text(0, -20, 'Original Image')
plt.imshow(balcony, cmap = 'gray', vmin = 0, vmax = 255)
plt.subplot(2, 2, 2)
plt.text(0, 90000, 'Original Histogram')
plt.hist(balcony.flatten(), bins = 256, range = (0, 255))
new_balcony = gammaTrans(balcony, 2)
new_balcony = histEq(balcony).astype(np.uint8)
plt.subplot(2, 2, 3)
plt.text(0, -20, 'Gamma Transformed, Histogram Equalized Image')
plt.imshow(new_balcony, cmap = 'gray', vmin = 0, vmax = 255)
plt.subplot(2, 2, 4)
plt.text(0, 90000, 'New Histogram')
plt.hist(new_balcony.flatten(), bins = 256, range = (0, 255))

# 4e
# The image does not use enough of the dynamic range, and is a bit too bright.
    # To remedy this, I first used a slight gamma transform to spread the light
    # intensities out, and then used contrast stretching to make better use
    # of the dynamic range. After this, it looked a little too dark, so I used
    # histogram equalization to get a more dispersed set of intensity values.
bay = io.imread('/Users/brodyvogel/Desktop/images/bay.jpg')
fig, ax = plt.subplots(2, 2, figsize = (10, 10))
plt.suptitle('Gamma Transformed, Stretched, and Equalized Bay')
plt.subplot(2, 2, 1)
plt.text(0, -20, 'Original Image')
plt.imshow(bay, cmap = 'gray', vmin = 0, vmax = 255)
plt.subplot(2, 2, 2)
plt.text(0, 33000, 'Original Histogram')
plt.hist(bay.flatten(), bins = 256, range = (0, 255))
new_bay = gammaTrans(bay, 1.1)
new_bay = contStretch(new_bay)
new_bay = histEq(new_bay)
plt.subplot(2, 2, 3)
plt.text(0, -20, 'Gamma Transformed, Stretched, and Equalized Image')
plt.imshow(new_bay, cmap = 'gray', vmin = 0, vmax = 255)
plt.subplot(2, 2, 4)
plt.text(0, 33000, 'New Histogram')
plt.hist(new_bay.flatten(), bins = 256, range = (0, 255))

# 5
# The second circuit board is defective. There are three inconsistencies, as
    # shown below.
PCB1 = io.imread('/Users/brodyvogel/Desktop/images/pcb1.png')
PCB2 = io.imread('/Users/brodyvogel/Desktop/images/pcb2.png')
fig, ax = plt.subplots(1, 3, figsize = (10, 10))
plt.suptitle('Defective Circuit Board')
plt.subplot(1, 3, 1)
plt.text(0, -20, 'Good Circuit Board')
plt.imshow(PCB1, cmap = 'gray', vmin = 0, vmax = 255)
plt.subplot(1, 3, 2)
plt.text(0, -20, 'Bad Circuit Board')
plt.imshow(PCB2, cmap = 'gray', vmin = 0, vmax = 255)
plt.subplot(1, 3, 3)
diff = difference(PCB2, PCB1)
plt.text(0, -20, 'Defects')
plt.imshow(diff, cmap = 'gray', vmin = 0, vmax = 255)












