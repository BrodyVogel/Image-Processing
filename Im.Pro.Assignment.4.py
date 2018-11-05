#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 09:52:24 2018

@author: vogebr01
"""
# Brody Vogel Homework 4

################### Takeaways #########################################
# If no image processing is done in between the shrinking steps,      #
# there appears to be no difference between shrinking the image by a  #
# factor of 2 five times and shrinkng it by a factor of 32 once.      #
#                                                                     #
# If some image processing is done, though, the step at which it is   #
# done matters greatly. If blurring, for example, is done earlier,    #
# it has a greater impact on the final, shrunken image (I think)      #
# because the image has more pixels to do the blurring and so         #
# produces better images.                                             #
#                                                                     #
# The blurring filters all seemed to perform about the same, although #
# I'm sure the weighted box filter would've performed differently had #
# I used more extreme weights. So the weighted box filter, the median #
# filter, and the Gaussian filter all had similarly nice smoothing    #
# effects on the image. The sharpening Laplace filter, though, created#
# a mess; the final image had far too high contrast. Thus, it seems   #
# shrinking this image was best aided by a blurring filter applied at #
# an early stage of the shrinking process.                            #
#######################################################################

# import everything I'll need
import skimage
import os
from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import copy

# read in image
VG = io.imread('/Users/brodyvogel/Desktop/VanGogh.jpg')

# original large image
plt.figimage(VG)

# this is a standard sampling function for shrinking the image
def shrink(img, alpha):
    # build an empty matrix to hold the new image
    x = img.shape[0]
    y = img.shape[1]
    new_img = np.zeros((int(np.floor(x / alpha)), int(np.floor(y / alpha)), 3))
    # to deal with odd dimensions
    if np.floor(y/alpha) % 2 != 0:
        y = y - alpha
    if np.floor(x/alpha) % 2 != 0:
        x = x - alpha 
    # fill in the new image
    for z in range(0, x, alpha):
        for z1 in range(0, y, alpha):
            new_img[int(np.floor(z/alpha)), int(np.floor(z1/alpha))] = img[z, z1]
    # assign the new images the correct type
    return(new_img.astype('uint8'))

# this shrinks the image with simple down-sampling, one step at a time
half = shrink(VG, 2)
fourth = shrink(half, 2)
eighth = shrink(fourth, 2)
sixteenth = shrink(eighth, 2)
thirtysecond = shrink(sixteenth, 2)

# take a look at the down-sampling shrinking process
# the images get blurry and pretty unseemly around the 1/16th-sized image
for img in [half, fourth, eighth, sixteenth, thirtysecond]:
    plt.figimage(img)
    
# here, one can see that shrinking by a factor of 2 five times is the same 
# as shrinking by a factor of 32 once       
plt.figimage(shrink(VG, 32))

############### Gaussian Filter Experiments ##########################
# build the filter
unit = np.zeros((5, 5))
unit[2, 2] = 1
G_filt = ndi.gaussian_filter(unit, 3.5)

# function to help apply the filter
def Gaus_Filt(some_img):
    img = copy.deepcopy(some_img)
    img[:, :, 0] = ndi.convolve(img[:, :, 0], G_filt)
    img[:, :, 1] = ndi.convolve(img[:, :, 1], G_filt)
    img[:, :, 2] = ndi.convolve(img[:, :, 2], G_filt)
    return(img)

# try applying the filter at one of the final steps
# this looks pretty bad; I think too many pixels have already been lost, 
# so the effects of the blurring are limited
sixteenth_gaus = Gaus_Filt(sixteenth)
plt.figimage(sixteenth_gaus)
# here I try applying the filter at the second step 
# it looks a little better, because the blurring is allowed to have a greater
# impact
fourth_gaus = Gaus_Filt(fourth)
sixteenth_gaus_second_step = shrink(fourth_gaus, 4)
plt.figimage(sixteenth_gaus_second_step)
# here I try applying the filter at the earliest step
# this looks much better; the blurring effect is allowed to have a much
# greater impact
half_gaus = Gaus_Filt(half)
sixteenth_gaus_first_step = shrink(half_gaus, 8)
plt.figimage(sixteenth_gaus_first_step)

#######################################################################

############### Median Filter Experiments #############################

# function to help
def Med_Filt(some_img):
    img = copy.deepcopy(some_img)
    img[:, :, 0] = ndi.filters.median_filter(img[:, :, 0], size = (3, 3))
    img[:, :, 1] = ndi.filters.median_filter(img[:, :, 1], size = (3, 3))
    img[:, :, 2] = ndi.filters.median_filter(img[:, :, 2], size = (3, 3))
    return(img)

# try it at one of the final steps - kind of meh
sixteenth_med = Med_Filt(sixteenth)
plt.figimage(sixteenth_med)
# try it at the second step - little better
fourth_med = Med_Filt(fourth)
sixteenth_med_second_step = shrink(fourth_med, 4)
plt.figimage(sixteenth_med_second_step)
# try it at the first step - pretty good; about the same as Gaussian
half_med = Med_Filt(half)
sixteenth_med_first_step = shrink(half_med, 8)
plt.figimage(sixteenth_med_first_step)

#######################################################################

############### Laplacian Filter Experiments ##########################

# make the filter
Lap = np.array([[0, 1, 0],
                [0, -4, 0],
                [0, 1, 0]])

# function to help
def Lap_Filt(some_img):
    img = copy.deepcopy(some_img)
    img[:, :, 0] = ndi.convolve(img[:, :, 0], Lap)
    img[:, :, 1] = ndi.convolve(img[:, :, 1], Lap)
    img[:, :, 2] = ndi.convolve(img[:, :, 2], Lap)
    return(img)
    
# try it at one of the final steps - nonsense
sixteenth_lap = Lap_Filt(sixteenth)
plt.figimage(sixteenth_lap)
# try it at the second step - still nonsense
fourth_lap = Lap_Filt(fourth)
sixteenth_lap_second_step = shrink(fourth_lap, 4)
plt.figimage(sixteenth_lap_second_step)
# try it at the first step - no idea
half_lap = Lap_Filt(half)
sixteenth_lap_first_step = shrink(half_lap, 8)
plt.figimage(sixteenth_lap_first_step)

#######################################################################
    
############### Box Filter Experiments ################################

# make the (weighted) box filter
Box = np.array([[1/64, 3/64, 1/64],
                [3/64, 24/32, 3/64],
                [1/64, 3/64, 1/64]])

# function to help
def Box_Filt(some_img):
    img = copy.deepcopy(some_img)
    img[:, :, 0] = ndi.convolve(img[:, :, 0], Box)
    img[:, :, 1] = ndi.convolve(img[:, :, 1], Box)
    img[:, :, 2] = ndi.convolve(img[:, :, 2], Box)
    return(img)
    
# try it at one of the final steps - not bad, not good
sixteenth_box = Box_Filt(sixteenth)
plt.figimage(sixteenth_box)
# try it at the second step - pretty good
fourth_box = Box_Filt(fourth)
sixteenth_box_second_step = shrink(fourth_box, 4)
plt.figimage(sixteenth_box_second_step)
# try it at the first step - very good; about the same as the Gaussian
half_box = Box_Filt(half)
sixteenth_box_first_step = shrink(half_box, 8)
plt.figimage(sixteenth_box_first_step)    
    
#######################################################################


