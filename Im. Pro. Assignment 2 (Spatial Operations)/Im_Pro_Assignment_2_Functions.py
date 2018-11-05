#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 10:17:33 2018

@author: vogebr01
"""
import skimage
import os
from skimage import io
from matplotlib import pyplot as plt
import numpy as np

# Helper function for creating some divisors
def dimMult(img):
    prod = 1
    for dim in img.shape:
        prod = prod * dim
    return(prod)

# Histogram Equalization function
def histEq(img):
    # get the largest intensity value from the image, for the scale
    maximum = np.max(img)
    # build an array that holds the counts for each value from 0 to the maximum
        # value in the image
    counts = np.zeros(maximum + 1)
    # go through every unique intesity value in the image and add the count
        # for that value to the appropriate spot in the 'counts' array
    for z in np.unique(img):
        counts[z] = (img == z).sum()
    # type conversions so things don't get messy    
    counts = [int(t) for t in counts]
    # this first takes the probability of each intensity value from the original
        # image and multiplies by the desired maximum output intensity (Normalizes)
    # it then finds the cumulative sum of the values in the list so as to map the
        # original intensity values (bins) to the desired new ones
    counts = np.cumsum([255 * (t / dimMult(img)) for t in counts])
    # round so as to use whole-numbered intensity values
    new_vals = [int(round(t, 0)) for t in counts]
    # go through the input image and map all the original intensity values to
        # their new ones as determined by the above 
    new_img = np.asarray([new_vals[t] for t in img.flatten()])
    # reverse the flattening that was used for convenience in the previous line
    if len(img.shape) == 3:
        return(new_img.reshape(img.shape[0], img.shape[1], img.shape[2]))
    else:
        return(new_img.reshape(img.shape[0], img.shape[1])) 

# Contrast Stretching function    
def contStretch(img, r1 = 0, r2 = 0, s1 = 0, s2 = 255):
    # if the optional parameters are not passed in, they're fixed here
    if r1 == 0:
        r1 = np.min(img)
    if r2 == 0:
        r2 = np.max(img)
    # apply the transformation (stretching) function to every 
        # intensity value in the image (also make every value a rounded integer)
    new_img = np.asarray([int(round(((s2-s1)/(r2-r1) * (t - r1) + s1), 0)) 
                        for t in img.flatten()])
    # reverse the flattening and return the new image
    if len(img.shape) == 3:
        return(new_img.reshape(img.shape[0], img.shape[1], img.shape[2]))
    else:
        return(new_img.reshape(img.shape[0], img.shape[1]))

# Gamma Transformation function    
def gammaTrans(img, gamma):
    # raise all the intensity values to gamma; also round and make them integers
    new_img = [int(round(i**gamma, 0)) for i in img.flatten()]
    maxx = np.max(new_img)
    minn = np.min(new_img)
    div = maxx - minn
    new_img = np.asarray([int(round(255/div * (i - minn), 0)) 
                        for i in new_img])
    # return the un-flattened image
    if len(img.shape) == 3:
        return(new_img.reshape(img.shape[0], img.shape[1], img.shape[2]))
    else:
        return(new_img.reshape(img.shape[0], img.shape[1])) 
  
# Intensity Quantization function    
def intensityQuan(img, num_bits):
    # possible bit masks
    bits = [0b00000000, 0b10000000, 0b11000000, 0b11100000, 0b11110000, 
            0b11111000, 0b11111100, 0b11111110, 0b11111111]
    # & all the intensities with the specified bit representation
    new_img = np.asarray([t & bits[num_bits] for t in img.flatten()])
    # return un-flattened image
    if len(img.shape) == 3:
        return(new_img.reshape(img.shape[0], img.shape[1], img.shape[2]))
    else:
        return(new_img.reshape(img.shape[0], img.shape[1]))   

# Inverse Transform function    
def negative(img):
    # apply the inverse function to every intensity value in the image
    maxx = np.max(img)
    new_img = np.asarray([int(round(maxx-1-r)) for r in img.flatten()])
    # reverse the flatten and return the new image
    return(new_img.reshape(img.shape[0], img.shape[1]))

# Binary Transform function
def Binary(img):
    # turn every intensity below 127 to 0 (white) and all those above
        # to 255 (black)
    new_img = np.asarray([np.where(x > 127, 255, 0) for x in img.flatten()])
    # reverse the flatten and return the new image
    return(new_img.reshape(img.shape[0], img.shape[1])) 
   
# Blend function    
def blend(imgA, imgB, alpha, steps):
    # image that'll slowly be blended to second image
    new_img = imgA
    # list to hold the new blended images
    imgs = []
    # populate the list with the increasingly blended images
    for z in range(0, steps):
        # blend function
        new_img = alpha * new_img + (1 - alpha)*imgB
        # round all the intensities
        new_img = np.asarray([int(round(t, 0)) for t in new_img.flatten()])
        # reshape the image
        new_img = new_img.reshape(imgA.shape[0], imgA.shape[1])
        # add the new image
        imgs.append(new_img)
    # return the blended images
    return(imgs) 
  
# Difference function    
def difference(imgA, imgB):
    # absolute value of the difference between the images' intensities
    new_img = abs(imgA - imgB)
    # return it
    return(new_img)    
    
    