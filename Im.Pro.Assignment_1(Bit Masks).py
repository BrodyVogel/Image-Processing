#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 16:39:23 2018

@author: vogebr01
"""

# Brody Vogel's first COSC 455 Homework (Bit Masks) #

# import statements
import skimage
import os
from skimage import io
from matplotlib import pyplot as plt

# read in the image
bill_img = io.imread('/Users/brodyvogel/Desktop/COSC 455/Im. Pro. Assignment 1/100-dollars.tif')

# set the plot size to 8 rows of 2; make figure size a little bigger, too
fig, ax = plt.subplots(8, 2, figsize = (12, 12))
# give each image some headspace so they don't overlap
plt.subplots_adjust(hspace = .7)

#%matplotlib qt

# list of the bits that I need to mask
bits = [0b11111110, 0b11111101, 0b11111011, 0b11110111, 0b11101111, 0b11011111,
             0b10111111, 0b01111111]

new_img = bill_img

# loop through the bits
for bit in bits:
    # this puts each image in the right plot index
    which_bit = 2 * (bits.index(bit) + 1) - 1
    # apply the bit mask 
    new_img = new_img & bit
    
    # choose the spot for the plot
    plt.subplot(8, 2, which_bit)
    # # add text to the masked image
    plt.text(500, -20, 'Bit Masked: ' + str(bits.index(bit)+1))
    # show the masked image
    plt.imshow(new_img, cmap = 'gray')
    
    # move the plotting index over 1 for the histogram
    plt.subplot(8, 2, which_bit + 1)
    # add text to the histogram plot
    plt.text(125, 70000, 'Histogram')
    # plot the histogram
    plt.hist(new_img.flatten(), bins = 256, range = (0, 255))

# so things don't overlap 
plt.tight_layout()