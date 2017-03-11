'''  Utility functions for vehicle detection

'''
import os

import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

from vehicle_detector import *

def show_random_images(images, num_images=12, n_row=3, title=None):
    '''
    '''
    rx = np.random.choice(images, num_images)
    imgs_show = [images[r] for r in rx]
    
    grid = gridspec.GridSpec(num_images // n_row + 1, n_row)
    grid.update(wspace=0.1, hspace=0.1)
    plt.figure(figsize=(n_row, num_images // nrow + 1))

    for i in range(0, num_images):
        ax1 = plt.subplot(grid[i])
        ax1.axis('off')
        ax1.imshow(imgs_show[i])
    
    if title is not None:
        plt.suptitle(title)
    plt.show()

def random_image(image_files):
    rx = np.random.randint(len(image_files))
    img_file = image_files[rx]
    img = cv2.imread(img_file)
    return img

def show_hog_sample(img, hog_img, hog_features):
    plt.figure(figsize=(10,2))

    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(hog_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.plot(hog_features)
    plt.ylim(0,1)
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    plt.show()

def show_color_histogram_sample(img, color_space_img):
    plt.figure(figsize=(10,2))

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.plot(color_hist(color_space_img))
    #plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    #plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    plt.show()