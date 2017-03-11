'''  Utility to extract random frames from the project video, for testing

'''
import pickle
import os
import glob

from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

np.random.seed(22)

count = 0
MODEL = './model/'

video_in  = './project_video.mp4'

clip = VideoFileClip(video_in).subclip(1, 50)

NUM = 20

dur = clip.duration
print('duration:', dur)

for i in range(0, NUM):
    count += 1
    rand = np.random.randint(dur)
    fname = MODEL + '_rand' + str(rand) + '.jpg'
    rand = rand * 1.0
    frame = clip.get_frame(rand)
    plt.imsave(fname, frame)
    print('saved:', fname)
