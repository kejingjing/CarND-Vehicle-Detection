'''
    VehicleDetector
    - Main class used for vehicle detection
    - First, train the Classifier (using train_classifier.py). This outputs MODEL_FILE and CONFIG_FILE.
    - Then, use this utility to detect vehicles
    - Ensure that MODEL_FILE and CONFIG_FILE is correctly available to VehicleDetector
    - Can be used on Video or static images

    Code and algorithm is largely based on Udacity lessons
'''

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog
from scipy.ndimage.measurements import label
import time

from sklearn.pipeline import Pipeline
import pickle
import os
import glob

from moviepy.editor import VideoFileClip

## -----------------------------------------------------------------------

## constants
BUFFER_SIZE      = 8   # was 3
COLOR_SALMON    = (250, 128, 114)

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

### - from Hog Sub Sampling Window - 
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

## Compute color histogram features
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    ''' Given an image img, this method returns coordinates of all possible sliding windows
    '''
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer   = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer   = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows  = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows  = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx  = xs*nx_pix_per_step + x_start_stop[0]
            endx    = startx + xy_window[0]
            starty  = ys*ny_pix_per_step + y_start_stop[0]
            endy    = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, 
                    clf, 
                    scaler, 
                    pca,
                    color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):
    ''' Given an image img, and list of windows to search for, this function 
        returns a list of hot_windows where there might be a vehicle

        :param img - the image within which to search
        :param windows - list of windows to search for (output of slide_windows())
        :param clf - the trained classifier that predicts if a vehicle exists in the image
        :param scaler - the scaler transformer
        :param pca      - PCA transformer

        :return hot_windows - a list of windows (bounding boxes) that possibly contain a vehicle
    '''

    last_feat_vec = None
    last_test_feat = None
    last_pca_feat = None 

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        # 5a) Transform using our pipeline: Scale and apply PCA
        # test_features = estimator_pipeline.transform(np.array(features).reshape(1, -1))
        test_features   = scaler.transform(np.array(features).reshape(1, -1))
        if pca is not None:   # perform PCA
            pca_features    = pca.transform(test_features)
        else:               
            pca_features    = test_features
        #6) Predict using your classifier
        prediction = clf.predict(pca_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
        
    ### DEBUG ONLY: TODO: REMOVE THIS
    last_feat_vec = features  # TODO: DEBUG ONLY
    last_test_feat = test_features
    last_pca_feat = pca_features
    # print(' [len features=', len(last_feat_vec), ' test_features=', last_test_feat.shape, ' pca_feat=', last_pca_feat.shape)
    #8) Return windows for positive detections
    return on_windows


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(250, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Heatmap -- add heat in given bounding boxes
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

# Apply threshold to given heatmap: filter out bits that do not cross the threshold 
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], COLOR_SALMON, 6)
    # Return the image
    return img

class FrameBuffer:
    ''' Buffer of heatmaps
    '''
    def __init__(self, max_frames):
        self.frames = []
        self.max_frames = max_frames
    
    def add_queue(self, frame):
        self.frames.insert(0, frame)

    def bufsize(self):
        return len(self.frames)
        
    def pop_queue(self):
        before = len(self.frames)
        self.frames.pop()
        after  = len(self.frames)
    
    def add_frames(self):
        if self.bufsize() > self.max_frames:
            self.pop_queue()
        all_frames = np.array(self.frames)
        return np.sum(all_frames, axis=0)


class VehicleDetector:
    '''  Main class that detects vehicles in a frame
        Initialize with CONFIG PARAMs that were used to train the classifier
        Then use process_frame() method on each frame/image to detect vehicles. Output is a frame with bounding box drawn.
    '''
    def __init__(self, 
                color_space, 
                orient, 
                pix_per_cell, 
                cell_per_block,
                hog_channel,
                spatial_size,
                hist_bins,
                spatial_feat,
                hist_feat,
                hog_feat,
                x_start_stop,
                y_start_stop,
                xy_window,
                xy_overlap,
                heat_threshold,
                clf,                # trained classifier
                scaler,             # scaler for transforming input feature vector
                pca                 # PCA transformer
        ):
        self.color_space     = color_space      # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient          = orient           # HOG orientations
        self.pix_per_cell    = pix_per_cell     # HOG pixels per cell
        self.cell_per_block  = cell_per_block   # HOG cells per block
        self.hog_channel     = hog_channel      # 0, 1, 2, 'ALL'
        self.spatial_size    = spatial_size     # Spatial binning dimensions
        self.hist_bins       = hist_bins        # Number of histogram bins
        self.spatial_feat    = spatial_feat     # Spatial features on or off
        self.hist_feat       = hist_feat        # Histogram features on or off
        self.hog_feat        = hog_feat         # HOG features on or off
        self.y_start_stop    = y_start_stop     # Min and max in y to search in slide_window()
        self.x_start_stop    = x_start_stop
        self.xy_window       = xy_window        # window to search for
        self.xy_overlap      = xy_overlap       # overlap [0,1]
        self.heat_threshold  = heat_threshold   # threshold for false positives; typically 1-5
        self.clf             = clf              # trained classifier    
        self.scaler          = scaler           # scaler for transforming features: 
        self.pca             = pca              # PCA transformer (if not None)

        self.frame_buffer   = FrameBuffer(BUFFER_SIZE)
        self.process_time   = []   # saves seconds required to process each frame

    def print(self):
        ''' Print self config info 
        '''
        print('-'*40)
        msg  = '[color={}, orient={}, pix/cell={}, cell/bk={}, hog={}\n'.format(self.color_space,
            self.orient, self.pix_per_cell, self.cell_per_block, self.hog_channel)
        msg2 = ' spatial_size={}, hist_bins={}, y_start_stop={}, xy_window={}, xy_overlap={} thresh={}]'.format(
            self.spatial_size, self.hist_bins, self.y_start_stop, self.xy_window, self.xy_overlap, self.heat_threshold)
        print(msg + msg2)
        #print(self.estimator_pipeline)
        print(self.clf)
        print('PCA:', self.pca)
        print('-'*40)
    
    def process_frame(self, image, show_heatmap=False, show_box=False):
        '''  Process each frame to detect a vehicle in it, and return a bounded box on it.
            Uses sliding window search algorithm to detect vehicles. Sliding windows determined by Y start/stop and xy_window config params.
            Builds a list of slidind windows, then searches within those windows for vehicles using our given Classifier. 
            After detecting vehicles, draws a bounding box around them. Remove false positives by combining over the last N frames, 
            (maintained by FrameBuffer class) using heatmap. 

            :return image with bounding box drawn around detected vehicle
        '''
        draw_image = np.copy(image)
        image = image.astype(np.float32) / 255.

        t0 = time.time()   # for processing

        # 0.1 - define search window Y start/stop
        Y_START_STOPS = [
            [400, 500],
            [480, 680]
        ]

        # 0.2 - search window sizes: small windows in the horizon, larger windows closer 
        XY_WINDOWS = [
            (80, 80),
            (96, 96)
        ]

        sliding_windows = []

        for i, _ in enumerate(XY_WINDOWS):
            interim_y_start_stop = Y_START_STOPS[i]
            interim_xy_window    = XY_WINDOWS[i]

            # 1 - find sliding windows in frame
            _interim_windows = slide_window(image, 
                                        x_start_stop=self.x_start_stop,
                                        y_start_stop=interim_y_start_stop, 
                                        xy_window=interim_xy_window,
                                        xy_overlap=self.xy_overlap)
            sliding_windows.extend(_interim_windows)
                
        # 2 - search for hot windows within these sliding_windows
        hot_windows = search_windows(image, 
                                    sliding_windows,
                                    self.clf,               
                                    self.scaler,            
                                    self.pca,
                                    color_space=self.color_space,
                                    spatial_size=self.spatial_size,
                                    hist_bins=self.hist_bins,
                                    orient=self.orient,
                                    pix_per_cell=self.pix_per_cell,
                                    cell_per_block=self.cell_per_block,
                                    hog_channel=self.hog_channel,
                                    spatial_feat=self.spatial_feat,
                                    hist_feat=self.hist_feat,
                                    hog_feat=self.hog_feat)
        
        # DEBUG -- 
        # print(' [] hot_wins:', len(hot_windows))
        if show_box:
            boxed_img = draw_boxes(draw_image, hot_windows, color=(0,255,127), thick=2)
            return boxed_img

        # 3a - heat map init
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        # 3b - add heat
        heat = add_heat(heat, hot_windows)
        # 3c -  Buffer frame
        self.frame_buffer.add_queue(heat)

        all_frames = self.frame_buffer.add_frames()

        # 3d - apply heatmap threshold to excl false positives
        heat = apply_threshold(all_frames, self.heat_threshold)

        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)

        if show_heatmap:
            return heatmap

        # 4 - get labels
        labels = label(heatmap)

        # 5 - draw bboxes
        processed_img = draw_labeled_bboxes(draw_image, labels)

        # for debug - time reqd to process
        t2 = time.time()
        process_time = t2-t0
        self.process_time.append(process_time)

        return processed_img

    def show_process_info(self):
        ''' Show total processing info
        '''
        frames = len(self.process_time)
        avg_time    = np.mean(self.process_time) # secs
        total_time  = np.sum(self.process_time)  # secs

        msg = '[Total {0} frames, time: {1} s, avg:{2} sec/frame]'.format(
                frames, round(total_time, 2), round(avg_time, 2)
        )
        print(msg)


###################################################################
def main():
   
    #PIPELINE_FILE = './model/pipe_3_9_19_33.pkl'
    #MODEL_FILE  = './model/model_3_10_11_0.pkl'  # No PCA
    #MODEL_FILE  = './model/model_3_10_12_33.pkl' # PCA_64: acc 0.99
    #MODEL_FILE  = './model/model_3_10_13_27.pkl' # PCA_64, acc 0.99, 4932 feat
    #MODEL_FILE  = './model/model_3_10_15_49.pkl' # No PCA, acc 0.9938, 8460 feat
    MODEL_FILE  = './model/model_3_10_16_13.pkl'  # PCA_64; acc 0.9983, 8460 feat
    # SCALER_FILE = './model/scaler3_10_0_4.pkl'
    # PCA_FILE    = './model/pca_3_10_0_4.pkl'
    CONFIG_FILE = './model/params.cfg'

    """
    pipeline_estimator = None
    print('Loading pipeline estimator:', PIPELINE_FILE)
    with open(PIPELINE_FILE, 'rb') as FIN:
        pipeline_estimator = pickle.load(FIN)

    print('Estimator:\n', pipeline_estimator)
    """

    clf     = None
    scaler  = None
    pca     = None
    with open(MODEL_FILE, 'rb') as M:
        data    = pickle.load(M)
        clf     = data["clf"]
        scaler  = data["scaler"]
        do_pca  = data["do_pca"]
        if do_pca:
            pca     = data["pca"]   # DO perform PCA
        else:
            pca     = None          # DO NOT PERFORM PCA
        print('Read model:', MODEL_FILE)

    ## Read config params (used in training)
    with open(CONFIG_FILE, 'rb') as CF:
        cfg     = pickle.load(CF)

    color_space     = cfg["color_space"]
    orient          = cfg["orient"]
    pix_per_cell    = cfg["pix_per_cell"]
    cell_per_block  = cfg["cell_per_block"]
    hog_channel     = cfg["hog_channel"]
    spatial_size    = cfg["spatial_size"]
    hist_bins       = cfg["hist_bins"]
    spatial_feat    = cfg["spatial_feat"]
    hist_feat       = cfg["hist_feat"]
    hog_feat        = cfg["hog_feat"]

    y_start_stop    = [400, 680] # Min and max in y to search in slide_window()
    heat_threshold  = 4

    
    vehicle_detector = VehicleDetector(
        color_space=color_space,
        orient=orient,
        pix_per_cell=pix_per_cell,
        cell_per_block=cell_per_block,
        hog_channel=hog_channel,
        spatial_size=spatial_size,
        hist_bins=hist_bins,
        spatial_feat=spatial_feat,
        hist_feat=hist_feat,
        hog_feat=hog_feat,
        x_start_stop=[None, None],
        y_start_stop=y_start_stop,
        xy_window=(96, 96),
        xy_overlap=(0.75, 0.75),
        heat_threshold=heat_threshold,
        clf=clf,           
        scaler=scaler, 
        pca=pca)

    vehicle_detector.print()

    """# -- For testing static images ---
    test_images = [
        './test_images/test6.jpg',
        './test_images/test1.jpg',
        './test_images/test5.jpg',
        './test_images/test3.jpg',
        './test_images/test4.jpg',
        './test_images/test2.jpg'
        ]

    print('Running test..')


    for test_img in test_images:
        imgtest = mpimg.imread(test_img)
        out_img = vehicle_detector.process_frame(imgtest, show_box=False, show_heatmap=False)
        plt.imshow(out_img)
        plt.title(test_img)
        plt.show()

    
    ### More tests... mine

    TEST_DIR = './model/'
    OUT_DIR  = './out/'
    tests = glob.glob(TEST_DIR + '*_rand*.jpg')
    print('Running tests..')
    for test_img in tests:
        imgtest = mpimg.imread(test_img)
        #print('processing:', test_img)
        out_img = vehicle_detector.process_frame(imgtest, show_box=False, show_heatmap=False)
        outfile = OUT_DIR + 'out_' + test_img.split('/')[2]
        plt.imsave(outfile, out_img)
        #print('saved out:', outfile)


    """

    ### Testing video

    video_in  = './project_video.mp4'
    clip = VideoFileClip(video_in)

    video_out = './output_images/sub_video_new_pca_buf8_th4_y96_all.mp4'

    clip_out = clip.fl_image(vehicle_detector.process_frame)
    clip_out.write_videofile(video_out, audio=False)

    

    # processing info
    vehicle_detector.show_process_info()

    

if __name__ == '__main__':
    main()