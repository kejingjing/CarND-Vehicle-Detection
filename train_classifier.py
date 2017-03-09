import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from scipy.ndimage.measurements import label

from moviepy.editor import VideoFileClip

import pickle
import os
import glob
import time

from  vehicle_detector import *

DATA_DIR     = '/Users/aa/Developer/datasets/udacity-vehicle-detection/'
CAR_DIR      = DATA_DIR + 'vehicles/'
NON_VEHICLES = DATA_DIR + 'non-vehicles/'
MODEL_DIR    = './model/'

def save_model(clf, scaler, pca_transformer, featureX, labelsY):
    ''' Save classifier and scaler, and features/labels
    '''
    loc = time.localtime()
    msg = str(loc.tm_mon) + '_' + str(loc.tm_mday) + '_' + str(loc.tm_hour) + '_' + str(loc.tm_min)
    PICKLE_FILE     = MODEL_DIR + 'model_' + msg + '.pkl'
    FEATURE_FILE    = MODEL_DIR + 'feats_' + msg + '.pkl'
    PCA_FILE        = MODEL_DIR + 'pca_'   + msg + '.pkl'
    SCALER_FILE     = MODEL_DIR + 'scaler' + msg + '.pkl'
    #print('Saving model:  ', PICKLE_FILE)

    # save info
    dump_data = {
        "clf":      clf     # the classifier
    }
    with open(PICKLE_FILE, 'wb') as PF:
        pickle.dump(dump_data, PF)
        print('Model saved:   ', PICKLE_FILE)

    feature_labels = {
        "features": featureX,
        "labels":   labelsY
    }
    with open(FEATURE_FILE, 'wb') as MF:
        pickle.dump(feature_labels, MF)
        print('Features saved:', FEATURE_FILE)
    
    with open(PCA_FILE, 'wb') as FF:
        pickle.dump(pca_transformer, FF)
        print('PCA saved:     ', PCA_FILE)
    
    with open(SCALER_FILE, 'wb') as SF:
        pickle.dump(scaler, SF)
        print('Scaler saved:  ', SCALER_FILE)


def perform_pca(X, n_comp=128, show_plot=False):
    ''' Perform PCA on the given features X, and return a reduced size of X (the principal components)
        :param X:   the scaled feature vector X
        :param n_comp:  number of PCA components to collapse into

        :return     pca (the transformer) 
        :return     and the PCA of X
    '''
    from sklearn.decomposition import RandomizedPCA, PCA
    
    pca = PCA(n_components=n_comp, whiten=True)
    pca = pca.fit(X)
    pca_features = pca.transform(X)

    explained_variance = pca.explained_variance_ratio_
    components = pca.components_
    print("Explained variance by {} principal components: {:.4f}".format(n_comp, sum(explained_variance[:n_comp])))

    if show_plot is True:
        # plot PCA explanation
        plt.subplot(2, 1, 1)
        plt.xlabel('Dimension')
        plt.ylabel('Explained Variance')
        plt.title('Explained Variances of PCA')
        plt.plot(pca.explained_variance_ratio_)

        plt.subplot(2, 1, 2)
        plt.xlabel('Dimension')
        plt.ylabel('Cumu. Explained Variance')
        plt.title('Cumulative Explained Variances of PCA')
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.show()

    return pca, pca_features



##### main ###########################

print('Looking for images: ', DATA_DIR)
images =    glob.glob(DATA_DIR + '**/*.png', recursive=True)   # For Larger dataset
# images =    glob.glob(LOCAL_DATA_DIR + '**/*.jpeg', recursive=True)    # for LOCAL dataset
print('Images len:  ', len(images))

cars = []
notcars = []
for image in images:
    if 'non-vehicles' in image or 'non-vehicles_smallset' in image:
        notcars.append(image)
    else:
        cars.append(image)

## for testing only
limit   = 3000
cars    = cars[:limit]
notcars = notcars[:limit]

print('Cars len    :', len(cars))
print('Non-cars len:', len(notcars))

### TODO: Tweak these parameters and see how the results change.
color_space     = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient          = 9  # HOG orientations
pix_per_cell    = 8 # HOG pixels per cell
cell_per_block  = 2 # HOG cells per block
hog_channel     = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size    = (16, 16) # Spatial binning dimensions
hist_bins       = 16    # Number of histogram bins
spatial_feat    = True # Spatial features on or off
hist_feat       = True # Histogram features on or off
hog_feat        = True # HOG features on or off
y_start_stop    = [400, 680] # Min and max in y to search in slide_window()


print('Using:', 
    color_space, ' color_space, ', 
    orient,'orientations, ',
    pix_per_cell, 'pixels/cell,', 
    cell_per_block,'cells/block, ',
    'hog_channel=', hog_channel,
    'hist_bins=', hist_bins,
    'spatial_size=', spatial_size,
    'y_start_stop=', y_start_stop )

print('Extracting features...')

t0 = time.time()

car_features = extract_features(cars, 
                        color_space=color_space, 
                        spatial_size=spatial_size, 
                        hist_bins=hist_bins, 
                        orient=orient, 
                        pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, 
                        spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, 
                        hog_feat=hog_feat)

t1 = time.time()
print('  Car features done   : in {0} secs'.format(round(t1-t0, 2)))

notcar_features = extract_features(notcars, 
                        color_space=color_space, 
                        spatial_size=spatial_size, 
                        hist_bins=hist_bins, 
                        orient=orient, 
                        pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, 
                        spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, 
                        hog_feat=hog_feat)
                    
t2 = time.time()
print('  Notcar features done: in {0} secs'.format(round(t2-t1, 2)))
print('Features :', len(car_features[0]))


## Train SVM Classifier
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

## Perform PCA on scaled_X
pca_transformer, X_pca = perform_pca(scaled_X, n_comp=512)

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, 
                                                    test_size=0.2, random_state=rand_state)


print('X_train: ', X_train.shape)
print('y_train: ', y_train.shape)
print('X_test:  ', X_test.shape)
print('y_test:  ', y_test.shape)

## Fit the model
# Use a linear SVC 
svc = LinearSVC(max_iter=8000)
# Check the training time for the SVC
print(svc)
t3  = time.time()
## Train the Classifier
svc.fit(X_train, y_train)
t4  = time.time()
print('Train time:', round(t4-t3, 2), ' secs')
# Check the score of the SVC
print('>> Test Accuracy of SVC: ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t5  = time.time()
print('Test  time:', round(t5-t4, 2), ' secs')

## Save Model, Scaler and  Features/Labels
save_model(svc, X_scaler, pca_transformer, X_pca, y)

print('Done')