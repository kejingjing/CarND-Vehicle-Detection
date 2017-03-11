'''
    Main utility to Train an SVM Classifier to detect Vehicles. 

    - Builds a dataset of Cars/NotCars
    - Extracts features
    - Performs PCA for dimensionality reduction
    - Optional GridSearch to search for best model (working, but commented out)
    - Saves the best model and params in a pickle file
    - The model and params are now ready to be used by VehicleDetector for detecting vehicles in video/static images
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from scipy.ndimage.measurements import label

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import RandomizedPCA, PCA
from sklearn.pipeline import Pipeline

from moviepy.editor import VideoFileClip

import pickle
import os
import glob
import time

from  vehicle_detector import *

## -----------------------------------------------------------------------

DATA_DIR     = '/Users/aa/Developer/datasets/udacity-vehicle-detection/'
CAR_DIR      = DATA_DIR + 'vehicles/'
NON_VEHICLES = DATA_DIR + 'non-vehicles/'
MODEL_DIR    = './model/'

TOGGLE_PCA   = True # Toggle switch: True (do PCA); False (dont do PCA)

def save_model(clf, scaler, pca_transformer, featureX, labelsY):
    ''' Save classifier and scaler, and features/labels
    '''
    loc = time.localtime()
    msg = str(loc.tm_mon) + '_' + str(loc.tm_mday) + '_' + str(loc.tm_hour) + '_' + str(loc.tm_min)
    PICKLE_FILE     = MODEL_DIR + 'model_' + msg + '.pkl'
    FEATURE_FILE    = MODEL_DIR + 'feats_' + msg + '.pkl'
    PCA_FILE        = MODEL_DIR + 'pca_'   + msg + '.pkl'
    SCALER_FILE     = MODEL_DIR + 'scaler' + msg + '.pkl'

    # save info
    if TOGGLE_PCA:
        dump_data = {
            "clf":      clf,     # the classifier
            "scaler":   scaler,  # scaler
            "do_pca":   True,    # YES, do PCA
            "pca":      pca_transformer
        }
    else:
        dump_data = {
            "clf":      clf,     # the classifier
            "scaler":   scaler,  # scaler
            "do_pca":   False    # NO PCA
        }
    with open(PICKLE_FILE, 'wb') as PF:
        pickle.dump(dump_data, PF)
        print('Model saved:   ', PICKLE_FILE)

    """ Used for testing only
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
    """


def save_pipeline(pipeline_estimator):
    '''
    '''
    loc = time.localtime()
    msg = str(loc.tm_mon) + '_' + str(loc.tm_mday) + '_' + str(loc.tm_hour) + '_' + str(loc.tm_min)
    PIPELINE     = MODEL_DIR + 'pipe_' + msg + '.pkl'

    with open(PIPELINE, 'wb') as F:
        pickle.dump(pipeline_estimator, F)
        print('Saved pipeline: ', PIPELINE)

def perform_pca(X, n_comp=128, show_plot=False):
    ''' Perform PCA on the given features X, and return a reduced size of X (the principal components)
        :param X:   the scaled feature vector X
        :param n_comp:  number of PCA components to collapse into

        :return     pca (the transformer) 
        :return     and the PCA of X
    '''
    
    
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

def perform_grid_search(X, y):
    ''' Peforms grid search on X and y
        returns best estimator and grid search
    '''
    print('Performing grid search')
    pipeline_svc = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(whiten=True)),
        ('clf', SVC(random_state=1))
    ])
    n_components_range = [64, 128, 256]
    param_range = [0.01, 0.1, 1.0, 10.0]
    param_grid  = [
        {'clf__C':  param_range, 'clf__kernel': ['linear'], 'pca__n_components': n_components_range},
        {'clf__C':  param_range, 'clf__kernel': ['rbf'], 'clf__gamma': param_range, 'pca__n_components': n_components_range}
    ]

    t0 = time.time()
    gs = GridSearchCV(estimator=pipeline_svc, 
                    param_grid=param_grid,
                    scoring='accuracy',
                    cv=3,
                    n_jobs=1, 
                    verbose=1)
    gs = gs.fit(X, y)
    t1 = time.time()
    print('-'*80)
    print('Grid search time:', round(t1-t0, 2))
    print('Grid search best score:', gs.best_score_)
    print('Grid search params:', gs.best_params_)
    print('Best estimator:\n', gs.best_estimator_)
    print('-'*80)

    # return both the Best Estimator and grid search object
    return gs.best_estimator_, gs

def build_pipeline():
    '''
    '''
    pipeline_svc = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(whiten=True, n_components=128)),
        ('clf', SVC(C=1.0, kernel='rbf', gamma=0.01))
        ])
    return pipeline_svc

def save_train_params():
    ''' Save training params -- to be used later for detection
    '''
    params = {
        "color_space":  color_space,
        "orient":       orient,
        "pix_per_cell": pix_per_cell,
        "cell_per_block": cell_per_block,
        "hog_channel":  hog_channel,
        "spatial_size": spatial_size,
        "hist_bins":    hist_bins,
        "spatial_feat": spatial_feat,
        "hist_feat":    hist_feat,
        "hog_feat":     hog_feat
    }
    CONFIG_FILE = './model/params.cfg'
    with open(CONFIG_FILE, 'wb') as F:
        pickle.dump(params, F)

########################## main ###########################

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
# limit   = 2000
#cars    = cars[:limit]
#notcars = notcars[:limit]

print('Cars len    :', len(cars))
print('Non-cars len:', len(notcars))

### CONFIGURATION: Tweak these parameters and see how the results change.
color_space     = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient          = 9  # HOG orientations
pix_per_cell    = 8 # HOG pixels per cell
cell_per_block  = 2 # HOG cells per block
hog_channel     = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size    = (32, 32) # Spatial binning dimensions
hist_bins       = 32    # Number of histogram bins
spatial_feat    = True # Spatial features on or off
hist_feat       = True # Histogram features on or off
hog_feat        = True # HOG features on or off
y_start_stop    = [400, 680] # Min and max in y to search in slide_window()

# save params
save_train_params()

print('PCA: ', TOGGLE_PCA)

print('Using:', 
    color_space, ' color_space, ', 
    orient,'orientations, ',
    pix_per_cell, 'pixels/cell,', 
    cell_per_block,'cells/block, ',
    'hog_channel:', hog_channel,
    ',hist_bins:', hist_bins,
    ',spatial_size:', spatial_size,
    ',y_start_stop:', y_start_stop )

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


## Stack up the data
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

rand_state = np.random.randint(0, 100)

if TOGGLE_PCA:
    ## Perform PCA on scaled_X
    pca_transformer, X_pca = perform_pca(scaled_X, n_comp=64)

    # Split up data into randomized training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, 
                                                        test_size=0.2, random_state=rand_state)
else:  
    print('NOT performing PCA')
    pca_transformer = None
    # Split up data into randomized training and test sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, 
                                                        test_size=0.2, random_state=rand_state)
    

print('X_train: ', X_train.shape)
print('y_train: ', y_train.shape)
print('X_test:  ', X_test.shape)
print('y_test:  ', y_test.shape)

"""
## Do grid search
# best_estimator, gs  = perform_grid_search(X_train, y_train)

## Get built-in Pipeline
best_estimator = build_pipeline()

# now use the best estimator to train/fit and score
## Train the Classifier
t3  = time.time()
best_estimator.fit(X_train, y_train)
t4  = time.time()
print('Train time:', round(t4-t3, 2), ' secs')
# Check the score of the SVC
print('>> Test Accuracy of Estimator: ', round(best_estimator.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t5  = time.time()
print('Test  time:', round(t5-t4, 2), ' secs')

## Save Pipeline
save_pipeline(best_estimator)

"""

## Fit the model
# Use a  SVC 
if TOGGLE_PCA:
    svc = SVC(C=10.0, kernel='rbf', gamma=0.01)      # LinearSVC(max_iter=8000)
else:
    svc = LinearSVC(C=0.08, loss='hinge')
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
save_model(svc, X_scaler, pca_transformer, scaled_X, y)

print('Done')