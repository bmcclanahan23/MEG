#script to create features 
#features include frequency bins from FFT, Hjorth Parameters, Spectral Entropy, and Approximate Entropy

import numpy as np
from scipy.io import loadmat
import warnings


def create_features(XX, tmin, tmax, sfreq, tmin_original=-0.5):
    """Creation of the feature space:
    - restricting the time window of MEG data to [tmin, tmax]sec.
    - Concatenating the 306 timeseries of each trial in one long
      vector.
    - Normalizing each feature independently (z-scoring).
    """
    print "Applying the desired time window."
    beginning = np.round((tmin - tmin_original) * sfreq).astype(np.int)
    end = np.round((tmax - tmin_original) * sfreq).astype(np.int)
    XX = XX[:, :, beginning:end].copy()
    sample_length = XX.shape[2]
    
    print "2D Reshaping: concatenating all 306 timeseries."
    XX = XX.reshape(XX.shape[0], XX.shape[1] * XX.shape[2])
    #Feature normalization 
    XX -= XX.mean(0)
    #instead of normalizing by standard deviation normalize by max
    XX = np.nan_to_num(XX / XX.std(0))

        
    return XX

    
warnings.filterwarnings('ignore')    
    
print "DecMeg2014: https://www.kaggle.com/c/decoding-the-human-brain"
print
subjects_train = range(1, 7) # use range(1, 17) for all subjects
subjects_test = range(8,9)
print "Training on subjects", subjects_train 

# We throw away all the MEG data outside the first 0.5sec from when
# the visual stimulus start:
tmin = 0.0
tmax = 0.500
print "Restricting MEG data to the interval [%s, %s]sec." % (tmin, tmax)

#X_train = []
#y_train = []
all_scores = []
top_scores = []
est_data = []

print
print "Creating the trainset."
for subject in subjects_train:
    filename = 'data/train_subject%02d.mat' % subject
    print "Loading", filename
    data = loadmat(filename, squeeze_me=True)
    XX = data['X']
    yy = data['y']
    sfreq = data['sfreq']
    tmin_original = data['tmin']
    print "Dataset summary:"
    print "XX:", XX.shape
    print "yy:", yy.shape
    print "sfreq:", sfreq

    XX = create_features(XX, tmin, tmax, sfreq)
    
    if subject == 1:
        X_train = XX
        y_train = yy
    else:
        X_train = np.vstack((X_train,XX))
        y_train = np.hstack((y_train,yy))
    
#score features 
#load all test subjects 
for subject in subjects_test:
    filename = 'data/train_subject%02d.mat' % subject
    print "Loading", filename
    data = loadmat(filename, squeeze_me=True)
    XX = data['X']
    yy = data['y']
    sfreq = data['sfreq']
    tmin_original = data['tmin']
    print "Dataset summary:"
    print "XX:", XX.shape
    print "yy:", yy.shape
    print "sfreq:", sfreq

    XX = create_features(XX, tmin, tmax, sfreq)

    if subject == 8:
        X_test = XX
        y_test = yy
    else:
        X_test = np.vstack((X_test,XX))
        y_test = np.hstack((y_test,yy))

