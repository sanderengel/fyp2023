### Main classification function used to classify new images.

### To use as import: 
### 1. From classify import classify
### 2. Call classify function on the new image and its mask

### To run as script:
### 1. Run classify.py
### 2. Input image id (excluding ticks)
### 3. Input mask id (excluding ticks)

import pickle as pk
import pandas as pd
import numpy as np
import os
from skimage.transform import resize

import sys
sys.path.append("code")
from model import extract_features

feature_names = ['mean_assymmetry', 'best_asymmetry', 'worst_asymmetry', 'red_var', 'green_var', \
     'blue_var', 'hue_var', 'sat_var', 'val_var', 'dom_hue', 'dom_sat', 'dom_val', \
     'compactness', 'convexity', 'F1', 'F2', 'F3', 'F10', 'F11', 'F12']

def classify(img, mask):
    '''Predict label and and probability for image and mask using trained 
    PCA, feature selector, and classifier models
    
    Args:
        img (numpy.ndarray): input image
        mask (numpy.ndarray): input mask

    Return:
        pred_label (int): predicted label (either 1 for unhealthy or 0 for healthy)
        pred_prob (float): predicted probability
    '''

    # Cut of any potential extra color channels
    img = img[:, :, :3]
    if len(mask.shape) == 3:
        mask = mask[:, :, 0] # Some masks have more than 2 dimensions, which we slice off here

    # Resize
    img = resize(img, (300, 300))
    mask = resize(mask, (300, 300))

    # Assert mask is binary
    binary_mask = np.zeros_like(mask)
    binary_mask[mask > .5] = 1
    mask = binary_mask.astype(int)

    # Extract features into dataframe and add column names
    X = pd.DataFrame(extract_features(img, mask)).T # Transpose dataframe to get features as columns
    X.columns = feature_names

    # Apply scalar
    scaler = pk.load(open('code' + os.sep + 'scaler.pkl', 'rb'))
    X_scaled = scaler.transform(X)

    # Apply PCA
    pca = pk.load(open('code' + os.sep + 'pca.pkl', 'rb'))
    X_transformed = pca.transform(X_scaled)

    # Apply feature selector
    feature_selector = pk.load(open('code' + os.sep + 'selector.pkl', 'rb'))
    X_transformed = feature_selector.transform(X_transformed)

    # Import classifier
    classifier = pk.load(open('code' + os.sep + 'classifier.pkl', 'rb'))

    pred_label = classifier.predict(X_transformed)[0]
    pred_prob = classifier.predict_proba(X_transformed)[0][1]

    if pred_label:
        diagnoses = 'cancerous'
    else:
        diagnoses = 'not cancerous'

    print(f'Predicted label: {pred_label} ({diagnoses})')
    print(f'Predicted probability of lesion being cancerous: {round(pred_prob, 4)}')

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    print('\nInput image and mask file names (including file extensions, excluding ticks).\n')

    im_id = input('Image file name: ')
    mask_id = input('Mask file name: ')

    print('\n')

    im = plt.imread(im_id)
    mask = plt.imread(mask_id)

    classify(im, mask)
