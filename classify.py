### FUNCTION THAT CLASSIFIES NEW IMAGES 
### TO USE: 
### 1. from classify import classify
### 2. call classify function on the new image and its mask

import pickle as pk
import pandas as pd
import numpy as np
import os
from skimage.transform import resize

import sys
sys.path.append("code")
from extract_features import extract_features

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
    X = pd.DataFrame(extract_features(img, mask)).T
    X.columns = feature_names

    # Apply scalar
    scaler = pk.load(open('code' + os.sep + 'scaler.pkl', 'rb'))
    X_scaled = scaler.transform(X)

    # Apply PCA
    pca = pk.load(open('code' + os.sep + 'pca.pkl', 'rb'))
    # X_normalized = (X - X.mean()) / X.std()
    # X_normalized = X_normalized.T
    # X_normalized.columns = feature_names
    X_transformed = pca.transform(X_scaled) # Transpose dataframe to get features as columns

    # Apply feature selector
    feature_selector = pk.load(open('code' + os.sep + 'selector.pkl', 'rb'))
    X_transformed = feature_selector.transform(X_transformed)

    # Import classifier
    classifier = pk.load(open('code' + os.sep + 'classifier.pkl', 'rb'))

    pred_label = classifier.predict(X_transformed)[0]
    pred_prob = classifier.predict_proba(X_transformed)[0][1]

    if pred_label:
        diagnoses = 'unhealthy'
    else:
        diagnoses = 'healthy'

    print(f'Predicted label is: {pred_label} ({diagnoses})')
    print(f'Predicted probability of lesion being unhealthy is: {pred_prob}')
