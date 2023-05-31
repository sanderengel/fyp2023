### FUNCTION THAT CLASSIFIES NEW IMAGES 
### TO USE: 
### 1. from classify import classify
### 2. call classify function on the new image(s) and its/their mask(s)

import pickle as pk
import os
from code.extract_features import extract_features
from skimage.transform import resize

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

    img = resize(img, (300, 300))

    X = extract_features(img, mask)

    # Apply PCA
    pca = pk.load(open('pca.pkl', 'rb'))
    X_normalized = (X - X.mean()) / X.std()
    X_transformed = pca.transform(X_normalized)

    # Apply feature selector
    feature_selector = pk.load(open('selector.pkl', 'rb'))
    X_transformed = feature_selector.transform(X)

    # Imports classifier
    classifier = pk.load(open('code' + os.sep + 'classifier.pkl', 'rb'))

    pred_label = classifier.predict(X_transformed)
    pred_prob = classifier.predict_proba(X_transformed)

    return pred_label, pred_prob


