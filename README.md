## 📷 Skin Lesion Classification

### About

Project for skin lesion classification, attempting to predict diagnoses as either cancerous or non-cancerous. The main method used to accomplish this is image classification, employing methods from scikit-learn and scikit-image. However, a linear regression model based on image metadata is also included.

This is a finished repository. Re-training models (using previously extracted image features) and classifying new images is possible using the trainer.py and classify.py scripts, while reproduction of plots and figures is possible from the included jupyter notebooks. To save space, only the pictures needed to reproduce plots and figures are included in this repository. This means that re-extracting image features is not possible from this repository. A copy of this repository including all images and masks used to produce image features can be found at https://github.com/julielhagen/First-Year-Project.

### Dependencies

**To run classify (mentioned version or newer):**

* python 3
* numpy 1.21.5
* pandas 1.3.5
* matplotlib 3.5.1
* scipy 1.7.3
* scikit-learn 1.2.2
* scikit-image 0.19.2

**To reproduce plots (mentioned version or newer):**

* all of the above
* missingno 0.5.2

### Guide

**To run classify as import in python environment:**

1. Run "import matplotlib.pyplot"
2. Run "from classify import classify"
3. Read image as numpy array using matplotlib.pyplot.imread('YOUR_IMAGE_ID.png')
4. Read mask as numpy array using matplotlib.pyplot.imread('YOUR_MASK_ID.png')
5. Call classify function on the new image and its mask, e.g., classify(image, mask)

**To run classify as python script:**

1. Run "python classify.py"
2. Input image id (including file extension, excluding ticks)
3. Input mask id (including file extension, excluding ticks)

**To reproduce plots and figures:**

Open and run following jupyter notebooks in /code:

* segmentation.ipynb
* feature_selection.ipynb
* model_evaluation.ipynb
* open_question.ipynb
* initial_subset_analysis.ipynb

**To re-train standardizing scalar, PCA, feature selection and classification models (using existing image feature data):**

1. Open terminal
2. Run "python code/trainer.py

### Authors

Christian Sneftrup Fleischer  
chfl@itu.dk

Hugo Lysdahl Hoydal  
huly@itu.dk

Ida Ugilt Wennergaard  
idwe@itu.dk

Julie Langeland Hagen  
jhag@itu.dk

Sander Engel Thilo  
saet@itu.dk



