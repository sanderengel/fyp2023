### Compactness Module. Computes a compactness score for a given mask
### using the Polsby-Popper measure.

###############
### IMPORTS ###
###############

# Standard modules
import numpy as np
import matplotlib.pyplot as plt

# Image processing
from skimage import morphology

# Pi
from math import pi

###################
### COMPACTNESS ###
###################

def compactness_score(mask):
    '''Computes a compactness score for the given mask.
    The score is based of the Polsby-Popper measure. 
    The score falls between the value 0 and 1. Scores closer to 1 indicates a more compact mask.

    Args:
        mask (numpy.ndarray): input masked image
    
    Returns:
        compactness_score (float): Float between 0 and 1 indicating compactness. 
    '''

     #Area of ground truth 
    A = np.sum(mask)

    #Structural element, that we will use as a "brush" on our mask
    struct_el = morphology.disk(2)

    # Use this "brush" to erode the image - eat away at the borders
    mask_eroded = morphology.binary_erosion(mask, struct_el)

    #Finding the perimeter of the ground truth
    perimeter = mask - mask_eroded

    #Length of the perimeter
    l = np.sum(perimeter)

    compactness = (4*pi*A)/(l**2)

    score = round(1-compactness, 3)

    return compactness

################
### PLOTTING ###
################

def show_border(mask):
    '''Plot of the given mask's border i.e. it's perimeter.

    Args:
        mask (numpy.ndarray): input masked image
    
    Returns:
        plot (matplotlib.pyplot): plot showing the border of the given mask.
    '''
    #Area of ground truth 
    A = np.sum(mask)

    #Structural element, that we will use as a "brush" on our mask
    struct_el = morphology.disk(2)

    # Use this "brush" to erode the image - eat away at the borders
    mask_eroded = morphology.binary_erosion(mask, struct_el)

    #Finding the perimeter of the ground truth
    perimeter = mask - mask_eroded

    return plt.imshow(perimeter, cmap='gray')