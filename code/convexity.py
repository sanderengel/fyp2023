### Convexity module. Computes convexity score based on convex hull

###############
### IMPORTS ###
###############

# Standard modules
import numpy as np
import matplotlib.pyplot as plt

# Image processing
from scipy.spatial import ConvexHull

#################
### CONVEXITY ###
#################

def convexity_score(mask):
    '''Calculate convexity score between 0 and 1, 
    with 0 indicating a smoother border and 1 a more crooked border.

    Args:
        image (numpy.ndarray): input masked image
    
    Returns:
        convexity_score (float): Float between 0 and 1 indicating convexity.    
    '''

    # Get coordinates of all pixels in the lesion mask
    coords = np.transpose(np.nonzero(mask))

    # Compute convex hull of lesion pixels
    hull = ConvexHull(coords)

    # Compute area of lesion mask
    lesion_area = np.count_nonzero(mask)

    # Compute area of convex hull
    convex_hull_area = hull.volume + hull.area

    # Compute convexity as ratio of lesion area to convex hull
    convexity = lesion_area / convex_hull_area
    
    return convexity #round(1-convexity, 3)

################
### PLOTTING ###
################

def plot_convex_hull(mask):
    '''Plot mask with convex hull.
    Mask is plotted with its red convex hull line.

    Args:
        image (numpy.ndarray): input masked image
    
    Returns:
        plot: plot including mask and convex hull boundary.
    '''

    # Get coordinates of all pixels in the lesion mask
    coords = np.transpose(np.nonzero(mask))

    # Compute convex hull of lesion pixels
    hull = ConvexHull(coords)

    plt.imshow(mask, cmap='gray')

    # Plot convex hull
    for simplex in hull.simplices:
        plt.plot(coords[simplex, 1], coords[simplex, 0], 'r')
    
    return plt.show()