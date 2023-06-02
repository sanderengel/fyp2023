### Color module. First part uses SLIC segmentation algorithm to segment lesion by color in area marked by mask.
### Further describes the colors by calculating the mean of each segment from SLIC and computing variance
### or standard deviation between these means, either in RGB or HSV color space.
### Second part (dominance) uses KMeans to compute dominant colors from HSV color space.

################
#### IMPORTS ###
################

# Standard modules
import numpy as np
import matplotlib.pyplot as plt

# Statistics
from statistics import variance, stdev
from scipy.stats import circmean, circvar, circstd
from math import nan

# Image processing
from skimage.segmentation import slic
from skimage.color import rgb2hsv
from sklearn.cluster import KMeans

# Own modules
from cut import cut_im_by_mask

###############################
### SLIC COLOR SEGMENTATION ###
###############################

def slic_segmentation(image, mask, n_segments = 50, compactness = 0.1):
    '''Get color segments of lesion from SLIC algorithm. 
    Optional argument n_segments (defualt 50) defines desired amount of segments.
    Optional argument compactness (defualt 0.1) defines balance between color 
    and position.

    Args:
        image (numpy.ndarray): image to segment
        mask (numpy.ndarray):  image mask
        n_segments (int, optional): desired amount of segments
        compactness (float, optional): compactness score, decides balance between
            color and and position

    Returns:
        slic_segments (numpy.ndarray): SLIC color segments.
    '''
    slic_segments = slic(image,
                    n_segments = n_segments,
                    compactness = compactness,
                    sigma = 1,
                    mask = mask,
                    start_label = 1,
                    channel_axis = 2)
    
    return slic_segments

##############################
### EXTRACTING COLOR MEANS ###
##############################

def get_rgb_means(image, slic_segments):
    '''Get mean RGB values for each segment in a SLIC segmented image.

    Args:
        image (numpy.ndarray): original image
        slic_segments (numpy.ndarray): SLIC segmentation

    Returns:
        rgb_means (list): RGB mean values for each segment.
    '''

    max_segment_id = np.unique(slic_segments)[-1]

    rgb_means = []
    for i in range(1, max_segment_id + 1):

        #Create masked image where only specific segment is active
        segment = image.copy()
        segment[slic_segments != i] = -1

        #Get average RGB values from segment
        rgb_mean = np.mean(segment, axis = (0, 1), where = (segment != -1))
        
        rgb_means.append(rgb_mean) 
        
    return rgb_means

def get_hsv_means(image, slic_segments):
    '''Get mean HSV values for each segment in a SLIC segmented image.

    Args:
        image (numpy.ndarray): original image
        slic_segments (numpy.ndarray): SLIC segmentation

    Returns:
        hsv_means (list): HSV mean values for each segment.
    '''

    hsv_image = rgb2hsv(image)

    max_segment_id = np.unique(slic_segments)[-1]

    hsv_means = []
    for i in range(1, max_segment_id + 1):

        # Create masked image where only specific segment is active
        segment = hsv_image.copy()
        segment[slic_segments != i] = nan

        #Get average HSV values from segment
        hue_mean = circmean(segment[:, :, 0], high=1, low=0, nan_policy='omit') # Compute circular hue mean
        sat_mean = np.mean(segment[:, :, 1], where = (slic_segments == i)) # Compute saturation mean
        val_mean = np.mean(segment[:, :, 2], where = (slic_segments == i)) # Compute value mean

        hsv_mean = np.asarray([hue_mean, sat_mean, val_mean])

        hsv_means.append(hsv_mean)
        
    return hsv_means

################
### VARIANCE ###
################

def rgb_var(image, slic_segments):
    '''Get variance of RGB means for each segment in 
    SLIC segmentation in red, green and blue channels

    Args:
        image (numpy.ndarray): image to compute color variance for
        slic_segments (numpy.ndarray): array containing SLIC segmentation
    
    Returns:
        red_var (float): variance in red channel segment means
        green_var (float): variance in green channel segment means
        blue_var (float): variance in green channel segment means.
    '''

    # If there is only 1 slic segment, return (0, 0, 0)
    if len(np.unique(slic_segments)) == 2: # Use 2 since slic_segments also has 0 for area outside mask
        return 0, 0, 0

    rgb_means = get_rgb_means(image, slic_segments)
    n = len(rgb_means) # Amount of segments, used later to compute variance

    # Seperate and collect channel means together in lists
    red = []
    green = []
    blue = []
    for rgb_mean in rgb_means:
        red.append(rgb_mean[0])
        green.append(rgb_mean[1])
        blue.append(rgb_mean[2])

    # Compute variance for each channel seperately
    red_var = variance(red, sum(red)/n)
    green_var = variance(green, sum(green)/n)
    blue_var = variance(blue, sum(blue)/n)

    return red_var, green_var, blue_var

def hsv_var(image, slic_segments):
    '''Get variance of HSV means for each segment in 
    SLIC segmentation in hue, saturation and value channels

    Args:
        image (numpy.ndarray): image to compute color variance for
        slic_segments (numpy.ndarray): array containing SLIC segmentation
    
    Returns:
        hue_var (float): variance in hue channel segment means
        sat_var (float): variance in saturation channel segment means
        val_var (float): variance in value channel segment means.
    '''

    # If there is only 1 slic segment, return (0, 0, 0)
    if len(np.unique(slic_segments)) == 2: # Use 2 since slic_segments also has 0 marking for area outside mask
        return 0, 0, 0

    hsv_means = get_hsv_means(image, slic_segments)
    n = len(hsv_means) # Amount of segments, used later to compute variance

    # Seperate and collect channel means together in lists
    hue = []
    sat = []
    val = []
    for hsv_mean in hsv_means:
        hue.append(hsv_mean[0])
        sat.append(hsv_mean[1])
        val.append(hsv_mean[2])

    # Compute variance for each channel seperately
    hue_var = circvar(hue, high=1, low=0)
    sat_var = variance(sat, sum(sat)/n)
    val_var = variance(val, sum(val)/n)

    return hue_var, sat_var, val_var

##########################
### STANDARD DEVIATION ###
##########################

def rgb_sd(image, slic_segments):
    '''Get standard deviation of RGB means for each segment in 
    SLIC segmentation in red, green and blue channels

    Args:
        image (numpy.ndarray): image to compute color standard deviation for
        slic_segments (numpy.ndarray): array containing SLIC segmentation
    
    Returns:
        red_sd (float): standard deviation in red channel segment means
        green_sd (float): standard deviation in green channel segment means
        blue_sd (float): standard deviation in green channel segment means
    '''

    # If there is only 1 slic segment, return (0, 0, 0)
    if len(np.unique(slic_segments)) == 2:
        return 0, 0, 0
    
    rgb_means = get_rgb_means(image, slic_segments)
    n = len(rgb_means) # Amount of segments, used later to compute standard deviation

    #Seperate and collect channel means together in lists
    red = []
    green = []
    blue = []
    for rgb_mean in rgb_means:
        red.append(rgb_mean[0])
        green.append(rgb_mean[1])
        blue.append(rgb_mean[2])
    
    #Compute standard deviation for each channel seperately
    red_sd = stdev(red, sum(red)/n)
    green_sd = stdev(green, sum(green)/n)
    blue_sd = stdev(blue, sum(blue)/n)

    return red_sd, green_sd, blue_sd

def hsv_sd(image, slic_segments):
    '''Get standard deviation of HSV means for each segment in 
    SLIC segmentation in hue, saturation and value channels

    Args:
        image (numpy.ndarray): image to compute color standard deviation for
        slic_segments (numpy.ndarray): array containing SLIC segmentation
    
    Returns:
        hue_sd (float): standard deviation in hue channel segment means
        sat_sd (float): standard deviation in saturation channel segment means
        val_sd (float): standard deviation in value channel segment means.
    '''

    # If there is only 1 slic segment, return (0, 0, 0)
    if len(np.unique(slic_segments)) == 2: # Use 2 since slic_segments also has 0 marking for area outside mask
        return 0, 0, 0

    hsv_means = get_hsv_means(image, slic_segments)
    n = len(hsv_means) # Amount of segments, used later to compute standard deviation

    # Seperate and collect channel means together in lists
    hue = []
    sat = []
    val = []
    for hsv_mean in hsv_means:
        hue.append(hsv_mean[0])
        sat.append(hsv_mean[1])
        val.append(hsv_mean[2])

    # Compute standard deviation for each channel seperately
    hue_sd = circstd(hue, high=1, low=0)
    sat_sd = stdev(sat, sum(sat)/n)
    val_sd = stdev(val, sum(val)/n)

    return hue_sd, sat_sd, val_sd

#######################
### COLOR DOMINANCE ###
#######################

def color_dominance(image, mask, clusters = 5, include_ratios = False):
    '''Get the most dominent colors of the cut image that closest sorrounds the lesion using KMeans

    Args:
        image (numpy.ndarray): image to compute dominent colors for
        mask (numpy.ndarray): mask of lesion
        clusters (int, optional): amound of clusters and therefore dominent colors (defualt 3)
        include_ratios (bool, optional): whether to include the domination ratios for each color (defualt False)

    Return:  
        if include_ratios == True: 
            p_and_c (list): list of tuples, each containing the percentage and RGB array of the dominent color
        else: 
            dom_colors (array): array of RGB arrays of each dominent color.
    '''
    
    # Prepare image for KMeans
    cut_im = cut_im_by_mask(image, mask) # Cut image to remove excess skin pixels
    hsv_im = rgb2hsv(cut_im) # Convert image to HSV color space
    flat_im = np.reshape(hsv_im, (-1, 3)) # Flatten image to 2D array

    # Use KMeans to cluster image by colors
    k_means = KMeans(n_clusters=clusters, n_init=10, random_state=0)
    k_means.fit(flat_im)

    # Save cluster centers (dominant colors) in array
    dom_colors = np.array(k_means.cluster_centers_, dtype='float32') 

    if include_ratios:

        counts = np.unique(k_means.labels_, return_counts=True)[1] # Get count of each dominent color
        ratios = counts / flat_im.shape[0] # Get percentage of total image for each dominent color

        r_and_c = zip(ratios, dom_colors) # Percentage and colors
        r_and_c = sorted(r_and_c, key=lambda x: x[0],reverse=True) # Sort in descending order

        return r_and_c
    
    return dom_colors

def plot_dominance_bar(r_and_c):
    '''Plot dominance bar from percentage and count list.
    Necessitates percentages in input.
    
    Args r_and_c: a list of tuples, each containing the percentage and RGB array of the dominent color
    '''

    bar = np.ones((50, 500, 3), dtype='float32')
    plt.figure(figsize=(12,8))
    plt.title('Proportions of Dominent Colors in the Image')
    start = 0
    i = 1
    for percentage, color in r_and_c:
        end = start + int(percentage * bar.shape[1])
        if i == len(r_and_c):
            bar[:, start:] = color[::-1]
        else:
            bar[:, start:end] = color[::-1]
        start = end
        i += 1

    plt.imshow(bar)
    plt.tick_params(
    axis='both',
    which='both',
    bottom=False,
    left=False,
    labelbottom=False,
    labelleft=False)



def get_relative_rgb_means(image, slic_segments):
    '''Get mean RGB values for each segment in a SLIC segmented image.

    Args:
        image (numpy.ndarray): original image
        slic_segments (numpy.ndarray): SLIC segmentation

    Returns:
        rgb_means (list): RGB mean values for each segment.
    '''

    max_segment_id = np.unique(slic_segments)[-1]

    rgb_means = []
    for i in range(0, max_segment_id + 1):

        #Create masked image where only specific segment is active
        segment = image.copy()
        segment[slic_segments != i] = -1

        #Get average RGB values from segment
        rgb_mean = np.mean(segment, axis = (0, 1), where = (segment != -1))
        
        rgb_means.append(rgb_mean) 

    rgb_means_lesion = np.mean(rgb_means[1:],axis=0)
    rgb_means_skin = np.mean(rgb_means[0])

    F1, F2, F3 = rgb_means_lesion/sum(rgb_means_lesion)
    F10, F11, F12 = rgb_means_lesion - rgb_means_skin
        
    return F1, F2, F3, F10, F11, F12