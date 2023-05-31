### Main model script ... Finish this part when script is done ...

###############
### IMPORTS ###
###############

# Standard Modules
import os
from os.path import isfile, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, cross_validate
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, make_scorer 

# Feature extraction (PCA)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Feature selection
from sklearn.feature_selection import SelectKBest,mutual_info_classif

# For loading and saving classifiers
import pickle as pk

# Image preperation
from prep_image import prep_im_and_mask

# Feature extraction
from asymmetry import mean_asymmetry, best_asymmetry, worst_asymmetry
from color import slic_segmentation, rgb_var, hsv_var, color_dominance, get_relative_rgb_means
from compactness import compactness_score
from convexity import convexity_score

###########################
### FEATURE EXTRACTIONS ###
###########################

def extract_features(im, im_mask):

	# Assymmetry
	mean_asym = mean_asymmetry(im_mask,4)
	best_asym = best_asymmetry(im_mask,4)
	worst_asym = worst_asymmetry(im_mask,4)

	# Color variance
	segments = slic_segmentation(im, im_mask, n_segments=250)
	red_var, green_var, blue_var = rgb_var(im, segments)
	hue_var, sat_var, val_var = hsv_var(im, segments)

	# Color dominance
	dom_colors = color_dominance(im, im_mask, clusters=5, include_ratios=True) # Extract five most dominent colors, sorted by percentage of total area
	dom_hue, dom_sat, dom_val = dom_colors[0][1]     

	# Compactness
	compactness = compactness_score(im_mask)

	# Convexity
	convexity = convexity_score(im_mask)

	# Relative color scores
	F1, F2, F3, F10, F11, F12 = get_relative_rgb_means(im, segments)

	return [mean_asym, best_asym, worst_asym, red_var, green_var, \
		blue_var, hue_var, sat_var, val_var, dom_hue, dom_sat, \
		dom_val, compactness, convexity, F1, F2, F3, F10, F11, F12]

########################
### IMAGE PROCESSING ###
########################

def ProcessImages(file_data, image_folder, mask_folder, file_features, feature_names):
	# Import metadata from file
	df = pd.read_csv(file_data)

	# Remove images without masks
	df_mask = df['mask'] == 1
	df = df.loc[df_mask]

	# Features to extract
	features_n = len(feature_names)
	
	features = np.zeros(shape = [len(df), features_n], dtype = np.float16)
	img_ids = []
	
	# Extract features
	for i, id in enumerate(list(df['img_id'])):
		im, mask = prep_im_and_mask(id, image_folder, mask_folder)
		
		# Extract features
		x = extract_features(im, mask)
		img_ids.append(id)
		features[i,:] = x
		print(f"Done {i+1} image out of {len(list(df['img_id'])) + 1} images")

	# Insert image ids
	df_features = pd.DataFrame(features, columns = feature_names)
	df_features.insert(0, 'img_id', img_ids)

	# Insert patient ids
	df_features.insert(0, 'patient_id', df['patient_id'].tolist())

	# Save feature data as csv file
	df_features.to_csv(file_features, index = False)

#########################
### FEATURE SELECTION ###
#########################

def train_feature_selector(X_train, y_train, k):
    '''Using SelectKBest to extract features from train_X, down to k features as output
    Returns a selector object (which is applied to X_train and X_test afterwards) 
	and the score for each feature.

    Args:
        train_X (pandas.DataFrame): Data Frame of features from X_train.
		train_y (pandas.DataFrame): Data Frame of target values from y_train.
        k (int): Number of features to output.

    Returns:
		feature_selector (selector object): 
        scores (numpy.ndarray): Array containg scores for each feature.    
    '''

    feature_selector = SelectKBest(mutual_info_classif, k=k)
    feature_selector.fit_transform(X_train, y_train)
    
    pk.dump(feature_selector, open('selector.pkl', 'wb'))
    
    return feature_selector

def apply_feature_selector(X):

	feature_selector = pk.load(open('selector.pkl', 'rb'))
	X_transformed = feature_selector.transform(X)

	return X_transformed

def feature_scores(feature_selector):
    '''Using SelectKBest to extract features from train_X, down to k features as output
    Returns a selector object (which is applied to X_train and X_test afterwards) 
	and the score for each feature.

    Args:
        train_X (pandas.DataFrame): Data Frame of features from X_train.
		train_y (pandas.DataFrame): Data Frame of target values from y_train.
        k (int): Number of features to output.

    Returns:
		feature_selector (selector object): 
        scores (numpy.ndarray): Array containg scores for each feature.    
    '''
    
    scores = feature_selector.scores_
    
    return scores

###########
### PCA ###
###########

# Standardize feature data
def std_X(X):
	'''Standardized the input.

    Args:
        X (pandas.DataFrame): Data Frame of features.

    Returns:
        X_std (numpy.ndarray): Array containg standardized features.    
    '''

	std_scl = StandardScaler()
	X_std = std_scl.fit_transform(X)

	return X_std

def train_pca(X, n=0.95):
	'''Train PCA. Save the result.

	Args:
	    X (pandas.DataFrame): Data Frame of features.
	    n (float): Percentage of variation that should be explained by the chosen features.

	Returns:
		Nothing
	'''

	#X_std = std_X(X)
	X_normalized = (X - X.mean()) / X.std()
	pca = PCA(n_components=n)
	pca.fit_transform(X_normalized)

	pk.dump(pca, open('pca.pkl', 'wb'))

def apply_pca(X):
	'''Apply pca to X.

	Args:
	    X (pandas.DataFrame): Data Frame of features.
	    n (float): Percentage of variation that should be explained by the chosen features.

	Returns:
	    X_std_pca (numpy.ndarray): Array containg transformed, standardized features.    
	'''

	pca = pk.load(open('pca.pkl', 'rb'))

	#X_std =std_X(X)
	X_normalized = (X - X.mean()) / X.std()
	X_transformed = pca.transform(X_normalized)

	return X_transformed

########################
### TRAIN CLASSIFIER ###
########################

def cross_validate_clf(X, y, classifiers, groups):

	# Scores for evaluation
	scores ={'accuracy': make_scorer(accuracy_score), 'sensitivity': make_scorer(recall_score), 'specificity': make_scorer(recall_score, pos_label=0), 'precision': make_scorer(precision_score), 'roc_auc': make_scorer(roc_auc_score)}

	num_folds = 5
	cross_val = StratifiedGroupKFold(n_splits= num_folds)	

	evaluation_results = {}
	for classifier in classifiers:
		cv_results = cross_validate(classifier, X, y, scoring=scores, cv=cross_val, groups = groups)
		
		if type(classifier).__name__ == "KNeighborsClassifier":
			classifier_name = type(classifier).__name__
			params_dict = classifier.get_params()
			n_neigbors = params_dict["n_neighbors"]
			classifier_name = f"{classifier_name} with n_neighbors={n_neigbors}"
		else:
			classifier_name = type(classifier).__name__

		evaluation_results[classifier_name] = {
            'Accuracy': cv_results['test_accuracy'].mean(),
            'Sensitivity': cv_results['test_sensitivity'].mean(),
            'Specificity': cv_results['test_specificity'].mean(),
            'Precision': cv_results['test_precision'].mean(),
            'ROC AUC': cv_results['test_roc_auc'].mean()

        }

	return evaluation_results


def print_results_cv(evaluation_results):

	for classifier, scores in evaluation_results.items():
		print(classifier)
		for metric, score in scores.items():
			print(f'{metric}: {score:.4f}')
	    
		print()


def train_clf(X_train, y_train, classifiers):

	trained_classifiers = [classifier.fit(X_train, y_train) for classifier in classifiers]

	return trained_classifiers


def evaluate_clf(X_test, y_test, trained_classifiers):
	# Take trained classifiers as inputs

	results = {}
	for clf in trained_classifiers:
		y_pred = clf.predict(X_test)

		if type(clf).__name__ == "KNeighborsClassifier":
			classifier_name = type(clf).__name__
			params_dict = clf.get_params()
			n_neigbors = params_dict["n_neighbors"]
			classifier_name = f"{classifier_name} with n_neighbors: {n_neigbors}"
		else:
			classifier_name = type(clf).__name__
  
		results[classifier_name] = {
            'Accuracy': round(accuracy_score(y_test, y_pred), 3),
            'Sensitivity': round(recall_score(y_test, y_pred, pos_label = 1), 3),
            'Specificity': round(recall_score(y_test, y_pred, pos_label = 0),3),
            'Precision': round(precision_score(y_test, y_pred), 3),
        	'AUC ROC': round(roc_auc_score(y_test, y_pred), 3)
        }
    
	return results

def print_results(results):
	for classifier, scores in results.items():
		print(classifier)
		for metric, score in scores.items():
			print(f'{metric}: {score:.4f}')
	    
		print()
