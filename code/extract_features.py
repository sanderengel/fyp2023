from asymmetry import mean_asymmetry, best_asymmetry, worst_asymmetry
from color import slic_segmentation, rgb_var, hsv_var, color_dominance, get_relative_rgb_means
from compactness import compactness_score
from convexity import convexity_score

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