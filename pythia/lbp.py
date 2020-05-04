import numpy as np

from skimage.feature import local_binary_pattern

def hist(image):
    lbp_image = local_binary_pattern(
        image,
        8,  # n_points
        1  # radius
    )
    bins_values, _ = np.histogram(lbp_image, bins=255)
    return bins_values
    