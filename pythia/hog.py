"""Hog"""
import numpy as np

from skimage.feature import hog

def hist(image):
    """Calculate HOG histogram"""
    hog_image = hog(
        image,
        feature_vector=True
    )
    bins_values, _ = np.histogram(hog_image, bins=9)
    return bins_values
    