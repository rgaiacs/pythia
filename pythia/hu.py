"""Hu's set of image moments"""
from skimage.measure import moments_central, moments_normalized, moments_hu

def hist(image):
    """Create histogram"""
    return moments_hu(
        moments_normalized(
            moments_central(image)
        )
    )
