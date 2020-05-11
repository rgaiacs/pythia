"""GLC"""
from skimage.feature import greycomatrix

def hist(image):
    """Create histogram"""
    glcm_image = greycomatrix(
        image,
        [1],  # distances
        [0]  # angles
    )
    return glcm_image[:, :, 0, 0].ravel()
    