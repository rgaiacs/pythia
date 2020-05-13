"""GLC"""
import numpy as np
from skimage.feature import greycomatrix

def hist(image):
    """Create histogram"""
    # As the output matrix is at least levels x levels,
    # it might be preferable to use binning of the input image
    # rather than large values for levels.
    #
    # Use of NumPy's digitize as suggested in
    # https://stackoverflow.com/a/44051205/1802726
    binned = np.uint8(
        np.digitize(
            image,
            np.arange(0, 256, 8)
        )
    ) - 1

    glcm_image = greycomatrix(
        binned,
        [1],  # distances
        [0]  # angles
    )
    return glcm_image[:, :, 0, 0].ravel()
    