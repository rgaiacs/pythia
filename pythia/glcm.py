"""GLC"""
import numpy as np
from skimage.feature import greycomatrix

def hist(image, number_gray_levels=8):
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
            np.arange(
                0,  # Start of interval
                256,  # End of interval
                256 / number_gray_levels  # Spacing between values
            )
        )
    ) - 1

    glcm_image = greycomatrix(
        binned,
        [1],  # distances
        [0],  # angles
        number_gray_levels
    )
    return glcm_image[:, :, 0, 0].ravel()
