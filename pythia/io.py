"""IO module"""
import json

from joblib import dump, load

import skimage

def imread(filename):
    """Read image file to memory.

    Parameters
    ----------
    filename : str
        Name of the image file to read.

    Returns
    -------
    image
        Image as Numpy array.
    """
    return skimage.io.imread(filename)

def imread2gray(filename):
    """Read image file to memory and convert to gray.

    Parameters
    ----------
    filename : str
        Name of the image file to load.

    Returns
    -------
    image
        Image as Numpy array.
    """
    image = skimage.io.imread(
        filename,
        as_gray=True
    )
    return skimage.img_as_ubyte(image)

def imsave(filename, image):
    """Save image to file.

    Parameters
    ----------
    filename : str
        Name of the image file to write.
    image : Numpy arra
        Image as Numpy array.
    """
    skimage.io.imsave(filename, image)

def imshow(image, gray=False):
    """Show image.

    Parameters
    ----------
    image : Numpy array
        Image
    gray: boolean
        If to use gray scale.
    """
    if gray:
        skimage.io.imshow(
            image,
            cmap='gray'
        )
    else:
        skimage.io.imshow(image)

def jsonread(filename):
    """Read json file to memory.

    Parameters
    ----------
    filename : str
        Name of the json file to load.

    Returns
    -------
    json
        JSON object.
    """
    with open(filename, "r") as _file:
        data = json.load(_file)
    return data

def clfread(filename):
    """Read SVC classifier.

    Parameters
    ----------
    filename : str
        Name of the SVC classifier file to read.

    Returns
    -------
    image
        Image as Numpy array.
    """
    return load(filename)

def clfsave(filename, clf):
    """Save SVC classifier to file.

    Parameters
    ----------
    filename : str
        Name of the image file to write.
    clf : SVC classifier
        Scikit-learn SVC
    """
    dump(
        clf,
        filename
    )
