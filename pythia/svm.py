from sklearn import svm

from . import io
from . import util

from . import hog
from . import glcm
from . import lbp
    
def sample2features(sample):
    features = []
    
    # Local binary pattern
    features.extend(lbp.hist(sample))
    
    # Histogram of Oriented Gradients
    features.extend(hog.hist(sample))
    
    # Gray-Level Co-Occurrence Matrix
    features.extend(glcm.hist(sample))
    
    return features

def setup(data_filename, image_folder):
    X = []
    y = []
    
    data = io.jsonread(
        data_filename
    )
    
    for datum in data:
        image = io.imread2gray(
            "{}/{}".format(
                image_folder,
                datum["image_name"]
            )
        )
        for section, classification in util.image2sections_and_classes(image, datum["classifications"]):
            X.append(
                sample2features(section)
            )
            y.append(classification)
            
    return (X, y)

class SVC():
    def __init__(self, data_filename, image_folder):
        self.clf = svm.SVC()
        X, y = setup(data_filename, image_folder)
        self.clf.fit(X, y)

    def predict_sample(self, sample):
        return self.clf.predict([
            sample2features(sample)
        ])[0]