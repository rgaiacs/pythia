from sklearn import svm

from . import io
from . import util
from . import lbp
    
def sample2features(sample):
    features = []
    
    # Local binary pattern
    features.extend(lbp.hist(sample))
    
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
        self.clf.predict([
            sample2features(sample)
        ])