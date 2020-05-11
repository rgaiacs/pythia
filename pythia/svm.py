"""Support Vector Machine"""
from sklearn import svm

from . import util

class SVC():
    """Support Vector Classification"""
    def __init__(self, data_filename, image_folder):
        self.clf = svm.SVC()
        points, classification = util.collection2sections_and_classes(
            data_filename,
            image_folder
        )
        self.clf.fit(points, classification)

    def predict_sample(self, sample):
        """Predict classification from sample"""
        return self.clf.predict([
            util.sample2features(sample)
        ])[0]
