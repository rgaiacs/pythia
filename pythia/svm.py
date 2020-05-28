"""Support Vector Machine"""
from sklearn import svm

from . import util

class SVC():
    """Support Vector Classification"""
    def __init__(self, data_filename, image_folder, section_size=100):
        self.clf = svm.SVC()
        points, classification = util.collection2features_and_classes(
            data_filename,
            image_folder,
            section_size=section_size
        )
        self.clf.fit(points, classification)

    def predict_sample(self, sample):
        """Predict classification from sample"""
        return self.clf.predict([
            util.sample2features(sample)
        ])[0]
