"""Support Vector Machine"""
from sklearn import svm

from . import util

class SVC():
    """Support Vector Classification"""
    def __init__(
        self,
        data_filename,
        image_folder,
        section_size=100,
        crop_grid=True,
        crop_center=True):
        self.clf = svm.SVC(
            kernel="poly",
            degree=2,
            class_weight="balanced",
            max_iter=10_000
        )
        points, classification = util.collection2features_and_classes(
            data_filename,
            image_folder,
            section_size=section_size,
            crop_grid=True,
            crop_center=True
        )
        self.clf.fit(points, classification)

    def predict_sample(self, sample):
        """Predict classification from sample"""
        return self.clf.predict([
            util.sample2features(sample)
        ])[0]
