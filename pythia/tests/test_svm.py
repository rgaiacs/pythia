#pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import os

import unittest

from pythia import io
from pythia import svm

class TestUtil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = svm.SVC(
            os.path.dirname(__file__) + "/classifications.json",
            os.path.dirname(__file__)
        )
        cls.image = io.imread2gray(
            os.path.dirname(__file__) + "/9ae8a4edde40219bad6303cebc672ee4.png"
        )
        
    def test_prediction_no_cell(self):
        self.assertEqual(
            self.svc.predict_sample(self.image[0:100, 0:100]),
            "No cell",
            "incorrect prediction for cell"
        )
    
    def test_prediction_lsil(self):
        self.assertEqual(
            self.svc.predict_sample(self.image[0:100, 100:200]),
            "LSIL",
            "incorrect prediction for LSIL"
        )
        
    def test_prediction_hsil(self):
        self.assertEqual(
            self.svc.predict_sample(self.image[300:400, 100:200]),
            "HSIL",
            "incorrect prediction for HSIL"
        )
        
    def test_prediction_asch(self):
        self.assertEqual(
            self.svc.predict_sample(self.image[0:100, 300:400]),
            "ASC-H",
            "incorrect prediction for ASC-H"
        )
        
    def test_prediction_scc(self):
        self.assertEqual(
            self.svc.predict_sample(self.image[300:400, 400:500]),
            "SCC",
            "incorrect prediction for SCC"
        )

if __name__ == '__main__':
    unittest.main()
