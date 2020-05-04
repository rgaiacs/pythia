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
        
    def test_section(self):
        self.assertEqual(
            self.svc.predict_sample(self.image[0:100, 0:100]),
            "LSIL",
            "incorrect prediction"
        )

if __name__ == '__main__':
    unittest.main()