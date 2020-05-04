import os

import unittest

from pythia import svm

class TestUtil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf = svm.svc(
            os.path.dirname(__file__) + "/classifications.json",
            os.path.dirname(__file__)
        )
        
    def test_section(self):
        self.assertEqual(
            1,
            1,
            "incorrect section"
        )

if __name__ == '__main__':
    unittest.main()