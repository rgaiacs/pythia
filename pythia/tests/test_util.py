import os

import unittest

from pythia import io
from pythia import util

class TestUtil(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = io.imread(
            os.path.dirname(__file__) + "/9ae8a4edde40219bad6303cebc672ee4.png"
        )
        cls.json = io.jsonread(
            os.path.dirname(__file__) + "/classifications.json"
        )
        cls.sections = util.image2sections_and_classes(
            cls.image,
            cls.json[0]["classifications"]
        )
        
    def test_section(self):
        self.assertEqual(
            1,
            1,
            "incorrect section"
        )

if __name__ == '__main__':
    unittest.main()