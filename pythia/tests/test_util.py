#pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
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
        
    def test_section_no_cell(self):
        self.assertEqual(
            self.sections[0][1],
            "normal cell",
            "Section 0 is incorrected classify"
        )
    
    def test_section_lsil(self):
        self.assertEqual(
            self.sections[1][1],
            "altered cell",
            "Section 1 is incorrected classify"
        )

if __name__ == '__main__':
    unittest.main()
