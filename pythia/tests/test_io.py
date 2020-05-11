#pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import os

import unittest

from pythia import io

class TestIO(unittest.TestCase):
    def test_imread(self):
        image = io.imread(
            os.path.dirname(__file__) + "/9ae8a4edde40219bad6303cebc672ee4.png"
        )
        self.assertEqual(
            image.shape,
            (1020, 1376, 3),
            "incorrect shape for image"
        )

if __name__ == '__main__':
    unittest.main()
