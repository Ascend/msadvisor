import sys
import unittest

sys.path.append('src')

from model import evaluate


class TestNonexistDatapath(unittest.TestCase):
    def test_nonexist_datapath(self):
        self.assertRaises(FileNotFoundError, evaluate, 'data/project/nonexist', None)


if __name__ == "__main__":
    unittest.main()
