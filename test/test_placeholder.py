import unittest

import numpy as np

from auto_optimizer.graph_refactor.onnx.node import PlaceHolder
from auto_optimizer.test.test_node_common import create_node

class TestPlaceHolder(unittest.TestCase):

    def test_placeholder_get_dtype(self):
        ph = create_node('PlaceHolder')
        self.assertEqual(ph.dtype, np.dtype('float32'))

    def test_placeholder_set_dtype(self):
        ph = create_node('PlaceHolder')
        ph.dtype = np.dtype('float16')
        self.assertEqual(ph.dtype, np.dtype('float16'))

    def test_placeholder_get_shape(self):
        ph = create_node('PlaceHolder')
        self.assertEqual(ph.shape, [1,3,224,224])

    def test_placeholder_set_shape(self):
        ph = create_node('PlaceHolder')
        ph.shape = [-1,3,224,224]
        self.assertEqual(ph.shape, [-1,3,224,224])

if __name__ == "__main__":
    unittest.main()