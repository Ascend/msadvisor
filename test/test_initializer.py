import unittest

import numpy as np

from auto_opt_tool.onnx.node import Initializer
from auto_optimizer.test.test_node_common import create_node

class TestInitializer(unittest.TestCase):

    def test_initializer_get_value(self):
        ini = create_node('Initializer')
        self.assertTrue(np.array_equal(ini.value, np.array([[1,2,3,4,5]], dtype='int32'), equal_nan=True))
        self.assertEqual(ini.value.dtype, 'int32')

    def test_initializer_set_value(self):
        ini = create_node('Initializer')
        ini.value = np.array([[7,8,9], [10,11,12]], dtype='float32')
        self.assertTrue(np.array_equal(ini.value, np.array([[7,8,9], [10,11,12]], dtype='float32'), equal_nan=True))
        self.assertEqual(ini.value.dtype, 'float32')

if __name__ == "__main__":
    unittest.main()