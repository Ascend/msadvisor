# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

from auto_optimizer.graph_refactor.onnx.node import OnnxPlaceHolder
from test_node_common import create_node


class TestPlaceHolder(unittest.TestCase):

    def test_placeholder_get_dtype(self):
        ph = create_node('OnnxPlaceHolder')
        self.assertEqual(ph.dtype, np.dtype('float32'))

    def test_placeholder_set_dtype(self):
        ph = create_node('OnnxPlaceHolder')
        ph.dtype = np.dtype('float16')
        self.assertEqual(ph.dtype, np.dtype('float16'))

    def test_placeholder_get_shape(self):
        ph = create_node('OnnxPlaceHolder')
        self.assertEqual(ph.shape, [1, 3, 224, 224])

    def test_placeholder_set_shape(self):
        ph = create_node('OnnxPlaceHolder')
        ph.shape = [-1, 3, 224, 224]
        self.assertEqual(ph.shape, ['-1', 3, 224, 224])


if __name__ == "__main__":
    unittest.main()
