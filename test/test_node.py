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

from onnx import helper

from auto_optimizer.graph_refactor.onnx.node import OnnxNode
from test_node_common import create_node


class TestNode(unittest.TestCase):

    def test_node_op_type(self):
        node = create_node('OnnxNode')
        self.assertEqual(node.op_type, 'Conv')

    def test_node_get_inputs(self):
        node = create_node('OnnxNode')
        self.assertEqual(node.inputs, ['0', '1'])

    def test_node_set_inputs(self):
        node = create_node('OnnxNode')
        node.inputs = ['0', '3']
        self.assertEqual(node.inputs, ['0', '3'])

    def test_node_get_input_id(self):
        node = create_node('OnnxNode')
        self.assertEqual(node.get_input_id('0'), 0)
        self.assertEqual(node.get_input_id('1'), 1)

    def test_node_get_outputs(self):
        node = create_node('OnnxNode')
        self.assertEqual(node.outputs, ['2'])

    def test_node_set_outputs(self):
        node = create_node('OnnxNode')
        node.outputs = ['5']
        self.assertEqual(node.outputs, ['5'])

    def test_node_get_input_id(self):
        node = create_node('OnnxNode')
        self.assertEqual(node.get_output_id('2'), 0)

    def test_node_get_attrs(self):
        node = create_node('OnnxNode')
        self.assertEqual(node.attrs, {'kernel_shape': 3})

    def test_node_get_attr(self):
        node = create_node('OnnxNode')
        self.assertEqual(node['kernel_shape'], 3)

    def test_node_set_attr(self):
        node = create_node('OnnxNode')
        node['kernel_shape'] = 5
        self.assertEqual(node['kernel_shape'], 5)

    def test_node_set_attr(self):
        node = create_node('OnnxNode')
        node['kernel_shape'] = 5
        self.assertEqual(node['kernel_shape'], 5)

    def test_node_domain(self):
        node = create_node('OnnxNode')
        self.assertEqual(node.domain, '')


if __name__ == "__main__":
    unittest.main()
