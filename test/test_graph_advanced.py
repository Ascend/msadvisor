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

from itertools import chain

import unittest
import numpy as np

from auto_optimizer.graph_refactor.onnx.node import OnnxPlaceHolder, OnnxInitializer, OnnxNode
from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from test_node_common import is_ph_equal, is_ini_equal, is_node_equal
from test_graph_basic import is_graph_equal
from test_graph_crud import create_graph

class TestGraphAdvanced(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(OnnxNode, is_node_equal)
        self.addTypeEqualityFunc(OnnxPlaceHolder, is_ph_equal)
        self.addTypeEqualityFunc(OnnxInitializer, is_ini_equal)
        self.addTypeEqualityFunc(OnnxGraph, is_graph_equal)
        self.graph = create_graph()

    def test_get_prev_node(self):
        self.assertIs(self.graph.get_prev_node('2_out_0'), self.graph['Node_2'])
        self.assertIs(self.graph.get_prev_node('2_out_1'), self.graph['Node_2'])

    def test_get_next_nodes(self):
        self.assertEqual(self.graph.get_next_nodes('0_out_0'), [self.graph['Node_1'], self.graph['Node_2']])

    def test_infershape(self):
        input_0 = OnnxPlaceHolder('input_0', np.dtype('float32'), [1,  10000])
        output_0 = OnnxPlaceHolder('output_0', np.dtype('float32'), [5000, 2])
        ini_0 = OnnxInitializer('ini_0', np.array([2, 5000], dtype='int64'))
        ini_1 = OnnxInitializer('ini_1', np.array([4, 2500], dtype='int64'))
        ini_2 = OnnxInitializer('ini_2', np.array([5000, 2], dtype='int64'))
        node_0 = OnnxNode('Node_0', 'Reshape', inputs=['input_0', 'ini_0'], outputs=['0_out_0'], attrs={})
        node_1 = OnnxNode('Node_1', 'Reshape', inputs=['0_out_0', 'ini_1'], outputs=['1_out_0'], attrs={})
        node_2 = OnnxNode('Node_2', 'Reshape', inputs=['1_out_0', 'ini_2'], outputs=['output_0'], attrs={})
        graph = OnnxGraph([node_0,node_1,node_2], [input_0], [output_0], [ini_0,ini_1,ini_2], name='test')

        graph.infershape()
        self.assertEqual(graph.get_value_info('0_out_0'), OnnxPlaceHolder('0_out_0', np.dtype('float32'), [2, 5000]))
        self.assertEqual(graph.get_value_info('1_out_0'), OnnxPlaceHolder('1_out_0', np.dtype('float32'), [4, 2500]))
    
    def test_remove_unused_node(self):
        unused_node = self.graph.add_node('unused_node', 'Add')
        self.graph.remove_unused_nodes()
        self.assertEqual(
            self.graph.nodes, 
            [
                self.graph['Node_0'], 
                self.graph['Node_1'], 
                self.graph['Node_2'], 
                self.graph['Node_3'], 
                self.graph['Node_4'], 
                self.graph['Node_5'], 
                ]
            )


if __name__ == "__main__":
    unittest.main()