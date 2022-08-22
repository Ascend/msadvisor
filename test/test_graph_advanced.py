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
from itertools import chain

from auto_optimizer.graph_refactor.onnx.node import PlaceHolder, Initializer, Node
from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from test_node_common import is_ph_equal, is_ini_equal, is_node_equal
from test_graph_basic import is_graph_equal
from test_graph_crud import create_graph

class TestGraphAdvanced(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(Node, is_node_equal)
        self.addTypeEqualityFunc(PlaceHolder, is_ph_equal)
        self.addTypeEqualityFunc(Initializer, is_ini_equal)
        self.addTypeEqualityFunc(Graph, is_graph_equal)
        self.graph = creat_graph()

    def test_get_prev_node(self):
        self.assertIs(self.graph.get_prev_node('2_out_0'), self.graph['Node_2'])
        self.assertIs(self.graph.get_prev_node('2_out_1'), self.graph['Node_2'])

    def test_get_next_nodes(self):
        self.assertEqual(self.graph.get_next_nodes('0_out_0'), [self.graph['Node_1'], self.graph['Node_2']])

    def test_infershape(self):
        input_0 = PlaceHolder('input_0', np.dtype('float32'), [1,  10000])
        output_0 = PlaceHolder('output_0', np.dtype('float32'), [5000, 2])
        ini_0 = Initializer('ini_0', np.array([2, 5000]), dtype='int32')
        ini_1 = Initializer('ini_0', np.array([4, 2500]), dtype='int32')
        ini_2 = Initializer('ini_0', np.array([5000, 2]), dtype='int32')
        node_0 = Node('Node_0', 'Reshape', inputs=['input_0', 'ini_0'], outputs=['0_out_0'], attrs={})
        node_1 = Node('Node_1', 'Reshape', inputs=['0_out_0', 'ini_1'], outputs=['1_out_0'], attrs={})
        node_2 = Node('Node_2', 'Reshape', inputs=['1_out_0', 'ini_2'], outputs=['output_0'], attrs={})
        graph = OnnxGraph([node_0,node_1,node_2], [input_0], [output_0])

        graph.infershape()
        self.assertEqual(graph['0_out_0'], PlaceHolder('0_out_0', np.dtype('int32'), [2, 5000]))
        self.assertEqual(graph['1_out_0'], PlaceHolder('1_out_0', np.dtype('int32'), [4, 2500]))

if __name__ == "main":
    unittest.main()

