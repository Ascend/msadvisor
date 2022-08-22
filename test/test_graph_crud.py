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

def create_graph():
    input_0 = PlaceHolder('input_0', np.dtype('float32'), [1,3,224,224])
    output_0 = PlaceHolder('output_0', np.dtype('float32'), [1,3,224,224])
    node_0 = Node('Node_0', 'Sub', inputs=['input_0'], outputs=['0_out_0', '0_out_1'], attrs={})
    node_1 = Node('Node_1', 'Mul', inputs=['0_out_0'], outputs=['1_out_0'], attrs={})
    node_2 = Node('Node_2', 'Add', inputs=['0_out_0', '0_out_1'], outputs=['2_out_0', '2_out_1'], attrs={})
    node_3 = Node('Node_3', 'Sub', inputs=['1_out_0'], outputs=['3_out_0'], attrs={})
    node_4 = Node('Node_4', 'Add', inputs=['1_out_0', '2_out_0'], outputs=['4_out_0'], attrs={})
    node_5 = Node('Node_5', 'Mul', inputs=['3_out_0', '4_out_0', '2_out_1'], outputs=['output_0'], attrs={})
    return OnnxGraph([node_0,node_1,node_2,node_3,node_4,node_5], [input_0], [output_0])

class TestGraphCrud(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(Node, is_node_equal)
        self.addTypeEqualityFunc(PlaceHolder, is_ph_equal)
        self.addTypeEqualityFunc(Initializer, is_ini_equal)
        self.addTypeEqualityFunc(Graph, is_graph_equal)
        self.graph = creat_graph()

    # single input & single output
    def test_graph_remove_defualt(self):
        # create target
        input_0 = PlaceHolder('input_0', np.dtype('float32'), [1,3,224,224])
        output_0 = PlaceHolder('output_0', np.dtype('float32'), [1,3,224,224])
        node_0 = Node('Node_0', 'Sub', inputs=['input_0'], outputs=['0_out_0', '0_out_1'], attrs={})
        node_1 = Node('Node_1', 'Mul', inputs=['0_out_0'], outputs=['1_out_0'], attrs={})
        node_2 = Node('Node_2', 'Add', inputs=['0_out_0', '0_out_1'], outputs=['2_out_0', '2_out_1'], attrs={})
        node_4 = Node('Node_4', 'Add', inputs=['1_out_0', '2_out_0'], outputs=['4_out_0'], attrs={})
        node_5 = Node('Node_5', 'Mul', inputs=['1_out_0', '4_out_0', '2_out_1'], outputs=['output_0'], attrs={})
        target = OnnxGraph([node_0,node_1,node_2,node_4,node_5], [input_0], [output_0])

        self.graph.remove('Node_3')
        self.assertEqual(self.graph, target)

    def test_getitem(self):
        for node in chain(self.graph.inputs, self.graph.outputs, self.graph.nodes, self.graph.initializers, self.graph.value_infos):
            self.assertIs(self.graph[node.name], node)


if __name__ == "main":
    unittest.main()