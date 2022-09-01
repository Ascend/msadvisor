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

def create_graph():
    input_0 = OnnxPlaceHolder('input_0', np.dtype('float32'), [1,3,224,224])
    output_0 = OnnxPlaceHolder('output_0', np.dtype('float32'), [1,3,224,224])
    ini_0 = OnnxInitializer('ini_0', np.array([1,2,3], dtype='float32'))
    node_0 = OnnxNode('Node_0', 'Sub', inputs=['input_0'], outputs=['0_out_0', '0_out_1'], attrs={})
    node_1 = OnnxNode('Node_1', 'Mul', inputs=['0_out_0'], outputs=['1_out_0'], attrs={})
    node_2 = OnnxNode('Node_2', 'Add', inputs=['0_out_0', '0_out_1'], outputs=['2_out_0', '2_out_1'], attrs={})
    node_3 = OnnxNode('Node_3', 'Sub', inputs=['1_out_0', 'ini_0'], outputs=['3_out_0'], attrs={})
    node_4 = OnnxNode('Node_4', 'Add', inputs=['1_out_0', '2_out_0'], outputs=['4_out_0'], attrs={})
    node_5 = OnnxNode('Node_5', 'Mul', inputs=['3_out_0', '4_out_0', '2_out_1'], outputs=['output_0'], attrs={})
    return OnnxGraph([node_0,node_1,node_2,node_3,node_4,node_5], [input_0], [output_0], [ini_0])

def create_graph_1():
    input_0 = OnnxPlaceHolder('input_0', np.dtype('float32'), [1,3,224,224])
    output_0 = OnnxPlaceHolder('0_out_0', np.dtype('float32'), [1,3,224,224])
    output_1 = OnnxPlaceHolder('3_out_0', np.dtype('float32'), [1,3,224,224])
    ini_0 = OnnxInitializer('ini_0', np.array([1], dtype='int32'))
    node_0 = OnnxNode('Node_0', 'Sqrt', inputs=['input_0'], outputs=['0_out_0'], attrs={})
    node_1 = OnnxNode('Node_1', 'Sqrt', inputs=['input_0'], outputs=['1_out_0'], attrs={})
    node_2 = OnnxNode('Node_2', 'Add', inputs=['0_out_0', 'ini_0'], outputs=['2_out_0'], attrs={})
    node_3 = OnnxNode('Node_3', 'Add', inputs=['2_out_0', '1_out_0'], outputs=['3_out_0'], attrs={})
    return OnnxGraph([node_0,node_1,node_2,node_3], [input_0], [output_0, output_1], [ini_0], name='graph_1')
    
class TestGraphCrud(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(OnnxNode, is_node_equal)
        self.addTypeEqualityFunc(OnnxPlaceHolder, is_ph_equal)
        self.addTypeEqualityFunc(OnnxInitializer, is_ini_equal)
        self.addTypeEqualityFunc(OnnxGraph, is_graph_equal)
        self.graph = create_graph()
        self.graph_1 = create_graph_1()

    # def test_graph_test(self):
    #     target = OnnxGraph()
    #     self.assertEqual(self.graph, target)
    #     # self.assertTrue(is_graph_equal(self.graph, target))

    def test_add_input(self):
        test_input = self.graph.add_input('test_input', 'float32', [1,3,224,224])
        self.assertEqual(self.graph['test_input'], test_input)
        self.assertEqual(self.graph.inputs, [self.graph['input_0'], test_input])

    def test_add_output(self):
        test_output = self.graph.add_output('test_output', 'float32', [1,3,224,224])
        self.assertEqual(self.graph['test_output'], test_output)
        self.assertEqual(self.graph.outputs, [self.graph['output_0'], test_output])

    def test_add_initializer(self):
        test_ini = self.graph.add_initializer('test_ini', np.array([1,2,3]))
        self.assertEqual(self.graph['test_ini'], test_ini)
        self.assertEqual(self.graph.initializers, [self.graph['ini_0'], test_ini])

    def test_add_node(self):
        test_node = self.graph.add_node('test_node', 'Add')
        self.assertEqual(self.graph['test_node'], test_node)
        self.assertEqual(
            self.graph.nodes, 
            [
                self.graph['Node_0'], 
                self.graph['Node_1'], 
                self.graph['Node_2'], 
                self.graph['Node_3'], 
                self.graph['Node_4'], 
                self.graph['Node_5'], 
                test_node
                ]
            )

    def test_get_nodes(self):
        self.assertEqual(self.graph.get_nodes('Mul'), [self.graph['Node_1'], self.graph['Node_5']])
        self.assertEqual(self.graph.get_nodes('Sub'), [self.graph['Node_0'], self.graph['Node_3']])
        self.assertEqual(self.graph.get_nodes('Add'), [self.graph['Node_2'], self.graph['Node_4']])

    def test_insert_node_before(self):
        test_node = self.graph.add_node('test_node', 'Add')
        self.graph.insert_node('Node_4', test_node, 0, 'before')
        self.assertEqual(test_node.inputs, ['1_out_0'])
        self.assertEqual(test_node.outputs, ['test_node/Node_4'])
        self.assertEqual(self.graph.get_next_nodes('1_out_0'), [self.graph['Node_3'], test_node])
        self.assertEqual(self.graph.get_prev_node('1_out_0'), self.graph['Node_1'])
        self.assertEqual(self.graph.get_next_nodes('test_node/Node_4'), [self.graph['Node_4']])
        self.assertEqual(self.graph.get_prev_node('test_node/Node_4'), test_node)

    def test_insert_node_after(self):
        test_node = self.graph.add_node('test_node', 'Add')
        self.graph.insert_node('Node_0', test_node, 0, 'after')
        self.assertEqual(test_node.inputs, ['Node_0/test_node'])
        self.assertEqual(test_node.outputs, ['0_out_0'])
        self.assertEqual(self.graph.get_next_nodes('Node_0/test_node'), [test_node])
        self.assertEqual(self.graph.get_prev_node('Node_0/test_node'), self.graph['Node_0'])
        self.assertEqual(self.graph.get_next_nodes('0_out_0'), [self.graph['Node_1'], self.graph['Node_2']])
        self.assertEqual(self.graph.get_prev_node('0_out_0'), test_node)

    def test_insert_node_after_graph_input(self):
        test_node = self.graph_1.add_node('test_node', 'Add')
        self.graph_1.insert_node('input_0', test_node, 0, 'after')
        self.assertEqual(test_node.inputs, ['input_0'])
        self.assertEqual(test_node.outputs, ['test_node/Node_0'])
        self.assertEqual(
            self.graph_1.get_next_nodes('test_node/Node_0'), 
            [self.graph_1['Node_0'], self.graph_1['Node_1']]
            )
        self.assertEqual(self.graph_1.get_prev_node('test_node/Node_0'), test_node)
        self.assertEqual(self.graph_1.get_next_nodes('input_0'), [test_node])

    def test_insert_node_after_graph_init(self):
        test_node = self.graph_1.add_node('test_node', 'Add')
        self.graph_1.insert_node('ini_0', test_node, 0, 'after')
        self.assertEqual(test_node.inputs, ['ini_0'])
        self.assertEqual(test_node.outputs, ['test_node/Node_2'])
        self.assertEqual(self.graph_1.get_next_nodes('test_node/Node_2'), [self.graph_1['Node_2']])
        self.assertEqual(self.graph_1.get_prev_node('test_node/Node_2'), test_node)
        self.assertEqual(self.graph_1.get_next_nodes('ini_0'), [test_node])

    def test_insert_node_before_graph_output(self):
        test_node = self.graph_1.add_node('test_node', 'Add')
        self.graph_1.insert_node('0_out_0', test_node, 0, 'before')
        self.assertEqual(test_node.inputs, ['Node_0/test_node'])
        self.assertEqual(test_node.outputs, ['0_out_0'])
        self.assertEqual(self.graph_1.get_next_nodes('Node_0/test_node'), [test_node, self.graph_1['Node_2']])
        self.assertEqual(self.graph_1.get_prev_node('Node_0/test_node'), self.graph['Node_0'])
        self.assertEqual(self.graph_1.get_next_nodes('0_out_0'), [])
        self.assertEqual(self.graph_1.get_prev_node('0_out_0'), test_node)

    def test_connect_node_case_0(self):
        test_node = self.graph_1.add_node('test_node', 'Sqrt')
        self.graph_1.connect_node(
            test_node,
            ['Node_1'],
            ['Node_3:1']
            )
        self.assertEqual(test_node.inputs, ['1_out_0'])
        self.assertEqual(test_node.outputs, ['test_node_out_0'])
        self.assertEqual(self.graph_1.get_next_nodes('1_out_0'), [test_node])
        self.assertEqual(self.graph_1.get_prev_node('1_out_0'), self.graph_1['Node_1'])
        self.assertEqual(self.graph_1.get_next_nodes('test_node_out_0'), [self.graph_1['Node_3']])
        self.assertEqual(self.graph_1.get_prev_node('test_node_out_0'), test_node)
    
    def test_connect_node_case_1(self):
        test_node = self.graph_1.add_node('test_node', 'Sqrt')
        self.graph_1.connect_node(
            test_node,
            ['Node_0'],
            ['Node_2;0_out_0']
            )
        self.assertEqual(test_node.inputs, ['test_node_in_0'])
        self.assertEqual(test_node.outputs, ['0_out_0'])
        self.assertEqual(self.graph_1.get_next_nodes('test_node_in_0'), [test_node])
        self.assertEqual(self.graph_1.get_prev_node('test_node_in_0'), self.graph_1['Node_0'])
        self.assertEqual(self.graph_1.get_next_nodes('0_out_0'), [self.graph_1['Node_2']])
        self.assertEqual(self.graph_1.get_prev_node('0_out_0'), test_node)

    def test_connect_node_case_2(self):
        test_node = self.graph_1.add_node('test_node', 'Add')
        self.graph_1.connect_node(
            test_node,
            ['Node_1', 'Node_2'],
            ['Node_3:0,1']
            )
        self.assertEqual(test_node.inputs, ['1_out_0', '2_out_0'])
        self.assertEqual(test_node.outputs, ['test_node_out_0'])
        self.assertEqual(self.graph_1.get_next_nodes('1_out_0'), [test_node])
        self.assertEqual(self.graph_1.get_prev_node('1_out_0'), self.graph_1['Node_1'])
        self.assertEqual(self.graph_1.get_next_nodes('2_out_0'), [test_node])
        self.assertEqual(self.graph_1.get_prev_node('2_out_0'), self.graph_1['Node_2'])
        self.assertEqual(self.graph_1.get_next_nodes('test_node_out_0'), [self.graph_1['Node_3']])
        self.assertEqual(self.graph_1.get_prev_node('test_node_out_0'), test_node)
    
    def test_connect_node_case_3(self):
        test_node = self.graph_1.add_node('test_node', 'Split', {'axis':1})
        test_ini = self.graph_1.add_initializer('test_ini', np.array([1,2]))
        self.graph_1.connect_node(
            test_node,
            ['Node_0', 'test_ini'],
            ['Node_2:1', 'Node_1']
            )
        self.assertEqual(test_node.inputs, ['0_out_0', 'test_ini'])
        self.assertEqual(test_node.outputs, ['test_node_out_0', 'test_node_out_1'])
        self.assertEqual(self.graph_1.get_next_nodes('0_out_0'), [self.graph_1['Node_2'], test_node])
        self.assertEqual(self.graph_1.get_prev_node('0_out_0'), self.graph_1['Node_0'])
        self.assertEqual(self.graph_1.get_next_nodes('test_ini'), [test_node])
        self.assertEqual(self.graph_1.get_prev_node('test_ini'), None)
        self.assertEqual(self.graph_1.get_next_nodes('test_node_out_0'), [self.graph_1['Node_2']])
        self.assertEqual(self.graph_1.get_prev_node('test_node_out_0'), test_node)
        self.assertEqual(self.graph_1.get_next_nodes('test_node_out_1'), [self.graph_1['Node_1']])
        self.assertEqual(self.graph_1.get_prev_node('test_node_out_1'), test_node)
    
    # single input & single output
    def test_graph_remove_defualt(self):
        # create target
        input_0 = OnnxPlaceHolder('input_0', np.dtype('float32'), [1,3,224,224])
        output_0 = OnnxPlaceHolder('output_0', np.dtype('float32'), [1,3,224,224])
        ini_0 = OnnxInitializer('ini_0', np.array([1,2,3], dtype='float32'))
        node_0 = OnnxNode('Node_0', 'Sub', inputs=['input_0'], outputs=['0_out_0', '0_out_1'], attrs={})
        node_1 = OnnxNode('Node_1', 'Mul', inputs=['0_out_0'], outputs=['1_out_0'], attrs={})
        node_2 = OnnxNode('Node_2', 'Add', inputs=['0_out_0', '0_out_1'], outputs=['2_out_0', '2_out_1'], attrs={})
        node_4 = OnnxNode('Node_4', 'Add', inputs=['1_out_0', '2_out_0'], outputs=['4_out_0'], attrs={})
        node_5 = OnnxNode('Node_5', 'Mul', inputs=['1_out_0', '4_out_0', '2_out_1'], outputs=['output_0'], attrs={})
        target = OnnxGraph([node_0,node_1,node_2,node_4,node_5], [input_0], [output_0], [ini_0])

        self.graph.remove('Node_3')
        self.assertEqual(self.graph, target)

    def test_graph_remove_ini_node(self):
        # create target
        input_0 = OnnxPlaceHolder('input_0', np.dtype('float32'), [1,3,224,224])
        output_0 = OnnxPlaceHolder('output_0', np.dtype('float32'), [1,3,224,224])
        node_0 = OnnxNode('Node_0', 'Sub', inputs=['input_0'], outputs=['0_out_0', '0_out_1'], attrs={})
        node_1 = OnnxNode('Node_1', 'Mul', inputs=['0_out_0'], outputs=['1_out_0'], attrs={})
        node_2 = OnnxNode('Node_2', 'Add', inputs=['0_out_0', '0_out_1'], outputs=['2_out_0', '2_out_1'], attrs={})
        node_4 = OnnxNode('Node_4', 'Add', inputs=['1_out_0', '2_out_0'], outputs=['4_out_0'], attrs={})
        node_5 = OnnxNode('Node_5', 'Mul', inputs=['1_out_0', '4_out_0', '2_out_1'], outputs=['output_0'], attrs={})
        target = OnnxGraph([node_0,node_1,node_2,node_4,node_5], [input_0], [output_0])

        self.graph.remove('ini_0')
        self.graph.remove('Node_3')
        self.assertEqual(self.graph, target)

    def test_getitem(self):
        for node in chain(
            self.graph.inputs, 
            self.graph.outputs, 
            self.graph.nodes, 
            self.graph.initializers, 
            self.graph.value_infos
        ):
            self.assertIs(self.graph[node.name], node)

if __name__ == '__main__':
    unittest.main()
