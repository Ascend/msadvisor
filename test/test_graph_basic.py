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

import os
import random
import unittest
from typing import List

import numpy as np
from onnx import helper, numpy_helper, GraphProto, ModelProto, OperatorSetIdProto
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from auto_optimizer.graph_refactor.onnx.node import OnnxPlaceHolder, OnnxInitializer, OnnxNode
from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from test_node_common import is_ph_equal, is_ini_equal, is_node_equal


def is_elem_equal(elem1, elem2):
    if isinstance(elem1, OnnxPlaceHolder):
        return is_ph_equal(elem1, elem2)
    elif isinstance(elem1, OnnxInitializer):
        return is_ini_equal(elem1, elem2)
    else:
        return is_node_equal(elem1, elem2)

def is_list_equal(list1, list2):
    flag = True
    if len(list1) != len(list2):
        return False
    for idx, elem in enumerate(list1):
        flag &= is_elem_equal(elem, list2[idx])
    return flag

def is_map_equal(map1, map2):
    flag = True
    if map1.keys() != map2.keys():
        return False
    for key in map1.keys():
        if isinstance(map1[key], List):
            flag &= is_list_equal(map1[key], map2[key])
        else:
            flag &= is_elem_equal(map1[key], map2[key])
    return flag

def is_graph_equal(g1, g2, msg=None):
    if not is_list_equal(g1.nodes, g2.nodes):
        msg = 'graph nodes are not equal!'
        raise unittest.TestCase.failureException(msg)
    if not is_list_equal(g1.initializers, g2.initializers):
        msg = 'graph initializers are not equal!'
        raise unittest.TestCase.failureException(msg)
    if not is_list_equal(g1.inputs, g2.inputs):
        msg = 'graph inputs are not equal!'
        raise unittest.TestCase.failureException(msg)
    if not is_list_equal(g1.outputs, g2.outputs):
        msg = 'graph outputs are not equal!'
        raise unittest.TestCase.failureException(msg)
    if not is_list_equal(g1._value_infos, g2._value_infos):
        msg = 'graph value_infos are not equal!'
        raise unittest.TestCase.failureException(msg)
    if not is_map_equal(g1._node_map, g2._node_map):
        msg = 'graph node_map are not equal!'
        raise unittest.TestCase.failureException(msg)
    if not is_map_equal(g1._prev_map, g2._prev_map):
        msg = 'graph prev_map are not equal!'
        raise unittest.TestCase.failureException(msg)
    if not is_map_equal(g1._next_map, g2._next_map):
        msg = 'graph next_map are not equal!'
        raise unittest.TestCase.failureException(msg)
    return True

def create_graph():
    input_0 = OnnxPlaceHolder('input_0', np.dtype('float32'), [3,2])
    output_0 = OnnxPlaceHolder('output_0', np.dtype('float32'), [3,4])
    ini_0 = OnnxInitializer('ini_0', np.array([1,4], dtype='int32'))
    ini_1 = OnnxInitializer('ini_1', np.array([1], dtype='int32'))
    ini_2 = OnnxInitializer('const_0', np.array([1], dtype='int32'))
    node_0 = OnnxNode(
                'Node_0', 
                'Pad', 
                inputs=['input_0', 'ini_0', 'ini_1', 'const_0'], 
                outputs=['output_0'], 
                attrs={'mode':b'constant'}, 
                domain=''
    )
    return OnnxGraph(name='test_graph', nodes=[node_0], inputs=[input_0], outputs=[output_0], initializers=[ini_0, ini_1, ini_2])

def create_graph_1():
    input_0 = OnnxPlaceHolder('input_0', np.dtype('float32'), [1,3,224,224])
    output_0 = OnnxPlaceHolder('0_out_0', np.dtype('float32'), [1,3,224,224])
    output_1 = OnnxPlaceHolder('3_out_0', np.dtype('float32'), [1,3,224,224])
    ini_0 = OnnxInitializer('ini_0', np.array([1], dtype='int32'))
    node_0 = OnnxNode('Node_0', 'Sqrt', inputs=['input_0'], outputs=['0_out_0'], attrs={})
    node_1 = OnnxNode('Node_1', 'Sqrt', inputs=['input_0'], outputs=['1_out_0'], attrs={})
    node_2 = OnnxNode('Node_2', 'Add', inputs=['0_out_0', 'ini_0'], outputs=['2_out_0'], attrs={})
    node_3 = OnnxNode('Node_3', 'Add', inputs=['2_out_0', '1_out_0'], outputs=['3_out_0'], attrs={})
    return OnnxGraph(name='graph_1', nodes=[node_0, node_1, node_2, node_3], inputs=[input_0], outputs=[output_0, output_1], initializers=[ini_0])

class TestGraphBasic(unittest.TestCase):
    
    def setUp(self):
        self.graph = create_graph()
        self.graph_1 = create_graph_1()

    def test_graph_init(self):
        input_0 = OnnxPlaceHolder('input_0', np.dtype('float32'), [3,2])
        output_0 = OnnxPlaceHolder('output_0', np.dtype('float32'), [3,4])
        ini_0 = OnnxInitializer('ini_0', np.array([1,4], dtype='int32'))
        ini_1 = OnnxInitializer('ini_1', np.array([1], dtype='int32'))
        ini_2 = OnnxInitializer('const_0', np.array([1], dtype='int32'))
        node_0 = OnnxNode(
                    'Node_0', 
                    'Pad', 
                    inputs=['input_0', 'ini_0', 'ini_1', 'const_0'], 
                    outputs=['output_0'], 
                    attrs={'mode':b'constant'}, 
                    domain=''
        )


        self.assertTrue(is_list_equal(self.graph._nodes, [node_0]))
        self.assertTrue(is_list_equal(self.graph._inputs, [input_0]))
        self.assertTrue(is_list_equal(self.graph._outputs, [output_0]))
        self.assertTrue(is_list_equal(self.graph._initializers, [ini_0, ini_1, ini_2]))
        self.assertTrue(is_map_equal(self.graph._node_map, {
                                                    'input_0':input_0, 
                                                    'output_0':output_0, 
                                                    'ini_0':ini_0, 
                                                    'ini_1':ini_1,
                                                    'const_0': ini_2,
                                                    'Node_0':node_0
        }))
        self.assertTrue(is_map_equal(self.graph._prev_map, {'output_0':node_0}))
        self.assertTrue(is_map_equal(self.graph._next_map, {
                                                    'input_0':[node_0], 
                                                    'ini_0':[node_0], 
                                                    'ini_1':[node_0], 
                                                    'const_0':[node_0]
        }))


    def test_parse_proto(self):
        input_0 = helper.make_tensor_value_info('input_0', NP_TYPE_TO_TENSOR_TYPE[np.dtype('float32')], [3,2])
        ini_0 = helper.make_tensor('ini_0', NP_TYPE_TO_TENSOR_TYPE[np.dtype('int32')], [2], np.array([1,4]))
        ini_1 = helper.make_tensor('ini_1', NP_TYPE_TO_TENSOR_TYPE[np.dtype('int32')], [1], np.array([1]))
        node_0 = helper.make_node('Pad', ['input_0', 'ini_0', 'ini_1', 'const_0'], 
                                ['output_0'], 'Node_0', mode='constant')
        node_1 = helper.make_node('Constant', [], ['const_0'], 'Constant_0', 
                                value=numpy_helper.from_array(np.array([1], dtype='int32')))
        output_0 = helper.make_tensor_value_info('output_0', NP_TYPE_TO_TENSOR_TYPE[np.dtype('float32')], [3,4])
        value_info_0 = helper.make_tensor_value_info('const_0', NP_TYPE_TO_TENSOR_TYPE[np.dtype('int32')], [1])
        graph_proto = helper.make_graph(
                                        [node_0, node_1], 
                                        'test_parse', 
                                        [input_0], 
                                        [output_0], 
                                        [ini_0, ini_1], 
                                        value_info=[value_info_0]
        )
        model_proto = helper.make_model(graph_proto, producer_name='test_parse')
        
        parse_from_graph_proto = OnnxGraph.parse(graph_proto)
        parse_from_model_proto = OnnxGraph.parse(model_proto)
        self.assertTrue(is_graph_equal(parse_from_graph_proto, self.graph))
        self.assertTrue(is_graph_equal(parse_from_model_proto, self.graph))
    
    def test_to_proto(self):
        self.assertIsInstance(self.graph.proto(), GraphProto)
    
    def test_to_model(self):
        self.assertIsInstance(self.graph.model(), ModelProto)

    def test_save_after_add_node(self):
        self.graph.add_input('test_input', 'float32', [1, 2, 3])
        self.graph.add_output('test_output', 'float32', [1, 2, 3])
        self.graph.add_initializer('test_initializer', np.array([1, 2, 3]))
        self.graph.add_node('test_node', 'Add')
        self.graph.save('test.onnx')
        os.remove('test.onnx')

    def test_toposort(self):
        random.shuffle(self.graph_1._nodes)
        self.graph_1.toposort()
        sorted_order = [n.name for n in self.graph_1._nodes]
        possible_orders = [
            ['Node_0', 'Node_1', 'Node_2', 'Node_3'],
            ['Node_1', 'Node_0', 'Node_2', 'Node_3']
        ]
        self.assertIn(sorted_order, possible_orders)
    
    def test_toposort_with_cycle(self):
        self.graph_1['Node_2'].inputs[1] = '3_out_0'
        self.graph_1._next_map['3_out_0'] = self.graph_1['Node_2']
        with self.assertRaisesRegex(RuntimeError, "Cycle detected in graph!"):
            self.graph_1.toposort()

    def test_opset_imports(self):
        # specify opset_imports
        self.graph.opset_imports = 13
        opset = OperatorSetIdProto()
        opset.version = 13
        self.assertEqual(self.graph.opset_imports, [opset])
        # clear opset_imports
        self.graph.opset_imports = None
        self.assertEqual(self.graph.opset_imports, None)
        # exist two domain version fields
        opset_0 = OperatorSetIdProto()
        opset_0.domain = ""
        opset_0.version = 13
        opset_1 = OperatorSetIdProto()
        opset_1.domain = "ai.onnx.ml"
        opset_1.version = 2
        graph = OnnxGraph(name='test_opset_imports', opset_imports = [opset_0, opset_1])
        self.assertEqual(graph.opset_imports, [opset_0])

if __name__ == '__main__':
    unittest.main()
