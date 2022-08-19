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

import random
import unittest
from typing import List

import numpy as np
from onnx import helper, GraphProto, ModelProto
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from auto_optimizer.graph_refactor.onnx.node import PlaceHolder, Initializer, Node
from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from test_node_common import is_ph_equal, is_ini_equal, is_node_equal


def is_elem_equal(elem1, elem2):
    if isinstance(elem1, PlaceHolder):
        return is_ph_equal(elem1, elem2)
    elif isinstance(elem1, Initializer):
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

def is_graph_equal(g1, g2):
    return is_list_equal(g1.nodes, g2.nodes) and \
        is_list_equal(g1.initializers, g2.initializers) and \
        is_list_equal(g1.inputs, g2.inputs) and \
        is_list_equal(g1.outputs, g2.outputs) and \
        is_list_equal(g1._value_infos, g2._value_infos) and \
        is_map_equal(g1._node_map, g2._node_map) and \
        is_map_equal(g1._prev_map, g2._prev_map) and \
        is_map_equal(g1._next_map, g2._next_map)

def create_graph():
    input_0 = PlaceHolder('input_0', np.dtype('float32'), [3,2])
    output_0 = PlaceHolder('output_0', np.dtype('float32'), [3,4])
    ini_0 = Initializer('ini_0', np.array([1,4], dtype='int32'))
    ini_1 = Initializer('ini_1', np.array([1], dtype='int32'))
    node_0 = Node(
                'Node_0', 
                'Pad', 
                inputs=['input_0', 'ini_0', 'ini_1'], 
                outputs=['output_0'], 
                attrs={'mode':b'constant'}, 
                domain=''
    )
    graph = OnnxGraph([node_0], [input_0], [output_0], [ini_0, ini_1], name='test_graph')
    return graph


class TestGraphBasic(unittest.TestCase):
    
    def test_graph_init(self):
        input_0 = PlaceHolder('input_0', np.dtype('float32'), [3,2])
        output_0 = PlaceHolder('output_0', np.dtype('float32'), [3,4])
        ini_0 = Initializer('ini_0', np.array([1,4], dtype='int32'))
        ini_1 = Initializer('ini_1', np.array([1], dtype='int32'))
        node_0 = Node(
                    'Node_0', 
                    'Pad', 
                    inputs=['input_0', 'ini_0', 'ini_1'], 
                    outputs=['output_0'], 
                    attrs={'mode':b'constant'}, 
                    domain=''
        )

        graph = create_graph()
        self.assertTrue(is_list_equal(graph._nodes, [node_0]))
        self.assertTrue(is_list_equal(graph._inputs, [input_0]))
        self.assertTrue(is_list_equal(graph._outputs, [output_0]))
        self.assertTrue(is_list_equal(graph._initializers, [ini_0, ini_1]))
        self.assertTrue(is_map_equal(graph._node_map, 
                                    {'input_0':input_0, 'output_0':output_0, 'ini_0':ini_0, 'ini_1':ini_1, 'Node_0':node_0}))
        self.assertTrue(is_map_equal(graph._prev_map, {'output_0':node_0}))
        self.assertTrue(is_map_equal(graph._next_map, {'input_0':[node_0], 'ini_0':[node_0], 'ini_1':[node_0]}))

    def test_parse_proto(self):
        input_0 = helper.make_tensor_value_info('input_0', NP_TYPE_TO_TENSOR_TYPE[np.dtype('float32')], [3,2])
        ini_0 = helper.make_tensor('ini_0', NP_TYPE_TO_TENSOR_TYPE[np.dtype('int32')], [2], np.array([1,4]))
        ini_1 = helper.make_tensor('ini_1', NP_TYPE_TO_TENSOR_TYPE[np.dtype('int32')], [1], np.array([1]))
        node_0 = helper.make_node('Pad', ['input_0', 'ini_0', 'ini_1'], ['output_0'], 'Node_0', mode='constant')
        output_0 = helper.make_tensor_value_info('output_0', NP_TYPE_TO_TENSOR_TYPE[np.dtype('float32')], [3,4])
        graph_proto = helper.make_graph([node_0], 'test_parse', [input_0], [output_0], [ini_0, ini_1])
        model_proto = helper.make_model(graph_proto, producer_name='test_parse')
        
        expected_graph = create_graph()
        parse_from_graph_proto = OnnxGraph.parse(graph_proto)
        parse_from_model_proto = OnnxGraph.parse(model_proto)
        self.assertTrue(is_graph_equal(parse_from_graph_proto, expected_graph))
        self.assertTrue(is_graph_equal(parse_from_model_proto, expected_graph))
    
    def test_to_proto(self):
        graph = create_graph()
        self.assertIsInstance(graph.proto(), GraphProto)
    
    def test_to_model(self):
        graph = create_graph()
        self.assertIsInstance(graph.model(), ModelProto)

    def test_toposort(self):
        graph = create_graph()
        expected_order = [n.name for n in graph._nodes]
        random.shuffle(graph._nodes)
        graph.toposort()
        test_order = [n.name for n in graph._nodes]
        self.assertEqual(test_order, expected_order)

if __name__ == '__main__':
    unittest.main()