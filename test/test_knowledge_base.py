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

from typing import Dict, List
import operator as op
import numpy as np
import onnx

from onnx import (
    helper,
    TensorProto,
)

from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.interface.base_node import BaseNode
from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.pattern import MatchBase
from auto_optimizer.pattern.pattern import MATCH_PATTERN
from auto_optimizer.pattern.pattern import Pattern
from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase


class Conv1dMatch(MatchBase):
    def __init__(self):
        super().__init__()

    def match(self, node: BaseNode, graph: BaseGraph) -> bool:
        if node is None:
            return False
        if not op.eq(node.op_type, 'Conv'):
            return False
        if len(node.inputs) > 1:
            weight = graph[node.inputs[1]]
            return len(weight.value.shape) == 3
        return False


# element_wise允许出现0次，或者多次
pattern = Pattern() \
    .add_node('Conv', ['Conv'], [Conv1dMatch]) \
    .add_node('element_wise', ['Mul', 'Add', 'Sub', 'Div', 'BatchNormalization', 'LeakyRelu', 'Relu']) \
    .add_edge('Conv', 'element_wise') \
    .set_input('Conv') \
    .set_output('element_wise') \
    .set_node_loop('element_wise', MATCH_PATTERN.MATCH_ZERO_OR_MORE) \
    .set_loop(MATCH_PATTERN.MATCH_ONCE_OR_MORE)


class Knowledge_Example(KnowledgeBase):
    def __init__(self):
        super().__init__()
        self._register_apply_funcs(pattern, [self._example_apply])

    def _example_apply(self, graph, result):
        return True


class TestKnowledgeBase(unittest.TestCase):
    def make_twice_conv1d_model(self, onnx_name, x):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, None)

        weight_0 = helper.make_tensor('weight_0', TensorProto.FLOAT, (64, 3, 1), \
            np.random.randn(64, 3, 1).astype(np.float32))
        weight_1 = helper.make_tensor('weight_1', TensorProto.FLOAT, (128, 64, 1), \
            np.random.randn(128, 64, 1).astype(np.float32))

        conv_0 = helper.make_node("Conv", ['X', 'weight_0'], ['out_0'], 'Conv_0', None, None, \
            dilations = np.array([1], dtype=np.int64), \
            group = 1, \
            kernel_shape = np.array([1], dtype=np.int64), \
            pads = np.array([0, 0], dtype=np.int64), \
            strides = np.array([1], dtype=np.int64))

        conv_1 = helper.make_node("Conv", ['out_0', 'weight_1'], ['Z'], 'Conv_1', None, None, \
            dilations = np.array([1], dtype=np.int64), \
            group = 1, \
            kernel_shape = np.array([1], dtype=np.int64), \
            pads = np.array([0, 0], dtype=np.int64), \
            strides = np.array([1], dtype=np.int64))

        graph = helper.make_graph([conv_0, conv_1], "conv1d_test", [X], [Z], [weight_0, weight_1])
        model = helper.make_model(graph)

        del model.opset_import[:]
        opset = model.opset_import.add()
        opset.domain = ''
        opset.version = 14
        onnx.save(model, onnx_name)

    def make_multi_conv1d_and_split_graph_model(self, onnx_name, x):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, None)

        weight_0 = helper.make_tensor('weight_0', TensorProto.FLOAT, (64, 3, 1), \
            np.random.randn(64, 3, 1).astype(np.float32))
        weight_1 = helper.make_tensor('weight_1', TensorProto.FLOAT, (128, 64, 1), \
            np.random.randn(128, 64, 1).astype(np.float32))
        weight_2 = helper.make_tensor('weight_2', TensorProto.FLOAT, (128, 64, 1), \
            np.random.randn(128, 64, 1).astype(np.float32))

        conv_0 = helper.make_node("Conv", ['X', 'weight_0'], ['conv_0_out'], 'Conv_0', None, None, \
            dilations = np.array([1], dtype=np.int64), \
            group = 1, \
            kernel_shape = np.array([1], dtype=np.int64), \
            pads = np.array([0, 0], dtype=np.int64), \
            strides = np.array([1], dtype=np.int64))
        relu_0 = helper.make_node('Relu', ['conv_0_out'], ['relu_0_out'], 'Relu_0', None, None)
        relu_1 = helper.make_node('Relu', ['relu_0_out'], ['relu_1_out'], 'Relu_1', None, None)

        conv_1 = helper.make_node("Conv", ['relu_1_out', 'weight_1'], ['conv_1_out'], 'Conv_1', None, None, \
            dilations = np.array([1], dtype=np.int64), \
            group = 1, \
            kernel_shape = np.array([1], dtype=np.int64), \
            pads = np.array([0, 0], dtype=np.int64), \
            strides = np.array([1], dtype=np.int64))
        relu_2 = helper.make_node('Relu', ['conv_1_out'], ['relu_2_out'], 'Relu_2', None, None)

        conv_2 = helper.make_node("Conv", ['relu_1_out', 'weight_2'], ['conv_2_out'], 'Conv_2', None, None, \
            dilations = np.array([1], dtype=np.int64), \
            group = 1, \
            kernel_shape = np.array([1], dtype=np.int64), \
            pads = np.array([0, 0], dtype=np.int64), \
            strides = np.array([1], dtype=np.int64))
        relu_3 = helper.make_node('Relu', ['conv_2_out'], ['relu_3_out'], 'Relu_3', None, None)

        add = helper.make_node('Add', ['relu_2_out', 'relu_3_out'], ['Z'], 'Add_0', None, None)

        graph = helper.make_graph([conv_0, relu_0, relu_1, conv_1, relu_2, relu_3, conv_2, add], "conv1d_test", \
            [X], [Z], [weight_0, weight_1, weight_2])
        model = helper.make_model(graph)

        del model.opset_import[:]
        opset = model.opset_import.add()
        opset.domain = ''
        opset.version = 14
        onnx.save(model, onnx_name)

    def make_multi_conv1d_and_split_graph_and_shape_model(self, onnx_name, x):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, None)

        weight_0 = helper.make_tensor('weight_0', TensorProto.FLOAT, (64, 3, 1), \
            np.random.randn(64, 3, 1).astype(np.float32))
        weight_1 = helper.make_tensor('weight_1', TensorProto.FLOAT, (128, 64, 1), \
            np.random.randn(128, 64, 1).astype(np.float32))
        weight_2 = helper.make_tensor('weight_2', TensorProto.FLOAT, (128, 64, 1), \
            np.random.randn(128, 64, 1).astype(np.float32))

        conv_0 = helper.make_node("Conv", ['X', 'weight_0'], ['conv_0_out'], 'Conv_0', None, None, \
            dilations = np.array([1], dtype=np.int64), \
            group = 1, \
            kernel_shape = np.array([1], dtype=np.int64), \
            pads = np.array([0, 0], dtype=np.int64), \
            strides = np.array([1], dtype=np.int64))
        relu_0 = helper.make_node('Relu', ['conv_0_out'], ['relu_0_out'], 'Relu_0', None, None)
        relu_1 = helper.make_node('Relu', ['relu_0_out'], ['relu_1_out'], 'Relu_1', None, None)

        shape = helper.make_node('Shape', ['relu_1_out'], ['shape_out'], 'Shape_0', None, None)

        conv_1 = helper.make_node("Conv", ['shape_out', 'weight_1'], ['conv_1_out'], 'Conv_1', None, None, \
            dilations = np.array([1], dtype=np.int64), \
            group = 1, \
            kernel_shape = np.array([1], dtype=np.int64), \
            pads = np.array([0, 0], dtype=np.int64), \
            strides = np.array([1], dtype=np.int64))
        relu_2 = helper.make_node('Relu', ['conv_1_out'], ['relu_2_out'], 'Relu_2', None, None)

        conv_2 = helper.make_node("Conv", ['shape_out', 'weight_2'], ['conv_2_out'], 'Conv_2', None, None, \
            dilations = np.array([1], dtype=np.int64), \
            group = 1, \
            kernel_shape = np.array([1], dtype=np.int64), \
            pads = np.array([0, 0], dtype=np.int64), \
            strides = np.array([1], dtype=np.int64))
        relu_3 = helper.make_node('Relu', ['conv_2_out'], ['relu_3_out'], 'Relu_4', None, None)

        add = helper.make_node('Add', ['relu_2_out', 'relu_3_out'], ['Z'], 'Add_0', None, None)

        graph = helper.make_graph([conv_0, relu_0, relu_1, conv_1, relu_2, relu_3, conv_2, add, shape], \
            "conv1d_test", [X], [Z], [weight_0, weight_1, weight_2])
        model = helper.make_model(graph)

        del model.opset_import[:]
        opset = model.opset_import.add()
        opset.domain = ''
        opset.version = 14
        onnx.save(model, onnx_name)

    def test_knowledge_iterator_func(self):
        knowledge_example = Knowledge_Example()

        self.assertTrue(knowledge_example.has_next_pattern())
        knowledge_example.next_pattern()
        
        self.assertTrue(knowledge_example.has_next_apply())
        knowledge_example.next_apply()

        self.assertFalse(knowledge_example.has_next_apply())
        self.assertFalse(knowledge_example.has_next_pattern())

    def test_get_candidate_sub_graph_func_0(self):
        x = np.random.randn(1, 3, 2500).astype(np.float32)
        onnx_path = './twice_conv1d.onnx'

        self.make_twice_conv1d_model(onnx_path, x)

        graph = OnnxGraph.parse(onnx_path)
        knowledge_example = Knowledge_Example()

        result = knowledge_example.get_candidate_sub_graphs(graph)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].node_dicts), 2)
        self.assertEqual(len(result[0].node_dicts[0]), 1)
        self.assertEqual(len(result[0].node_dicts[1]), 1)
        self.assertEqual(result[0].node_dicts[0].get('Conv')[0].name, 'Conv_0')
        self.assertEqual(result[0].node_dicts[1].get('Conv')[0].name, 'Conv_1')

    def test_get_candidate_sub_graph_func_1(self):
        x = np.random.randn(1, 3, 2500).astype(np.float32)
        onnx_path = './multi_conv1d_and_split_graph.onnx'

        self.make_multi_conv1d_and_split_graph_model(onnx_path, x)

        graph = OnnxGraph.parse(onnx_path)
        knowledge_example = Knowledge_Example()

        result = knowledge_example.get_candidate_sub_graphs(graph)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].node_dicts), 3)

        for node_dict in result[0].node_dicts:
            self.assertEqual(len(node_dict), 2)
            self.assertEqual(len(node_dict.get('Conv')), 1)
            self.assertEqual(len(node_dict.get('element_wise')), 2)

    def test_get_candidate_sub_graph_func_2(self):
        x = np.random.randn(1, 3, 2500).astype(np.float32)
        onnx_path = './multi_conv1d_and_split_graph_and_shape.onnx'

        self.make_multi_conv1d_and_split_graph_and_shape_model(onnx_path, x)

        graph = OnnxGraph.parse(onnx_path)
        knowledge_example = Knowledge_Example()

        results = knowledge_example.get_candidate_sub_graphs(graph)
        self.assertEqual(len(results), 3)

        for result in results:
            for node_dict in result.node_dicts:
                self.assertEqual(len(node_dict), 2)
                self.assertEqual(len(node_dict.get('Conv')), 1)
                self.assertEqual(len(node_dict.get('element_wise')), 2)

    def test_apply_func(self):
        x = np.random.randn(1, 3, 2500).astype(np.float32)
        onnx_path = './twice_conv1d.onnx'

        self.make_twice_conv1d_model(onnx_path, x)

        graph = OnnxGraph.parse(onnx_path)
        knowledge_example = Knowledge_Example()

        match_result = knowledge_example.get_candidate_sub_graphs(graph)
        self.assertEqual(len(match_result), 1)

        res = knowledge_example.apply(graph, match_result[0])
        self.assertTrue(res)

    def test_get_apply_ids(self):
        knowledge_example = Knowledge_Example()
        apply_ids = knowledge_example.get_apply_ids()

        self.assertTrue(len(apply_ids) == 1)
        self.assertEqual(apply_ids[0], 0)

    def test_set_apply_id(self):
        knowledge_example = Knowledge_Example()

        res = knowledge_example.set_apply_id(-1)
        self.assertFalse(res)

        res = knowledge_example.set_apply_id(1)
        self.assertFalse(res)

        res = knowledge_example.set_apply_id(0)
        self.assertTrue(res)


if __name__ == "__main__":
    unittest.main()
