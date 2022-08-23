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

import operator as op
import numpy as np
import onnx

from onnx import (
    helper,
    TensorProto,
)

from magiconnx import OnnxGraph
from magiconnx.interface import BaseGraph as GraphBase
from magiconnx.interface import BaseNode as NodeBase
from auto_optimizer.pattern.pattern import MatchBase
from auto_optimizer.pattern.pattern import MATCH_PATTERN
from auto_optimizer.pattern.pattern import Pattern
from auto_optimizer.pattern.matcher import Matcher


class Conv1dMatch(MatchBase):
    def __init__(self):
        super().__init__()

    def match(self, node: NodeBase, graph: GraphBase) -> bool:
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
    .set_loop(MATCH_PATTERN.MATCH_ONECE_OR_MORE)


class TestMatcher(unittest.TestCase):

    def make_single_conv1d_model(self, onnx_name, x):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, None)

        weight_0 = helper.make_tensor('weight_0', TensorProto.FLOAT, (64, 3, 1), \
            np.random.randn(64, 3, 1).astype(np.float32))

        conv_0 = helper.make_node("Conv", ['X', 'weight_0'], ['Z'], 'Conv_0', None, None, \
            dilations = np.array([1], dtype=np.int64), \
            group = 1, \
            kernel_shape = np.array([1], dtype=np.int64), \
            pads = np.array([0, 0], dtype=np.int64), \
            strides = np.array([1], dtype=np.int64))

        graph = helper.make_graph([conv_0], "conv1d_test", [X], [Z], [weight_0])
        model = helper.make_model(graph)

        del model.opset_import[:]
        opset = model.opset_import.add()
        opset.domain = ''
        opset.version = 14
        onnx.save(model, onnx_name)

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

    def make_single_conv1d_and_relu_model(self, onnx_name, x):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, None)

        weight_0 = helper.make_tensor('weight_0', TensorProto.FLOAT, (64, 3, 1), \
            np.random.randn(64, 3, 1).astype(np.float32))

        conv_0 = helper.make_node("Conv", ['X', 'weight_0'], ['out_0'], 'Conv_0', None, None, \
            dilations = np.array([1], dtype=np.int64), \
            group = 1, \
            kernel_shape = np.array([1], dtype=np.int64), \
            pads = np.array([0, 0], dtype=np.int64), \
            strides = np.array([1], dtype=np.int64))
        relu_0 = helper.make_node('Relu', ['out_0'], ['Z'], 'Relu_0', None, None)

        graph = helper.make_graph([conv_0, relu_0], "conv1d_test", [X], [Z], [weight_0])
        model = helper.make_model(graph)

        del model.opset_import[:]
        opset = model.opset_import.add()
        opset.domain = ''
        opset.version = 14
        onnx.save(model, onnx_name)

    def make_single_conv1d_and_twice_relu_model(self, onnx_name, x):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, None)

        weight_0 = helper.make_tensor('weight_0', TensorProto.FLOAT, (64, 3, 1), \
            np.random.randn(64, 3, 1).astype(np.float32))

        conv_0 = helper.make_node("Conv", ['X', 'weight_0'], ['out_0'], 'Conv_0', None, None, \
            dilations = np.array([1], dtype=np.int64), \
            group = 1, \
            kernel_shape = np.array([1], dtype=np.int64), \
            pads = np.array([0, 0], dtype=np.int64), \
            strides = np.array([1], dtype=np.int64))
        relu_0 = helper.make_node('Relu', ['out_0'], ['out_1'], 'Relu_0', None, None)
        relu_1 = helper.make_node('Relu', ['out_1'], ['Z'], 'Relu_1', None, None)

        graph = helper.make_graph([conv_0, relu_0, relu_1], "conv1d_test", [X], [Z], [weight_0])
        model = helper.make_model(graph)

        del model.opset_import[:]
        opset = model.opset_import.add()
        opset.domain = ''
        opset.version = 14
        onnx.save(model, onnx_name)

    def make_single_conv1d_and_relu_and_shape_model(self, onnx_name, x):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, None)

        weight_0 = helper.make_tensor('weight_0', TensorProto.FLOAT, (64, 3, 1), \
            np.random.randn(64, 3, 1).astype(np.float32))

        conv_0 = helper.make_node("Conv", ['X', 'weight_0'], ['out_0'], 'Conv_0', None, None, \
            dilations = np.array([1], dtype=np.int64), \
            group = 1, \
            kernel_shape = np.array([1], dtype=np.int64), \
            pads = np.array([0, 0], dtype=np.int64), \
            strides = np.array([1], dtype=np.int64))
        relu_0 = helper.make_node('Relu', ['out_0'], ['out_1'], 'Relu_0', None, None)
        shape = helper.make_node('Shape', ['out_1'], ['out_2'], 'Shape_0', None, None)
        relu_1 = helper.make_node('Relu', ['out_2'], ['Z'], 'Relu_1', None, None)

        graph = helper.make_graph([conv_0, relu_0, shape, relu_1], "conv1d_test", [X], [Z], [weight_0])
        model = helper.make_model(graph)

        del model.opset_import[:]
        opset = model.opset_import.add()
        opset.domain = ''
        opset.version = 14
        onnx.save(model, onnx_name)

    def make_single_conv1d_and_split_relu_model(self, onnx_name, x):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, None)

        weight_0 = helper.make_tensor('weight_0', TensorProto.FLOAT, (64, 3, 1), \
            np.random.randn(64, 3, 1).astype(np.float32))

        conv_0 = helper.make_node("Conv", ['X', 'weight_0'], ['out_0'], 'Conv_0', None, None, \
            dilations = np.array([1], dtype=np.int64), \
            group = 1, \
            kernel_shape = np.array([1], dtype=np.int64), \
            pads = np.array([0, 0], dtype=np.int64), \
            strides = np.array([1], dtype=np.int64))
        relu_0 = helper.make_node('Relu', ['out_0'], ['out_1'], 'Relu_0', None, None)
        relu_1 = helper.make_node('Relu', ['out_1'], ['Z'], 'Relu_1', None, None)
        relu_2 = helper.make_node('Relu', ['out_1'], ['Z'], 'Relu_2', None, None)

        graph = helper.make_graph([conv_0, relu_0, relu_1, relu_2], "conv1d_test", [X], [Z], [weight_0])
        model = helper.make_model(graph)

        del model.opset_import[:]
        opset = model.opset_import.add()
        opset.domain = ''
        opset.version = 14
        onnx.save(model, onnx_name)

    def test_get_candidate_nodes_func_0(self):
        x = np.random.randn(1, 3, 2500).astype(np.float32)
        onnx_path = './single_conv1d.onnx'

        self.make_single_conv1d_model(onnx_path, x)

        graph = OnnxGraph(onnx_path)
        matcher = Matcher(graph, pattern)
        nodes = matcher.get_candidate_nodes()

        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].name, 'Conv_0')

    def test_get_candidate_nodes_func_1(self):
        x = np.random.randn(1, 3, 2500).astype(np.float32)
        onnx_path = './twice_conv1d.onnx'

        self.make_twice_conv1d_model(onnx_path, x)

        graph = OnnxGraph(onnx_path)
        matcher = Matcher(graph, pattern)
        nodes = matcher.get_candidate_nodes()

        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].name, 'Conv_0')
        self.assertEqual(nodes[1].name, 'Conv_1')

    def test_get_candidate_nodes_func_2(self):
        x = np.random.randn(1, 3, 2500).astype(np.float32)
        onnx_path = './single_conv1d_and_relu.onnx'

        self.make_single_conv1d_and_relu_model(onnx_path, x)

        graph = OnnxGraph(onnx_path)
        matcher = Matcher(graph, pattern)
        nodes = matcher.get_candidate_nodes()

        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].name, 'Conv_0')

    def test_get_match_map_func_0(self):
        x = np.random.randn(1, 3, 2500).astype(np.float32)
        onnx_path = './single_conv1d_and_relu.onnx'

        self.make_single_conv1d_and_relu_model(onnx_path, x)

        graph = OnnxGraph(onnx_path)
        matcher = Matcher(graph, pattern)

        nodes = matcher.get_candidate_nodes()
        self.assertEqual(len(nodes), 1)

        result = matcher.get_match_map(nodes[0])
        self.assertEqual(len(result.node_dicts), 1)
        self.assertEqual(len(result.node_dicts[0]), 2)
        self.assertEqual(len(result.node_dicts[0].get('Conv')), 1)
        self.assertEqual(len(result.node_dicts[0].get('element_wise')), 1)
        self.assertEqual(result.node_dicts[0]['Conv'][0].name, 'Conv_0')
        self.assertEqual(result.node_dicts[0]['element_wise'][0].name, 'Relu_0')

    def test_get_match_map_func_1(self):
        x = np.random.randn(1, 3, 2500).astype(np.float32)
        onnx_path = './single_conv1d_and_twice_relu.onnx'

        self.make_single_conv1d_and_twice_relu_model(onnx_path, x)

        graph = OnnxGraph(onnx_path)
        matcher = Matcher(graph, pattern)

        nodes = matcher.get_candidate_nodes()
        self.assertEqual(len(nodes), 1)

        result = matcher.get_match_map(nodes[0])
        self.assertEqual(len(result.node_dicts), 1)
        self.assertEqual(len(result.node_dicts[0]), 2)
        self.assertEqual(len(result.node_dicts[0].get('Conv')), 1)
        self.assertEqual(len(result.node_dicts[0].get('element_wise')), 2)
        self.assertEqual(result.node_dicts[0]['Conv'][0].name, 'Conv_0')
        self.assertEqual(result.node_dicts[0]['element_wise'][0].name, 'Relu_0')
        self.assertEqual(result.node_dicts[0]['element_wise'][1].name, 'Relu_1')

    def test_get_match_map_func_2(self):
        x = np.random.randn(1, 3, 2500).astype(np.float32)
        onnx_path = './single_conv1d_and_relu_and_shape.onnx'

        self.make_single_conv1d_and_relu_and_shape_model(onnx_path, x)

        graph = OnnxGraph(onnx_path)
        matcher = Matcher(graph, pattern)

        nodes = matcher.get_candidate_nodes()
        self.assertEqual(len(nodes), 1)

        result = matcher.get_match_map(nodes[0])
        self.assertEqual(len(result.node_dicts), 1)
        self.assertEqual(len(result.node_dicts[0]), 2)
        self.assertEqual(len(result.node_dicts[0].get('Conv')), 1)
        self.assertEqual(len(result.node_dicts[0].get('element_wise')), 1)
        self.assertEqual(result.node_dicts[0]['Conv'][0].name, 'Conv_0')
        self.assertEqual(result.node_dicts[0]['element_wise'][0].name, 'Relu_0')

    def test_get_match_map_func_3(self):
        x = np.random.randn(1, 3, 2500).astype(np.float32)
        onnx_path = './single_conv1d_and_twice_relu.onnx'

        self.make_single_conv1d_and_split_relu_model(onnx_path, x)

        graph = OnnxGraph(onnx_path)
        matcher = Matcher(graph, pattern)

        nodes = matcher.get_candidate_nodes()
        self.assertEqual(len(nodes), 1)

        result = matcher.get_match_map(nodes[0])
        self.assertEqual(len(result.node_dicts), 1)
        self.assertEqual(len(result.node_dicts[0]), 2)
        self.assertEqual(len(result.node_dicts[0].get('Conv')), 1)
        self.assertEqual(len(result.node_dicts[0].get('element_wise')), 3)
        self.assertEqual(result.node_dicts[0]['Conv'][0].name, 'Conv_0')
        self.assertEqual(result.node_dicts[0]['element_wise'][0].name, 'Relu_0')

if __name__ == "__main__":
    unittest.main()
