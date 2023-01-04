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
import onnx
import onnxruntime as ort

from onnx import (
    helper,
    TensorProto,
)

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges.knowledge_conv1d2conv2d import KnowledgeConv1d2Conv2d
from helper import KnowledgeTestHelper, OptimizationConfig


class TestKnowledgeConv1d2Conv2d(unittest.TestCase, KnowledgeTestHelper):
    def make_twice_conv1d_model(self, onnx_name, x):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 128, 2500])

        weight_0 = helper.make_tensor('weight_0', TensorProto.FLOAT, (64, 3, 1),
                                      np.random.randn(64, 3, 1).astype(np.float32))
        weight_1 = helper.make_tensor('weight_1', TensorProto.FLOAT, (128, 64, 1),
                                      np.random.randn(128, 64, 1).astype(np.float32))

        conv_0 = helper.make_node("Conv", ['X', 'weight_0'], ['out_0'], 'Conv_0', None, None,
                                  dilations=[1],
                                  group=1,
                                  kernel_shape=[1],
                                  pads=[0, 0],
                                  strides=[1])

        conv_1 = helper.make_node("Conv", ['out_0', 'weight_1'], ['Z'], 'Conv_1', None, None,
                                  dilations=[1],
                                  group=1,
                                  kernel_shape=[1],
                                  pads=[0, 0],
                                  strides=[1])

        graph = helper.make_graph([conv_0, conv_1], "conv1d_test", [X], [Z], [weight_0, weight_1])
        model = helper.make_model(graph)

        del model.opset_import[:]
        opset = model.opset_import.add()
        opset.domain = ''
        opset.version = 14
        onnx.save(model, onnx_name)

    def make_multi_conv1d_and_split_graph_model(self, onnx_name, x):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 128, 2500])

        weight_0 = helper.make_tensor('weight_0', TensorProto.FLOAT, (64, 3, 1),
                                      np.random.randn(64, 3, 1).astype(np.float32))
        weight_1 = helper.make_tensor('weight_1', TensorProto.FLOAT, (128, 64, 1),
                                      np.random.randn(128, 64, 1).astype(np.float32))
        weight_2 = helper.make_tensor('weight_2', TensorProto.FLOAT, (128, 64, 1),
                                      np.random.randn(128, 64, 1).astype(np.float32))

        conv_0 = helper.make_node("Conv", ['X', 'weight_0'], ['conv_0_out'], 'Conv_0', None, None,
                                  dilations=[1],
                                  group=1,
                                  kernel_shape=[1],
                                  pads=[0, 0],
                                  strides=[1])
        relu_0 = helper.make_node('Relu', ['conv_0_out'], ['relu_0_out'], 'Relu_0', None, None)
        relu_1 = helper.make_node('Relu', ['relu_0_out'], ['relu_1_out'], 'Relu_1', None, None)

        conv_1 = helper.make_node("Conv", ['relu_1_out', 'weight_1'], ['conv_1_out'], 'Conv_1', None, None,
                                  dilations=[1],
                                  group=1,
                                  kernel_shape=[1],
                                  pads=[0, 0],
                                  strides=[1])
        relu_2 = helper.make_node('Relu', ['conv_1_out'], ['relu_2_out'], 'Relu_2', None, None)

        conv_2 = helper.make_node("Conv", ['relu_1_out', 'weight_2'], ['conv_2_out'], 'Conv_2', None, None,
                                  dilations=[1],
                                  group=1,
                                  kernel_shape=[1],
                                  pads=[0, 0],
                                  strides=[1])
        relu_3 = helper.make_node('Relu', ['conv_2_out'], ['relu_3_out'], 'Relu_3', None, None)

        add = helper.make_node('Add', ['relu_2_out', 'relu_3_out'], ['Z'], 'Add_0', None, None)

        graph = helper.make_graph([conv_0, relu_0, relu_1, conv_1, relu_2, relu_3, conv_2, add], "conv1d_test",
                                  [X], [Z], [weight_0, weight_1, weight_2])
        model = helper.make_model(graph)

        del model.opset_import[:]
        opset = model.opset_import.add()
        opset.domain = ''
        opset.version = 14
        onnx.save(model, onnx_name)

    def test_conv1d2conv2d_optimizer_0(self):
        x = np.random.randn(1, 3, 2500).astype(np.float32)
        onnx_ori = './onnx/twice_conv1d.onnx'
        onnx_opt = f'{onnx_ori}_new.onnx'
        self.make_twice_conv1d_model(onnx_ori, x)

        graph = OnnxGraph.parse(onnx_ori)
        cfg = OptimizationConfig(
            graph=graph,
            knowledge=KnowledgeConv1d2Conv2d(),
            onnx_ori=onnx_ori,
            onnx_opt=onnx_opt,
        )
        self.assertTrue(self.check_optimization(cfg=cfg, expect=True))
        feeds = [{'X': np.random.randn(*x.shape).astype(x.dtype)} for _ in range(10)]
        self.assertTrue(self.check_precision(onnx_ori, onnx_opt, feeds))

    def test_conv1d2conv2d_optimizer_1(self):
        x = np.random.randn(1, 3, 2500).astype(np.float32)
        onnx_ori = './onnx/multi_conv1d_and_split_graph.onnx'
        onnx_opt = f'{onnx_ori}_new.onnx'
        self.make_multi_conv1d_and_split_graph_model(onnx_ori, x)

        graph = OnnxGraph.parse(onnx_ori)
        cfg = OptimizationConfig(
            graph=graph,
            knowledge=KnowledgeConv1d2Conv2d(),
            onnx_ori=onnx_ori,
            onnx_opt=onnx_opt,
        )
        self.assertTrue(self.check_optimization(cfg=cfg, expect=True))
        feeds = [{'X': np.random.randn(*x.shape).astype(x.dtype)} for _ in range(10)]
        self.assertTrue(self.check_precision(onnx_ori, onnx_opt, feeds))


if __name__ == "__main__":
    unittest.main()
