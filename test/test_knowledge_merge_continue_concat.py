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
import os

import numpy as np
import onnx
from onnx import (
    helper,
    TensorProto,
)

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges.knowledge_merge_continue_concat import KnowledgeMergeContinueConcat
from utils import inference, optimize


def make_c2_concat_model(onnx_name, x, y, z, diff_axis=False):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, x.shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, y.shape)
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, z.shape)

    O = helper.make_tensor_value_info("O", TensorProto.FLOAT, None)

    axis = 1 if diff_axis else 0
    node_concat0 = helper.make_node("Concat", ["X", "Y"], ["X_S"], "Concat0", axis=0)
    node_concat1 = helper.make_node("Concat", ["X_S", "Z"], ["O"], "Concat1", axis=axis)

    graph = helper.make_graph([node_concat0, node_concat1], "continue_concat_test", [X, Y, Z], [O])
    model = helper.make_model(graph)

    del model.opset_import[:]
    opset = model.opset_import.add()
    opset.domain = ''
    opset.version = 14
    onnx.save(model, onnx_name)


class TestKnowledgeMergeContinueConcat(unittest.TestCase):

    def test_merge_c2_concat(self):
        x = np.random.randn(2, 1, 2).astype(np.float32)
        y = np.random.randn(1, 1, 2).astype(np.float32)
        z = np.random.randn(3, 1, 2).astype(np.float32)

        c2_concat_onnx = "./c2_concat.onnx"
        c2_concat_optimize_onnx = "{}_optimize.onnx".format(os.path.splitext(c2_concat_onnx)[0])
        os.system("rm -rf {} {}".format(c2_concat_onnx, c2_concat_optimize_onnx))

        make_c2_concat_model(c2_concat_onnx, x, y, z, False)
        ret = inference(c2_concat_onnx, (x, y, z))

        graph = OnnxGraph.parse(c2_concat_onnx)
        knowledge = KnowledgeMergeContinueConcat()
        res = optimize(graph, knowledge)
        self.assertTrue(res)
        graph.save(c2_concat_optimize_onnx)

        ret1 = inference(c2_concat_optimize_onnx, (x, y, z))
        self.assertTrue(np.array_equal(ret[0], ret1[0]))

    def test_merge_c2_diff_axis_concat(self):
        x = np.random.randn(2, 1, 2).astype(np.float32)
        y = np.random.randn(1, 1, 2).astype(np.float32)
        z = np.random.randn(3, 1, 2).astype(np.float32)

        c2_concat_onnx = "./c2_concat_diff_axis.onnx"
        c2_concat_optimize_onnx = "{}_optimize.onnx".format(os.path.splitext(c2_concat_onnx)[0])
        os.system("rm -rf {} {}".format(c2_concat_onnx, c2_concat_optimize_onnx))

        make_c2_concat_model(c2_concat_onnx, x, y, z, True)
        _ = inference(c2_concat_onnx, (x, y, z))

        graph = OnnxGraph.parse(c2_concat_onnx)
        knowledge = KnowledgeMergeContinueConcat()
        res = optimize(graph, knowledge)
        self.assertFalse(res)
        graph.save(c2_concat_optimize_onnx)


if __name__ == "__main__":
    unittest.main()
