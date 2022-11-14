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
from auto_optimizer.pattern.knowledges.knowledge_merge_consecutive_concat import KnowledgeMergeConsecutiveConcat
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


class TestKnowledgeMergeConsecutiveConcat(unittest.TestCase):

    def test_merge_c2_concat(self):
        x = np.random.rand(2, 1, 2).astype(np.float32) + 0.5
        y = np.random.rand(1, 1, 2).astype(np.float32) + 0.5
        z = np.random.rand(3, 1, 2).astype(np.float32) + 0.5

        onnx_name = "c2_concat"
        c2_concat_onnx = f"./onnx/{onnx_name}.onnx"
        c2_concat_optimize_onnx = f"./onnx/{onnx_name}_optimize.onnx"

        make_c2_concat_model(c2_concat_onnx, x, y, z, False)

        graph = OnnxGraph.parse(c2_concat_onnx)
        knowledge = KnowledgeMergeConsecutiveConcat()
        res = optimize(graph, knowledge)
        self.assertTrue(res)
        graph.save(c2_concat_optimize_onnx)

        matrix_before_apply = inference(c2_concat_onnx, [x, y, z])
        matrix_after_apply = inference(c2_concat_optimize_onnx, [x, y, z])
        self.assertTrue(len(matrix_before_apply) == len(matrix_after_apply))
        for lmatrix, rmatrix in zip(matrix_before_apply, matrix_after_apply):
            self.assertTrue(np.allclose(lmatrix, rmatrix, atol=1e-4, rtol=1e-2))

        result = optimize(graph, knowledge)
        self.assertFalse(result)

    def test_merge_c2_diff_axis_concat(self):
        x = np.random.rand(2, 1, 2).astype(np.float32) + 0.5
        y = np.random.rand(1, 1, 2).astype(np.float32) + 0.5
        z = np.random.rand(3, 1, 2).astype(np.float32) + 0.5

        onnx_name = "c2_concat_diff_axis"
        c2_concat_onnx = f"./onnx/{onnx_name}.onnx"

        make_c2_concat_model(c2_concat_onnx, x, y, z, True)

        graph = OnnxGraph.parse(c2_concat_onnx)
        knowledge = KnowledgeMergeConsecutiveConcat()
        res = optimize(graph, knowledge)
        self.assertFalse(res)


if __name__ == "__main__":
    unittest.main()
