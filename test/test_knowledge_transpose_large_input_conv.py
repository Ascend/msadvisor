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

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges.knowledge_transpose_large_input_conv import KnowledgeTransposeLargeInputConv
from utils import inference, optimize


class TestKnowledgeTransposeLargeInputConv(unittest.TestCase):

    def test_aasist(self):
        models = [
            (True, './onnx/aasist_bs1_ori.onnx', (1, 64600), 1),
        ]
        for expect, path, shape, count in models:
            with self.subTest(path):
                optimized_path = f'{os.path.splitext(path)[0]}_transpose_conv.onnx'
                graph = OnnxGraph.parse(path)
                knowledge = KnowledgeTransposeLargeInputConv()
                result = optimize(graph, knowledge)
                self.assertEqual(result, expect)
                graph.save(optimized_path)
                for _ in range(count):
                    input_ = np.random.rand(*shape).astype(np.float32) + 0.5
                    matrix_before_apply = inference(path, [input_])
                    matrix_after_apply = inference(optimized_path, [input_])
                    self.assertTrue(len(matrix_before_apply) == len(matrix_after_apply))
                    for lmatrix, rmatrix in zip(matrix_before_apply, matrix_after_apply):
                        self.assertTrue(np.allclose(lmatrix, rmatrix, atol=1e-4, rtol=1e-2))

                result = optimize(graph, knowledge)
                self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
