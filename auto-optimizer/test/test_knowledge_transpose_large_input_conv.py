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
from helper import KnowledgeTestHelper, OptimizationConfig


class TestKnowledgeTransposeLargeInputConv(unittest.TestCase, KnowledgeTestHelper):

    def test_aasist(self):
        models = [
            (True, './onnx/aasist_bs1_ori.onnx', (1, 64600), 10),
        ]
        for expect, onnx_ori, shape, count in models:
            with self.subTest(onnx_ori):
                onnx_opt = f'{os.path.splitext(onnx_ori)[0]}_transpose_conv.onnx'
                graph = OnnxGraph.parse(onnx_ori)
                cfg = OptimizationConfig(
                    graph=graph,
                    knowledge=KnowledgeTransposeLargeInputConv(),
                    onnx_ori=onnx_ori,
                    onnx_opt=onnx_opt,
                )
                self.assertTrue(self.check_optimization(cfg=cfg, expect=expect))
                if not expect:
                    continue
                feeds = [
                    {
                        'input': np.random.randn(*shape).astype(np.float32),
                    }
                    for _ in range(count)
                ]
                self.assertTrue(self.check_precision(onnx_ori, onnx_opt, feeds))


if __name__ == "__main__":
    unittest.main()
