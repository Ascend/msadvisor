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
import time
import random

import numpy as np
import onnx
from onnx import (
    helper,
    TensorProto,
)

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges.knowledge_transpose_huge_conv import KnowledgeTransposeHugeConv
from utils import inference, optimize


class TestKnowledgeTransposeHugeConv(unittest.TestCase):

    def test_aasist(self):
        models = [
            (True, './onnx/aasist_bs1_ori.onnx', (1, 64600), 1),
        ]
        for expect, path, shape, count in models:
            with self.subTest(path):
                optimized_path = f'{os.path.splitext(path)[0]}_transpose_conv.onnx'
                graph = OnnxGraph.parse(path)
                knowledge = KnowledgeTransposeHugeConv()
                res = optimize(graph, knowledge)
                self.assertEqual(res, expect)
                graph.save(optimized_path)
                for _ in range(count):
                    x = np.random.randn(*shape).astype(np.float32)
                    ret0 = inference(path, x)
                    ret1 = inference(optimized_path, x)
                    for r0, r1 in zip(ret0, ret1):
                        diff = np.sum(np.abs(r0 - r1))
                        base = np.sum(np.abs(r0))
                        acc = diff / base
                        self.assertTrue(acc < 0.000001)


if __name__ == "__main__":
    unittest.main()
