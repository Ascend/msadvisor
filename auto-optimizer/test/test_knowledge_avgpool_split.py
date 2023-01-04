# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges.knowledge_avgpool_split import KnowledgeAvgPoolSplit

from helper import KnowledgeTestHelper, OptimizationConfig


def make_dynamic_model(onnx_name, x, attrs):
    graph = OnnxGraph(name=onnx_name)
    graph.add_input('X', x['dtype'], x['shape'])
    graph.add_output('OUT_0', x['dtype'], None)

    graph.add_node('AveragePool', 'AveragePool', ['X'], ['OUT_0'], attrs)

    graph.update_map()
    return graph

class TestKnowledgeAvgPoolSplit(unittest.TestCase, KnowledgeTestHelper):
    def test_basic_avgpool_split(self):
        usecases = [
            # ceil_mode, kernel_shape, pads, strides, kernel_shape split result
            (0, [32, 64], [0, 0, 0, 0], [32, 64], [[8, 16], [4, 4]]),
            (0, [16, 32], [0, 0, 0, 0], [16, 32], [[8, 16], [2, 2]]),
            (0, [22, 31], [0, 0, 0, 0], [22, 31], [[2, 31], [11, 1]]),
            (0, [16, 32], [0, 0, 0, 0], [8, 16], []),
            (0, [17, 32], [0, 0, 0, 0], [17, 32], [[17, 8], [1, 4]]),
            (0, [17, 31], [0, 0, 0, 0], [17, 31], [[1, 31], [17, 1]]),
            (0, [4, 8], [0, 0, 0, 0], [4, 8], [])
        ]

        for ceil_mode, kernel_shape, pads, strides, split_result in usecases:
            x = {'shape': (1, 8, 32, 64), 'dtype': np.float32}
            attrs = {
                'ceil_mode': ceil_mode,
                'kernel_shape': kernel_shape,
                'pads': pads,
                'strides': strides,
            }

            onnx_name = 'knowledge_avgpool_split_test'
            onnx_ori = f'onnx/{onnx_name}.onnx'
            onnx_opt = f'onnx/{onnx_name}_optimize.onnx'
            graph = make_dynamic_model(onnx_name, x, attrs)
            expect = len(split_result) != 0
            cfg = OptimizationConfig(
                graph=graph,
                knowledge=KnowledgeAvgPoolSplit(),
                onnx_ori=onnx_ori,
                onnx_opt=onnx_opt,
            )
            self.assertTrue(self.check_optimization(cfg=cfg, expect=expect))
            if not expect:
                continue

            graph_opt = OnnxGraph.parse(onnx_opt)
            nodes = graph_opt.get_nodes('AveragePool')
            for i, node in enumerate(nodes):
                self.assertTrue(np.all(node.attrs['kernel_shape'] == split_result[i]))
                self.assertTrue(np.all(node.attrs['strides'] == split_result[i]))


if __name__ == "__main__":
    unittest.main()
