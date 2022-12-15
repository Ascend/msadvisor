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

from utils import inference, optimize


def make_dynamic_model(onnx_name, x, attrs):
    graph = OnnxGraph(name=onnx_name)
    graph.add_input('X', x['dtype'], x['shape'])
    graph.add_output('OUT_0', x['dtype'], None)

    graph.add_node('AveragePool', 'AveragePool', ['X'], ['OUT_0'], attrs)

    graph.update_map()
    return graph

class TestKnowledgeDynamicReshape(unittest.TestCase):
    def test_basic_dynamic_reshape(self):
        usecases = [
            # ceil_mode, kernel_shape, pads, strides, kernel_shape split result
            (0, [32, 64], [0, 0, 0, 0], [32, 64], [[8, 16], [4, 4]]),
            (0, [16, 32], [0, 0, 0, 0], [16, 32], [[8, 16], [2, 2]]),
            (0, [16, 32], [0, 0, 0, 0], [8, 16], []),
            (0, [17, 32], [0, 0, 0, 0], [17, 32], [[17, 8], [1, 4]]),
            (0, [17, 31], [0, 0, 0, 0], [17, 31], []),
            (0, [4, 8], [0, 0, 0, 0], [4, 8], [])
        ]

        for ceil_mode, kernel_shape, pads, strides, split_result in usecases:
            x = {'shape': (1, 8, 32, 64), 'dtype': np.float32}
            attrs = {
                'ceil_mode': ceil_mode,
                'kernel_shape': np.array(kernel_shape, dtype = np.int64),
                'pads': np.array(pads, dtype = np.int64),
                'strides': np.array(strides, dtype = np.int64)
            }

            onnx_name = 'knowledge_avgpool_split_test'
            origin_file = f'onnx/{onnx_name}.onnx'
            optimized_file = f'onnx/{onnx_name}_optimize.onnx'
            graph = make_dynamic_model(onnx_name, x, attrs)
            graph.save(origin_file)

            knowledge = KnowledgeAvgPoolSplit()
            result = optimize(graph, knowledge)
            if len(split_result) == 0:
                self.assertFalse(result)
                continue
            self.assertTrue(result)
            graph.save(optimized_file)

            nodes = graph.get_nodes('AveragePool')
            for i, node in enumerate(nodes):
                self.assertTrue(np.all(node.attrs['kernel_shape'] == split_result[i]))
                self.assertTrue(np.all(node.attrs['strides'] == split_result[i]))

            input0 = np.random.randn(1, 8, 32, 64).astype(np.float32)
            out0 = inference(origin_file, [input0])
            out1 = inference(optimized_file, [input0])
            self.assertTrue(len(out0) == len(out1))
            for lmatrix, rmatrix in zip(out0, out1):
                self.assertTrue(np.allclose(lmatrix, rmatrix, atol=1e-4, rtol=1e-2))

            result = optimize(graph, knowledge)
            self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
