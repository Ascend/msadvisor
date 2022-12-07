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

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges.knowledge_dynamic_reshape import KnowledgeDynamicReshape

from utils import inference, optimize

def make_dynamic_model(onnx_name, x, y):
    graph = OnnxGraph(name=onnx_name)
    graph.add_input('X', x.dtype, x.shape)
    graph.add_input('Y', y.dtype, y.shape)
    graph.add_output('OUT_0', x.dtype, None)

    graph.add_node('Reshape', 'Reshape', ['X', 'Y'], ['OUT_O'])
    graph.update_map()
    return graph


class TestKnowledgeDynamicReshape(unittest.TestCase):
    def test_basic_dynamic_reshape(self):
        X = {'shape': ('bs', 'len', 512), 'dtype': np.int32}
        Y = {'shape': (8*'bs', 'len', 64), 'dtype': np.int32}

        onnx_name = 'knowledge_dynamic_reshape_test'
        origin_file = f'onnx/{onnx_name}.onnx'
        optimized_file = f'onnx/{onnx_name}_optimize.onnx'
        graph = make_dynamic_model(onnx_name, X, Y)
        graph.save(origin_file)

        knowledge = KnowledgeDynamicReshape()
        result = optimize(graph, knowledge)
        graph.save(optimized_file)
        self.assertTrue(result)

        shape_name = graph['Reshape'].inputs[1]
        shape = graph[shape_name]
        self.assertTrue(np.all(shape == [-1, 0, 64]))

        result = optimize(graph, knowledge)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()


