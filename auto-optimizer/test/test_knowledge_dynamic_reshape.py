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
from auto_optimizer.pattern.knowledges.knowledge_dynamic_reshape import KnowledgeDynamicReshape
from auto_optimizer.pattern.knowledges.utils import insert_squeeze

from utils import inference, optimize


def make_dynamic_model(onnx_name, x, y, shape):
    graph = OnnxGraph(name=onnx_name)
    graph.add_input('X', x['dtype'], x['shape'])
    graph.add_input('Y', y['dtype'], y['shape'])
    graph.add_output('OUT_0', x['dtype'], None)

    graph.add_initializer('Shape_0', shape)
    graph.add_node('Reshape0', 'Reshape', ['Y', 'Shape_0'], ['Reshape0_out'])
    graph.add_node('Shape', 'Shape', ['Reshape0_out'], ['Shape_out'])
    graph.add_node('Reshape1', 'Reshape', ['X', 'Shape_out'], ['OUT_0'])
    graph.update_map()
    return graph


def make_dynamic_model_and_squeeze(onnx_name, x, y, shape):
    graph = OnnxGraph(name=onnx_name)
    graph.add_input('X', x['dtype'], x['shape'])
    graph.add_input('Y', y['dtype'], y['shape'])
    graph.add_output('OUT_0', x['dtype'], None)

    graph.add_initializer('Shape_0', shape)
    graph.add_node('Reshape0', 'Reshape', ['Y', 'Shape_0'], ['Reshape0_out'])
    graph.add_node('Shape', 'Shape', ['Reshape0_out'], ['Shape_out'])
    graph.add_node('Reshape1', 'Reshape', ['X', 'Shape_out'], ['OUT_0'])
    attrs = {'axes': np.array([1], dtype = np.int64)}
    insert_squeeze(graph, graph['Reshape0'], attrs, mode = 'after', refer_index = 0)
    graph.update_map()
    return graph


class TestKnowledgeDynamicReshape(unittest.TestCase):
    def test_basic_dynamic_reshape(self):
        usecases = [
            (('bs', 512), np.float32, np.array([-1, 64], dtype = np.int64)),
            (('bs', 'len', 512), np.float32, np.array([-1, 0, 64], dtype = np.int64))
        ]

        for in_shape, in_dtype, shape in usecases:
            x = {'shape': in_shape, 'dtype': in_dtype}
            y = {'shape': in_shape, 'dtype': in_dtype}

            onnx_name = 'knowledge_dynamic_reshape_test'
            origin_file = f'onnx/{onnx_name}.onnx'
            optimized_file = f'onnx/{onnx_name}_optimize.onnx'
            graph = make_dynamic_model(onnx_name, x, y, shape)
            graph.save(origin_file)

            knowledge = KnowledgeDynamicReshape()
            result = optimize(graph, knowledge)
            graph.save(optimized_file)
            self.assertTrue(result)

            shape_name = graph['Reshape1'].inputs[1]
            self.assertTrue(np.all(graph[shape_name].value == list(shape)))

            result = optimize(graph, knowledge)
            self.assertFalse(result)

    def test_basic_dynamic_reshape_and_squeeze(self):
        usecases = [
            (('bs', 8, 'len', 32), np.float32, np.array([-1, 1, 0, 32], dtype = np.int64))
        ]

        for in_shape, in_dtype, shape in usecases:
            x = {'shape': in_shape, 'dtype': in_dtype}
            y = {'shape': in_shape, 'dtype': in_dtype}

            onnx_name = 'knowledge_dynamic_reshape_test'
            origin_file = f'onnx/{onnx_name}.onnx'
            optimized_file = f'onnx/{onnx_name}_optimize.onnx'
            graph = make_dynamic_model_and_squeeze(onnx_name, x, y, shape)
            graph.save(origin_file)

            knowledge = KnowledgeDynamicReshape()
            result = optimize(graph, knowledge)
            graph.save(optimized_file)
            self.assertTrue(result)

            shape_name = graph['Reshape1'].inputs[1]
            self.assertTrue(np.all(graph[shape_name].value == list(shape)))

            result = optimize(graph, knowledge)
            self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()


