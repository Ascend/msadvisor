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
from auto_optimizer.pattern.utils import insert_squeeze
from helper import KnowledgeTestHelper, OptimizationConfig


def make_dynamic_model(onnx_name, x, y, shape):
    graph = OnnxGraph(name=onnx_name)
    graph.add_input('X', x['dtype'], x['shape'])
    graph.add_input('Y', y['dtype'], y['shape'])
    graph.add_output('OUT_0', x['dtype'], None)

    graph.add_initializer('Shape_0', shape)
    graph.add_node('Reshape0', 'Reshape', ['Y', 'Shape_0'], ['Reshape0_out'])
    graph.add_node('Shape', 'Shape', ['Reshape0_out'], ['Shape_out'])
    graph.add_node('Add', 'Add', ['X', 'Y'], ['Add_out'])
    graph.add_node('Reshape1', 'Reshape', ['Add_out', 'Shape_out'], ['OUT_0'])
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
    graph.add_node('Add', 'Add', ['X', 'Y'], ['Add_out'])
    graph.add_node('Reshape1', 'Reshape', ['Add_out', 'Shape_out'], ['OUT_0'])
    attrs = {'axes': [1]}
    insert_squeeze(graph, graph['Reshape0'], attrs, mode = 'after', refer_index = 0)
    graph.update_map()
    return graph


class TestKnowledgeDynamicReshape(unittest.TestCase, KnowledgeTestHelper):
    def test_basic_dynamic_reshape(self):
        usecases = [
            (('bs', 512), np.float32, np.array([-1, 64], dtype = np.int64)),
            (('bs', 'len', 512), np.float32, np.array([-1, 0, 64], dtype = np.int64))
        ]

        for in_shape, in_dtype, shape in usecases:
            x = {'shape': in_shape, 'dtype': in_dtype}
            y = {'shape': in_shape, 'dtype': in_dtype}

            onnx_name = 'knowledge_dynamic_reshape_test'
            onnx_ori = f'onnx/{onnx_name}.onnx'
            onnx_opt = f'onnx/{onnx_name}_optimize.onnx'
            graph = make_dynamic_model(onnx_name, x, y, shape)
            cfg = OptimizationConfig(
                graph=graph,
                knowledge=KnowledgeDynamicReshape(),
                onnx_ori=onnx_ori,
                onnx_opt=onnx_opt,
            )
            self.assertTrue(self.check_optimization(cfg=cfg, expect=True))

            var = {'bs': 8, 'len': 128}
            shape_ = [var[x] if x in var else x for x in in_shape]
            feeds = [
                {
                    'X': np.random.randn(*shape_).astype(x['dtype']),
                    'Y': np.random.randn(*shape_).astype(y['dtype']),
                } for _ in range(10)
            ]
            self.assertTrue(self.check_precision(onnx_ori, onnx_opt, feeds))

            graph_opt = OnnxGraph.parse(onnx_opt)
            shape_name = graph_opt['Reshape1'].inputs[1]
            self.assertTrue(np.all(graph_opt[shape_name].value == list(shape)))

    def test_basic_dynamic_reshape_and_squeeze(self):
        usecases = [
            (('bs', 8, 'len', 32), np.float32, np.array([-1, 1, 0, 32], dtype = np.int64))
        ]

        for in_shape, in_dtype, shape in usecases:
            x = {'shape': in_shape, 'dtype': in_dtype}
            y = {'shape': in_shape, 'dtype': in_dtype}

            onnx_name = 'knowledge_dynamic_reshape_and_squeeze_test'
            onnx_ori = f'onnx/{onnx_name}.onnx'
            onnx_opt = f'onnx/{onnx_name}_optimize.onnx'
            graph = make_dynamic_model_and_squeeze(onnx_name, x, y, shape)
            cfg = OptimizationConfig(
                graph=graph,
                knowledge=KnowledgeDynamicReshape(),
                onnx_ori=onnx_ori,
                onnx_opt=onnx_opt,
            )
            self.assertTrue(self.check_optimization(cfg=cfg, expect=True))

            var = {'bs': 8, 'len': 128}
            shape_ = [var[x] if x in var else x for x in in_shape]
            feeds = [
                {
                    'X': np.random.randn(*shape_).astype(x['dtype']),
                    'Y': np.random.randn(*shape_).astype(y['dtype']),
                } for _ in range(10)
            ]
            self.assertTrue(self.check_precision(onnx_ori, onnx_opt, feeds))

            graph_opt = OnnxGraph.parse(onnx_opt)
            shape_name = graph_opt['Reshape1'].inputs[1]
            self.assertTrue(np.all(graph_opt[shape_name].value == list(shape)))


if __name__ == '__main__':
    unittest.main()
