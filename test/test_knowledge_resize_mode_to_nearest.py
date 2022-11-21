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
from auto_optimizer.pattern.knowledges.knowledge_resize_mode_to_nearest import KnowledgeResizeModeToNearest
from utils import inference, optimize


def make_resize_model(onnx_name, x: np.ndarray, y: np.ndarray, value_type: np.dtype):
    graph = OnnxGraph(name=onnx_name)
    graph.add_input('input', value_type, x.shape)
    graph.add_output('11', value_type, None)

    roi = np.random.randn(0).astype(np.int64)
    scales = np.random.randn(0).astype(np.int64)
    graph.add_initializer('roi', roi)
    graph.add_initializer('scales', scales)
    graph.add_node('Resize0', 'Resize', ['input', 'scales'], ['11'], attrs={
            'coordinate_transformation_mode': str("half_pixel"),
            'cubic_coeff_a': -0.75,
            'exclude_outside': 0,
            'mode': "linear",
            'nearest_mode': "round_prefer_floor",
        })
    graph.update_map()

    graph.infershape()
    return graph


class TestKnowledgeResizeModeToNearest(unittest.TestCase):
    def test_basic_resize_mode(self):
        for value_type in [np.int64]:
            X = np.random.randn(10, 10).astype(value_type)
            Y = np.random.randn(10, 10).astype(value_type)

            onnx_name = 'resize_mode_test'
            origin_file = f'onnx/{onnx_name}1.onnx'
            optimized_file = f'onnx/{onnx_name}_optimize1.onnx'
            graph = make_resize_model(onnx_name, X, Y, value_type)
            graph.save(origin_file)
            newgraph = OnnxGraph.parse(origin_file)
            knowledge = KnowledgeResizeModeToNearest()
            result = optimize(newgraph, knowledge)
            newgraph.save(optimized_file)
            self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()