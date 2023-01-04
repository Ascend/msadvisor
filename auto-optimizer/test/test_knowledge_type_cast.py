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
from auto_optimizer.pattern.knowledge_factory import KnowledgeType
from auto_optimizer.pattern.knowledges.knowledge_type_cast import KnowledgeTypeCast
from helper import KnowledgeTestHelper, OptimizationConfig


def make_type_cast_model(onnx_name, x: np.ndarray, y: np.ndarray, value_type: np.dtype):
    graph = OnnxGraph(name=onnx_name)
    graph.add_input('X', value_type, x.shape)
    graph.add_input('Y', value_type, y.shape)
    graph.add_output('O_0', value_type, None)
    graph.add_output('O_1', value_type, None)

    concat_value_0 = np.random.randn(*x.shape).astype(value_type)
    concat_value_1 = np.random.randn(*x.shape).astype(value_type)
    mul_value_0 = np.random.randn(*x.shape).astype(value_type)
    graph.add_initializer('Concat_value_0', concat_value_0)
    graph.add_initializer('Concat_value_1', concat_value_1)
    graph.add_initializer('Mul_value_0', mul_value_0)

    graph.add_node('Add0', 'Add', ['X', 'Y'], ['Add_O'])
    graph.add_node('Squeeze0', 'Squeeze', ['Add_O'], ['O_0'])
    graph.add_node('Mul0', 'Mul', ['Mul_value_0', 'Add_O'], ['Mul_O'])
    graph.add_node('Concat0', 'Concat', ['Concat_value_0', 'Concat_value_1', 'Add_O'], ['O_1'], attrs={'axis': 0})
    graph.update_map()

    graph.infershape()
    return graph


class TestKnowledgeTypeCast(unittest.TestCase, KnowledgeTestHelper):
    def test_basic_type_cast(self):
        for value_type in [np.int64, np.float64]:
            x = np.random.randn(10, 10).astype(value_type)
            y = np.random.randn(10, 10).astype(value_type)

            onnx_name = 'type_cast_test'
            onnx_ori = f'onnx/{onnx_name}.onnx'
            onnx_opt = f'onnx/{onnx_name}_optimize.onnx'
            graph = make_type_cast_model(onnx_name, x, y, value_type)
            cfg = OptimizationConfig(
                graph=graph,
                knowledge=KnowledgeTypeCast(),
                onnx_ori=onnx_ori,
                onnx_opt=onnx_opt,
            )
            self.assertTrue(self.check_optimization(cfg=cfg, expect=True))
            feeds = [
                {
                    'X': np.random.randn(*x.shape).astype(x.dtype),
                    'Y': np.random.randn(*y.shape).astype(y.dtype),
                }
                for _ in range(10)
            ]
            self.assertTrue(self.check_precision(onnx_ori, onnx_opt, feeds))


if __name__ == '__main__':
    unittest.main()
