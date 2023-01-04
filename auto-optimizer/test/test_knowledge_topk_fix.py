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
import onnx

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges import KnowledgeTopkFix
from helper import KnowledgeTestHelper, OptimizationConfig


def make_topk_model(onnx_name, x: np.ndarray):
    graph = OnnxGraph(name=onnx_name)
    graph.add_input('input', np.float32, x.shape)
    graph.add_output('output_v', np.float32, (10, ))
    graph.add_output('output_i_0', np.int32, (10, ))
    graph.add_output('output_i_1', np.int64, (10, ))
    graph.add_output('topk_i', np.int64, (10, ))

    graph.add_initializer(
        name='topk_k',
        value=np.array([10], np.int64)
    )
    graph.add_node(
        name='topk_0',
        op_type='TopK',
        inputs=['input', 'topk_k'],
        outputs=['topk_v', 'topk_i'],
        attrs={'axis': 0, 'largest': 1, 'sorted': 1}
    )

    graph.add_node(
        name='relu_0',
        op_type='Relu',
        inputs=['topk_v'],
        outputs=['output_v'],
    )

    graph.add_node(
        name='cast_0',
        op_type='Cast',
        inputs=['topk_i'],
        outputs=['output_i_0'],
        attrs={'to': onnx.TensorProto.INT32}
    )

    graph.add_initializer(
        name='add_init',
        value=np.array(list(range(10)), dtype=np.int64)
    )

    graph.add_node(
        name='add_0',
        op_type='Add',
        inputs=['topk_i', 'add_init'],
        outputs=['output_i_1'],
    )

    graph.update_map()
    graph.infershape()
    return graph


class TestKnowledgeTopkFix(unittest.TestCase, KnowledgeTestHelper):
    def test_basic_topk_fix(self):
        input_ = np.random.rand(100).astype(np.float32)

        onnx_name = 'topk_model'
        onnx_ori = f'onnx/{onnx_name}.onnx'
        onnx_opt = f'onnx/{onnx_name}_fixed.onnx'
        graph = make_topk_model(onnx_name, input_)
        cfg = OptimizationConfig(
            graph=graph,
            knowledge=KnowledgeTopkFix(),
            onnx_ori=onnx_ori,
            onnx_opt=onnx_opt,
        )
        self.assertTrue(self.check_optimization(cfg=cfg, expect=True))


if __name__ == '__main__':
    unittest.main()
