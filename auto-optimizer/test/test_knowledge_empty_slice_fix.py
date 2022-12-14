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

from itertools import product
import unittest

import numpy as np
import onnx

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges import KnowledgeEmptySliceFix
from utils import inference, optimize


def make_multi_concat_empty_slice_model(onnx_name, x: np.ndarray):
    graph = OnnxGraph(name=onnx_name)
    graph.add_input('input', np.int32, x.shape)
    graph.add_output('output', np.int32, [x.shape[0] * 2, *x.shape[1:]])

    for idx in [0, 1, 2]:
        for param in ['starts', 'ends', 'axes']:
            value = np.array([0], dtype=np.int64)
            if idx != 0 and param == 'ends':
                value = np.array([x.shape[0]], dtype=np.int64)
            graph.add_initializer(
                name=f'Slice_{idx}_{param}',
                value=value
            )
        graph.add_node(
            name=f'Slice_{idx}',
            op_type='Slice',
            inputs=[
                'input',
                f'Slice_{idx}_starts',
                f'Slice_{idx}_ends',
                f'Slice_{idx}_axes'
            ],
            outputs=[f'Slice_{idx}_output']
        )
    graph.add_node(
        name='Concat_0',
        op_type='Concat',
        inputs=['Slice_0_output', 'Slice_1_output', 'Slice_2_output'],
        outputs=['output'],
        attrs={'axis': 0}
    )

    graph.update_map()
    graph.infershape()
    return graph


def make_empty_slice_model(onnx_name, x: np.ndarray, add_slice=True, add_cast=True):
    graph = OnnxGraph(name=onnx_name)
    graph.add_input('input', np.int32, x.shape)
    graph.add_output('output', np.int64 if add_cast else np.int32, x.shape)

    for idx in [0, 1]:
        if not add_slice and idx == 1:
            continue
        for param in ['starts', 'ends', 'axes']:
            value = np.array([0], dtype=np.int64)
            if idx == 1 and param == 'ends':
                value = np.array([x.shape[0]], dtype=np.int64)
            graph.add_initializer(
                name=f'Slice_{idx}_{param}',
                value=value
            )
        graph.add_node(
            name=f'Slice_{idx}',
            op_type='Slice',
            inputs=[
                'input',
                f'Slice_{idx}_starts',
                f'Slice_{idx}_ends',
                f'Slice_{idx}_axes'
            ],
            outputs=[f'Slice_{idx}_output']
        )
    graph.add_node(
        name='Concat_0',
        op_type='Concat',
        inputs=['Slice_0_output', 'Slice_1_output' if add_slice else 'input'],
        outputs=['Concat_0_output' if add_cast else 'output'],
        attrs={'axis': 0}
    )
    if add_cast:
        graph.add_node(
            name='Cast_0',
            op_type='Cast',
            inputs=['Concat_0_output'],
            outputs=['output'],
            attrs={'to': onnx.TensorProto.INT64}
        )

    graph.update_map()
    graph.infershape()
    return graph


def make_two_outputs_empty_slice_model(onnx_name, x: np.ndarray, add_slice=True):
    graph = OnnxGraph(name=onnx_name)
    graph.add_input('input', np.int32, x.shape)
    graph.add_output('out0', np.int32, x.shape)
    graph.add_output('out1', np.int64, x.shape)

    for idx in [0, 1]:
        if not add_slice and idx == 1:
            continue
        for param in ['starts', 'ends', 'axes']:
            value = np.array([0], dtype=np.int64)
            if idx == 1 and param == 'ends':
                value = np.array([x.shape[0]], dtype=np.int64)
            graph.add_initializer(
                name=f'Slice_{idx}_{param}',
                value=value
            )
        graph.add_node(
            name=f'Slice_{idx}',
            op_type='Slice',
            inputs=[
                'input',
                f'Slice_{idx}_starts',
                f'Slice_{idx}_ends',
                f'Slice_{idx}_axes'
            ],
            outputs=[f'Slice_{idx}_output']
        )
    graph.add_node(
        name='Concat_0',
        op_type='Concat',
        inputs=['Slice_0_output', 'Slice_1_output' if add_slice else 'input'],
        outputs=['out0'],
        attrs={'axis': 0}
    )
    graph.add_node(
        name='Cast_0',
        op_type='Cast',
        inputs=['out0'],
        outputs=['out1'],
        attrs={'to': onnx.TensorProto.INT64}
    )

    graph.update_map()
    graph.infershape()
    return graph


class TestKnowledgeEmptySliceFix(unittest.TestCase):
    def test_two_outputs_empty_slice_fix(self):
        input_ = np.random.randn(10).astype(np.int32)

        for add_slice in [False, True]:
            with self.subTest(add_slice=add_slice):
                postfix = '_slice' if add_slice else ''
                onnx_name = f'empty_slice_fix_combined{postfix}'
                origin_file = f'onnx/{onnx_name}.onnx'
                optimized_file = f'onnx/{onnx_name}_optimize.onnx'
                graph = make_two_outputs_empty_slice_model(onnx_name, input_, add_slice)
                graph.save(origin_file)

                knowledge = KnowledgeEmptySliceFix()
                result = optimize(graph, knowledge)
                graph.save(optimized_file)
                self.assertTrue(result)

                output_origin = inference(origin_file, [input_])
                output_optimized = inference(optimized_file, [input_])
                self.assertTrue(len(output_origin) == len(output_optimized))
                for lmat, rmat in zip(output_origin, output_optimized):
                    self.assertTrue(np.allclose(lmat, rmat, atol=1e-4, rtol=1e-2))

                result = optimize(graph, knowledge)
                self.assertFalse(result)

    def test_multi_concat_empty_slice_fix(self):
        input_ = np.random.randn(10).astype(np.int32)

        onnx_name = 'empty_slice_fix_multi_concat'
        origin_file = f'onnx/{onnx_name}.onnx'
        optimized_file = f'onnx/{onnx_name}_optimize.onnx'
        graph = make_multi_concat_empty_slice_model(onnx_name, input_)
        graph.save(origin_file)

        knowledge = KnowledgeEmptySliceFix()
        result = optimize(graph, knowledge)
        graph.save(optimized_file)
        self.assertTrue(result)

        output_origin = inference(origin_file, [input_])
        output_optimized = inference(optimized_file, [input_])
        self.assertTrue(len(output_origin) == len(output_optimized))
        for lmat, rmat in zip(output_origin, output_optimized):
            self.assertTrue(np.allclose(lmat, rmat, atol=1e-4, rtol=1e-2))

        result = optimize(graph, knowledge)
        self.assertFalse(result)

    def test_basic_empty_slice_fix(self):
        input_ = np.random.randn(10).astype(np.int32)

        for add_slice, add_cast in product([False, True], repeat=2):
            with self.subTest(add_slice=add_slice, add_cast=add_cast):
                postfix0 = '_slice' if add_slice else ''
                postfix1 = '_cast' if add_cast else ''
                onnx_name = f'empty_slice_fix_single{postfix0}{postfix1}'
                origin_file = f'onnx/{onnx_name}.onnx'
                optimized_file = f'onnx/{onnx_name}_optimize.onnx'
                graph = make_empty_slice_model(onnx_name, input_, add_slice, add_cast)
                graph.save(origin_file)

                knowledge = KnowledgeEmptySliceFix()
                result = optimize(graph, knowledge)
                graph.save(optimized_file)
                self.assertTrue(result)

                output_origin = inference(origin_file, [input_])
                output_optimized = inference(optimized_file, [input_])
                self.assertTrue(len(output_origin) == len(output_optimized))
                for lmat, rmat in zip(output_origin, output_optimized):
                    self.assertTrue(np.allclose(lmat, rmat, atol=1e-4, rtol=1e-2))

                result = optimize(graph, knowledge)
                self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()