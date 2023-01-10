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

import random
from typing import Dict, List, Tuple
import unittest
import numpy as np

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges import KnowledgeGatherToSplit
from helper import KnowledgeTestHelper, OptimizationConfig


def make_gather_to_split_graph(
    name: str,
    input_shape: Tuple[int, ...],
    gathers: List[Dict],
    version: int,
    extra_node: bool,
    extra_output: bool,
) -> OnnxGraph:
    graph = OnnxGraph(name=name, opset_imports=version)
    graph.add_input('input', np.float32, input_shape)
    output_shape = list(input_shape)
    output_shape[0] = sum(1 if isinstance((idx := g['indices']), int) else len(idx) for g in gathers)
    if extra_node:
        output_shape[0] += input_shape[0]
    graph.add_output('output', np.float32, output_shape)

    graph.add_node(
        name='relu_0',
        op_type='Relu',
        inputs=['input'],
        outputs=['relu_0_out'],
    )

    outputs = []
    for idx, gather in enumerate(gathers):
        axis, indices = gather['axis'], gather['indices']
        gather_name = f'gather_{idx}'
        ini_name = f'{gather_name}_indices'
        graph.add_initializer(
            name=ini_name,
            value=np.array(indices),
        )
        graph.add_node(
            name=gather_name,
            op_type='Gather',
            inputs=['relu_0_out', ini_name],
            outputs=[f'{gather_name}_out'],
            attrs={'axis': axis},
        )

        if extra_output and idx == 0:
            # extra_shape = list(input_shape)
            # extra_shape[axis] = 1 if isinstance(indices, int) else len(indices)
            graph.add_output(f'{gather_name}_out', np.float32, None)

        if isinstance(indices, int):
            if version < 13:
                graph.add_node(
                    name=f'unsqueeze_5_{idx}',
                    op_type='Unsqueeze',
                    inputs=[f'{gather_name}_out'],
                    outputs=[f'unsqueeze_5_{idx}_out'],
                    attrs={'axes': [axis]},
                )
            else:
                graph.add_initializer(
                    name=f'unsqueeze_5_{idx}_axes',
                    value=np.array([axis]),
                )
                graph.add_node(
                    name=f'unsqueeze_5_{idx}',
                    op_type='Unsqueeze',
                    inputs=[f'{gather_name}_out', f'unsqueeze_5_{idx}_axes'],
                    outputs=[f'unsqueeze_5_{idx}_out'],
                )

        out = f'unsqueeze_5_{idx}_out' if isinstance(indices, int) else f'{gather_name}_out'

        graph.add_node(
            name=f'relu_1_{idx}',
            op_type='Relu',
            inputs=[out],
            outputs=[f'relu_1_{idx}_out'],
        )
        if axis != 0:
            perm = list(range(len(input_shape)))
            perm[axis], perm[0] = perm[0], perm[axis]
            graph.add_node(
                name=f'transpose_2_{idx}',
                op_type='Transpose',
                inputs=[f'relu_1_{idx}_out'],
                outputs=[f'transpose_2_{idx}_out'],
                attrs={'perm': perm},
            )
            outputs.append(f'transpose_2_{idx}_out')
        else:
            outputs.append(f'relu_1_{idx}_out')

    if extra_node:
        graph.add_node(
            name='relu_extra',
            op_type='Relu',
            inputs=['relu_0_out'],
            outputs=['relu_extra_out'],
        )
        outputs.append('relu_extra_out')

    graph.add_node(
        name='concat_3',
        op_type='Concat',
        inputs=outputs,
        outputs=['output'],
        attrs={'axis': 0},
    )

    graph.update_map()
    graph.infershape()
    return graph


class TestKnowledgeGatherToSplit(unittest.TestCase, KnowledgeTestHelper):
    def test_basic_split(self):
        tests = [
            # normal case
            (10, True, (10, 10, 10), [
                {'axis': 0, 'indices': [i]} for i in range(10)
            ], False, False, 13),
            # lower version
            (10, True, (10, 10, 10), [
                {'axis': 0, 'indices': [i]} for i in range(10)
            ], False, False, 11),
            # add extra node should fail
            (10, False, (10, 10, 10), [
                {'axis': 0, 'indices': [i]} for i in range(10)
            ], True, False, 13),
            # add extra outputs should fail
            (10, False, (10, 10, 10), [
                {'axis': 0, 'indices': [i]} for i in range(10)
            ], False, True, 13),
            # add both extra node and output should fail
            (10, False, (10, 10, 10), [
                {'axis': 0, 'indices': [i]} for i in range(10)
            ], True, True, 13),
            # shuffle order should be ok
            (10, True, (10, 10, 10), [
                {'axis': 0, 'indices': [i]} for i in random.sample(list(range(10)), 10)
            ], False, False, 13),
            # 0-d indices in lower version should be ok
            (10, True, (3, 10, 10), [
                {'axis': 0, 'indices': i} for i in range(3)
            ], False, False, 11),
            # 0-d indices in higher version should be ok
            (10, True, (3, 10, 10), [
                {'axis': 0, 'indices': i} for i in range(3)
            ], False, False, 15),
            # mix and match 0-d and 1-d indices should also be ok
            (10, True, (6, 10, 10), [
                {'axis': 0, 'indices': slice_} for slice_ in [[1, 2], [3], 5]
            ], False, False, 11),
            # gather from different axes should fail
            (10, False, (10, 10, 10), [
                {'axis': 0 if i != 1 else 1, 'indices': [i]} for i in range(10)
            ], False, False, 13),
            # gather multiple indices should be ok
            (10, True, (3, 10, 10), [
                {'axis': 0, 'indices': slice_} for slice_ in [[0, 1], [2]]
            ], False, False, 15),
            # gather overlap should fail
            (10, False, (5, 10, 10), [
                {'axis': 0, 'indices': slice_} for slice_ in [[1, 2], [2, 3]]
            ], False, False, 15),
            # gather only part of input should be ok
            (10, True, (5, 10, 10), [
                {'axis': 0, 'indices': slice_} for slice_ in [[1, 2], [3]]
            ], False, False, 15),
            # non-continuous gather is not supported, should fail
            (10, False, (5, 10, 10), [
                {'axis': 0, 'indices': slice_} for slice_ in [[1, 3], [2]]
            ], False, False, 15),
            # only 1 gather should fail
            (10, False, (5, 10, 10), [
                {'axis': 0, 'indices': slice_} for slice_ in [[1, 2]]
            ], False, False, 15),
            # gather indices of rank r > 1 is not supported
            # however this is not tested because make_gather_to_split_graph
            # is not general enough
        ]
        for i, (count, expect, ishape, gathers, enode, eout, version) in enumerate(tests):
            ishape_s = 'x'.join(str(x) for x in ishape)
            axis_s = 1 if len(set(g['axis'] for g in gathers)) == 1 else 0
            indices_s = '_'.join(str(idx) if isinstance((idx := g['indices']), int) else 'x'.join(str(x) for x in idx) for g in gathers)
            name_ = f'test_gather_to_split_{i}_i{ishape_s}_a{axis_s}_idx{indices_s}_n{int(enode)}_o{int(eout)}_v{version}'
            with self.subTest(name=name_):
                onnx_ori = f'onnx/{name_}.onnx'
                graph = make_gather_to_split_graph(
                    name_,
                    input_shape=ishape,
                    gathers=gathers,
                    extra_node=enode,
                    extra_output=eout,
                    version=version
                )
                graph.save(onnx_ori)
                onnx_opt = f'onnx/{name_}_opt.onnx'
                cfg = OptimizationConfig(
                    graph=graph,
                    knowledge=KnowledgeGatherToSplit(),
                    onnx_ori=onnx_ori,
                    onnx_opt=onnx_opt,
                )
                self.assertTrue(self.check_optimization(cfg=cfg, expect=expect))
                if not expect:
                    continue

                feeds = [
                    {
                        'input': np.random.randn(*ishape).astype(np.float32)
                    }
                    for _ in range(count)
                ]
                self.assertTrue(
                    self.check_precision(
                        onnx_ori,
                        onnx_opt,
                        feeds,
                    )
                )


if __name__ == '__main__':
    unittest.main()
