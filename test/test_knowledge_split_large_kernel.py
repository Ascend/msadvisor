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

from typing import Tuple
import unittest
import numpy as np

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges.knowledge_split_large_kernel import KnowledgeSplitLargeKernel
from utils import inference, optimize


def make_graph(
    name,
    input_shape: Tuple[int, ...],
    kernel_shape: Tuple[int, ...],
    kweight_shape: Tuple[int, ...],
    kernel_pads: Tuple[int, ...],
    insert_relu_before_conv: bool = False,
    insert_relu_after_conv: bool = False,
) -> OnnxGraph:
    graph = OnnxGraph(name=name)
    graph.add_input('input', np.float32, input_shape)
    graph.add_output('output', np.float32, None)

    pre_ = 'input'
    next_ = 'relu_after_out' if insert_relu_after_conv else 'output'
    if insert_relu_before_conv:
        graph.add_node(
            name='relu_before',
            op_type='Relu',
            inputs=['input'],
            outputs=['relu_before_out'],
        )
        pre_ = 'relu_before_out'

    graph.add_initializer(
        name='weight',
        value=np.random.randn(*kweight_shape).astype(np.float32) + 0.5,
    )
    graph.add_initializer(
        name='bias',
        value=np.random.randn(kweight_shape[0]).astype(np.float32) + 0.5,
    )
    graph.add_node(
        name='conv_large',
        op_type='Conv',
        inputs=[pre_, 'weight', 'bias'],
        outputs=[next_],
        attrs={
            'kernel_shape': list(kernel_shape),
            'pads': list(kernel_pads),
            'group': 1,
        }
    )

    if insert_relu_after_conv:
        graph.add_node(
            name='relu_after',
            op_type='Relu',
            inputs=[next_],
            outputs=['output'],
        )

    graph.update_map()
    graph.infershape()
    return graph


class TestKnowledgeSplitLargeKernel(unittest.TestCase):
    def test_basic_split(self):
        tests = [
            # small kernel
            (1, False, (1, 3, 1024), (3, ), (128, 3), (1, 2), True, True, ),
            (1, False, (1, 3, 1344, 1344), (3, 3), (1, 3, 3, 3), (0, 1, 2, 3), True, True, ),
            (1, False, (1, 3, 133, 133, 133), (3, 3, 3), (1, 3, 3, 3, 3), (0, 1, 2, 3, 4, 5), True, True, ),
            # 1d
            (1, True, (1, 3, 1024), (257, ), (12, 3, 257), (0, 1), True, True, ),
            (1, True, (16, 3, 1024), (257, ), (12, 3, 257), (0, 1), True, True, ),
            (1, True, (1, 1, 1024), (257, ), (1, 1, 257), (0, 1), True, True, ),
            (1, True, (1, 1, 1024), (257, ), (1, 1, 257), (0, 1), False, True, ),
            (1, True, (1, 1, 1024), (257, ), (1, 1, 257), (0, 1), True, False, ),
            (1, True, (1, 1, 1024), (257, ), (1, 1, 257), (0, 1), False, False, ),
            # 2d
            (1, True, (1, 1, 512, 512), (257, 257), (1, 1, 257, 257), (0, 1, 2, 3), True, True, ),
            (1, True, (1, 3, 512, 512), (257, 257), (12, 3, 257, 257), (0, 1, 2, 3), True, True, ),
            (1, True, (2, 3, 512, 512), (257, 257), (12, 3, 257, 257), (0, 1, 2, 3), True, True, ),
            (1, True, (1, 1, 512, 512), (257, 257), (1, 1, 257, 257), (0, 1, 2, 3), False, True, ),
            (1, True, (1, 1, 512, 512), (257, 257), (1, 1, 257, 257), (0, 1, 2, 3), True, False, ),
            (1, True, (1, 1, 512, 512), (257, 257), (1, 1, 257, 257), (0, 1, 2, 3), False, False, ),
            (1, True, (1, 1, 1344, 1344), (257, 257), (1, 1, 257, 257), (0, 0, 0, 0), True, True, ),
            (1, True, (1, 1, 1344, 1344), (257, 257), (1, 1, 257, 257), (0, 1, 2, 3), True, True, ),
            (1, True, (1, 3, 1344, 1344), (257, 257), (2, 3, 257, 257), (0, 1, 2, 3), True, True, ),
            (1, True, (1, 1, 1344, 1344), (257, 3), (1, 1, 257, 3), (0, 1, 2, 3), True, True, ),
            (1, True, (1, 1, 1344, 1344), (3, 257), (1, 1, 3, 257), (0, 1, 2, 3), True, True, ),
            # 3d
            (1, True, (1, 1, 260, 260, 260), (257, 257, 257), (1, 1, 257, 257, 257), (0, 0, 0, 0, 0, 0), True, True, ),
            (1, True, (1, 1, 260, 260, 260), (257, 257, 257), (1, 1, 257, 257, 257), (0, 1, 2, 3, 4, 5), True, True, ),
            (1, True, (1, 1, 260, 260, 260), (3, 257, 257), (1, 1, 3, 257, 257), (0, 1, 2, 3, 4, 5), True, True, ),
            (1, True, (1, 1, 260, 260, 260), (257, 3, 257), (1, 1, 257, 3, 257), (0, 1, 2, 3, 4, 5), True, True, ),
            (1, True, (1, 1, 260, 260, 260), (257, 257, 3), (1, 1, 257, 257, 3), (0, 1, 2, 3, 4, 5), True, True, ),
            (1, True, (1, 1, 260, 260, 260), (257, 3, 3), (1, 1, 257, 3, 3), (0, 1, 2, 3, 4, 5), True, True, ),
            (1, True, (1, 1, 260, 260, 260), (3, 257, 3), (1, 1, 3, 257, 3), (0, 1, 2, 3, 4, 5), True, True, ),
            (1, True, (1, 1, 260, 260, 260), (3, 3, 257), (1, 1, 3, 3, 257), (0, 1, 2, 3, 4, 5), True, True, ),
            (1, True, (1, 1, 260, 260, 260), (3, 3, 257), (1, 1, 3, 3, 257), (0, 1, 2, 3, 4, 5), False, True, ),
            (1, True, (1, 1, 260, 260, 260), (3, 3, 257), (1, 1, 3, 3, 257), (0, 1, 2, 3, 4, 5), True, False, ),
            (1, True, (1, 1, 260, 260, 260), (3, 3, 257), (1, 1, 3, 3, 257), (0, 1, 2, 3, 4, 5), False, False, ),
        ]
        for count, expect, ishape, kshape, kweight, pads, before, after in tests:
            ishape_s = 'x'.join(str(i) for i in ishape)
            kshape_s = 'x'.join(str(i) for i in kshape)
            kweight_s = 'x'.join(str(i) for i in kweight)
            pads_s = 'x'.join(str(i) for i in pads)
            name_ = f'split_kernel_in{ishape_s}_ks{kshape_s}_kw{kweight_s}_p{pads_s}_b{int(before)}_a{int(after)}'
            with self.subTest(name=name_):
                input_ = np.random.randn(*ishape).astype(np.float32) + 0.5
                origin_file = f'onnx/{name_}.onnx'
                optimized_file = f'onnx/{name_}_splitted.onnx'
                graph = make_graph(name_, ishape, kshape, kweight, pads, before, after)
                graph.save(origin_file)

                knowledge = KnowledgeSplitLargeKernel()
                result = optimize(graph, knowledge)
                self.assertEqual(result, expect)
                if not result:
                    continue
                graph.save(optimized_file)

                for _ in range(count):
                    matrix_before_apply = inference(origin_file, [input_])
                    matrix_after_apply = inference(optimized_file, [input_])
                    self.assertTrue(len(matrix_before_apply) == len(matrix_after_apply))
                    for lmatrix, rmatrix in zip(matrix_before_apply, matrix_after_apply):
                        self.assertTrue(np.allclose(lmatrix, rmatrix, atol=1e-4, rtol=1e-2))

                result = optimize(graph, knowledge)
                self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
