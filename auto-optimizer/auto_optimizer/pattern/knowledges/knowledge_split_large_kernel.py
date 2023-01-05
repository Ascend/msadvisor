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

from itertools import accumulate
from typing import List, Dict, Optional, Tuple
import logging
import operator as op

import numpy as np
import onnx

from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.interface.base_node import BaseNode, Node, Initializer
from auto_optimizer.pattern.pattern import MATCH_PATTERN, Pattern, MatchBase
from auto_optimizer.pattern.matcher import MatchResult
from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase


class LargeKernelConv(MatchBase):
    def __init__(self, threshold: int):
        super().__init__()
        self.threshold = threshold

    def match(self, node: BaseNode, graph: BaseGraph) -> bool:
        if not isinstance(node, (Node, )) or op.ne(node.op_type, 'Conv'):
            return False
        auto_pad: str = node.attrs.get('auto_pad', b'NOTSET')
        if auto_pad != b'NOTSET':
            return False
        strides: List[int] = node.attrs.get('strides', [1])
        dilations: List[int] = node.attrs.get('dilations', [1])
        if any(s != 1 for s in strides) or any(d != 1 for d in dilations):
            return False
        kernel_shape: List[int] = node.attrs.get('kernel_shape', [1])
        return any(ks > self.threshold for ks in kernel_shape)


# 拆分示意图
r"""
                                  ______________PreNode______________
       |                         /         ____/   |   \____         \
    PreNode                     /         /        |        \         \
       |                     slice0    slice1    slice2    slice3    slice4
       |                       |         |         |         |         |
       |            slice    conv0     conv1     conv2     conv3     conv4
 LargeKernelConv   ======>      \         \        |        /         /
       |                         \         \_____  |  _____/         /
       |                          \              \ | /              /
       |                           \______________Sum______________/
    NextNode                                       |
       |                                        NextNode
"""


@KnowledgeFactory.register()
class KnowledgeSplitLargeKernelConv(KnowledgeBase):
    """Split Large Conv Kernel to speed up inference."""

    def __init__(self, threshold: int = 192):
        super().__init__()
        # 卷积核阈值，任意一维超过该阈值均认为是大卷积核算子
        # 该阈值通过实际模型测试得到，实际阈值可能在200左右
        # 测试模型有限，无法得到确切的值
        # 由于128-176之间推理性能区别不大，取了一个保守的大值
        # 确保性能的前提下，尽量减少拆分
        self.threshold = (max(threshold - 1, 15) // 16 + 1) * 16
        self.large_kernel_match = LargeKernelConv(self.threshold)
        self.large_kernel_pattern = Pattern() \
            .add_node("LargeKernelConv", ["Conv"], [self.large_kernel_match]) \
            .set_loop(MATCH_PATTERN.MATCH_ONCE)
        self._register_apply_funcs(self.large_kernel_pattern, [self._large_kernel_pattern_apply])

    def pre_process(self, graph: BaseGraph) -> bool:
        try:
            graph.infershape()
        except onnx.onnx_cpp2py_export.shape_inference.InferenceError:
            logging.info('infershape failed before optimization.')
        return super().pre_process(graph)

    def post_process(self, graph: BaseGraph) -> bool:
        try:
            graph.infershape()
        except onnx.onnx_cpp2py_export.shape_inference.InferenceError:
            logging.info('infershape failed after optimization.')
        return super().post_process(graph)

    def _calculate_pads_and_slices(
        self, kslice: List[Tuple[int, int]], kshape: List[int], pads: List[int],
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        '''Calculate pads of Conv operator and parameters of Slice operator for each kernel slice.'''
        i32 = np.iinfo(np.int32)
        length = len(kslice)
        pads_s, pads_e = pads[:length], pads[length:]
        # 求出卷积核切片相对输入shape的移动范围
        # 起始侧负值表示需要补0，正值表示需要切片，终止侧相反
        move_range = [
            (
                -pad_s + kslice_s,           # 起始侧
                pad_e - (size_k - kslice_e)  # 终止侧
            )
            for pad_s, pad_e, (kslice_s, kslice_e), size_k in zip(pads_s, pads_e, kslice, kshape)
        ]
        # 求新的Conv算子的pads参数
        new_pads_s = [max(0, -start) for start, _ in move_range]
        new_pads_e = [max(0, end) for _, end in move_range]
        new_pads = new_pads_s + new_pads_e
        # 求slice算子的参数，当终止侧不需要slice时，取end为INT_MAX
        slices_ = [
            [
                max(0, start),                # starts
                end if end < 0 else i32.max,  # ends
                axis - length                 # axes
            ]
            for axis, (start, end) in enumerate(move_range)
            if start > 0 or end < 0           # 当不满足这个条件时，说明不需要slice
        ]
        starts, ends, axes = [s[0] for s in slices_], [s[1] for s in slices_], [s[2] for s in slices_]
        return new_pads, starts, ends, axes

    def _create_kernel_slice_branch(
        self, conv: Node, graph: BaseGraph, kslice: List[Tuple[int, int]], keep_bias: bool
    ) -> Node:
        '''Create Slice/Conv/Unsqueeze branch for each kernel slice.'''
        kweight: Initializer = graph.get_node(conv.inputs[1], node_type=Initializer)
        pads: List[int] = conv.attrs.get('pads', [1])

        identifier = '_'.join(str(i) for i, _ in kslice)
        len_extra = len(kweight.value.shape) - len(kslice)
        sliced_weight_name = f'sliced_weight_{conv.name}_{identifier}'

        # 卷积核权重切片，通过python内置的slice函数进行动态切片
        weight_slice = [slice(None)] * len(kweight.value.shape)
        for axis, (first, last) in enumerate(kslice):
            weight_slice[axis + len_extra] = slice(first, last)
        graph.add_initializer(
            name=sliced_weight_name,
            value=kweight.value[tuple(weight_slice)]
        )

        # 卷积输入切片，适配被切片的卷积核
        slice_name = f'slice_conv_{conv.name}_{identifier}'
        slice_start_name = f'{slice_name}_start'
        slice_end_name = f'{slice_name}_end'
        slice_axes_name = f'{slice_name}_axis'
        slice_output_name = f'{slice_name}_output'

        kshape: List[int] = conv.attrs.get('kernel_shape', [1])
        new_pads, start, end, axes = self._calculate_pads_and_slices(kslice, kshape, pads)
        graph.add_initializer(name=slice_start_name, value=np.array(start))
        graph.add_initializer(name=slice_end_name, value=np.array(end))
        graph.add_initializer(name=slice_axes_name, value=np.array(axes))
        graph.add_node(
            name=slice_name,
            op_type='Slice',
            inputs=[conv.inputs[0], slice_start_name, slice_end_name, slice_axes_name],
            outputs=[slice_output_name]
        )

        new_inputs = [slice_output_name, sliced_weight_name]
        # 只有一个切分后的卷积算子能保留bias输入
        if keep_bias:
            new_inputs += conv.inputs[2:]
        conv_name = f'conv_{conv.name}_{identifier}'
        return graph.add_node(
            name=conv_name,
            op_type='Conv',
            inputs=new_inputs,
            outputs=[f'{conv_name}_output'],
            attrs={
                **conv.attrs,
                'pads': new_pads,
                'group': conv.attrs.get('group', 1),
                'kernel_shape': [end_ - start_ for start_, end_ in kslice]
            }
        )

    def _calculate_kernel_slices(self, kshape: List[int]) -> List[List[Tuple[int, int]]]:
        """Calculate kernel slices for large kernel shape.
        :param kshape: kernel shape of origin conv kernel
        :return: return slices of kernel shape, each element of the returned list is a slice
        of the origin kernel.

        Example:
            kshape: [201, 5]
            return: [[(0, 100), (0, 5)], [(100, 201), (0, 5)]]
        """
        # 每个tuple都是对kernel shape某个维度的一个切片
        # 以kshape为[3, 201, 5]为例，kslices每次循环的变化如下:
        # [[]] --> [[(0, 3)]]
        # ...  --> [[(0, 3), (0, 100)], [(0, 3), (100, 201)]]
        # ...  --> [[(0, 3), (0, 100), (0, 5)], [(0, 3), (100, 201), (0, 5)]]
        # 最终kslices内每个元素都是kernel shape的一个切片
        kslices: List[List[Tuple[int, int]]] = [[]]
        for ksize in kshape:
            _16s = [16] * (ksize // 16) + ([ksize % 16] if ksize % 16 else [])
            num = (len(_16s) - 1) // (self.threshold // 16) + 1
            each = ((len(_16s) - 1) // num) + 1
            ksizes = [sum(_16s[i:i+each]) for i in range(0, len(_16s), each)]
            indices = list(accumulate([0, *ksizes[:-1]]))
            kslices = [[*slc, (i, i + s)] for i, s in zip(indices, ksizes) for slc in kslices]
        return kslices

    def _split_large_kernel(self, graph: BaseGraph, matchinfo: Dict[str, List[Node]]) -> bool:
        conv0: Optional[Node] = graph.get_node(matchinfo['LargeKernelConv'][0].name, node_type=Node)
        if conv0 is None:
            logging.warning('Conv operator is no longer exists.')
            return False

        kweight: Optional[Initializer] = graph.get_node(conv0.inputs[1], node_type=Initializer)
        if kweight is None:
            logging.warning('Failed to get conv kernel weight.')
            return False

        # modification start from here
        kshape: List[int] = conv0.attrs.get('kernel_shape', [1])
        slices = self._calculate_kernel_slices(kshape)
        outputs = [
            self._create_kernel_slice_branch(conv0, graph, slc, idx == 0).outputs[0]
            for idx, slc in enumerate(slices)
        ]

        graph.add_node(
            name=f'sum_{conv0.name}',
            op_type='Sum',
            inputs=outputs,
            outputs=[conv0.outputs[0]],
        )
        graph.remove(conv0.name, {})
        graph.update_map()
        return True

    def _large_kernel_pattern_apply(self, graph: BaseGraph, match_result: MatchResult) -> bool:
        flag = False
        for matchinfo in match_result.node_dicts:
            flag |= self._split_large_kernel(graph, matchinfo)
        return flag
