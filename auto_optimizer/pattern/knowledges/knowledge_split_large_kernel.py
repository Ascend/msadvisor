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


class LargeKernel(MatchBase):
    def __init__(self, threshold: int):
        super().__init__()
        self.threshold = threshold

    def match(self, node: BaseNode, graph: BaseGraph) -> bool:
        if not isinstance(node, (Node, )) or op.ne(node.op_type, 'Conv'):
            return False
        auto_pad: str = node.attrs.get('auto_pad', 'NOTSET')
        if auto_pad != 'NOTSET':
            return False
        strides: List[int] = node.attrs.get('strides', [1])
        dilations: List[int] = node.attrs.get('dilations', [1])
        if any(s != 1 for s in strides) or any(d != 1 for d in dilations):
            return False
        kernel_shape: List[int] = node.attrs.get('kernel_shape', [1])
        return any(ks > self.threshold for ks in kernel_shape)


# pattern
r"""
                                            _________PreNode_________
                                       ____/         /  |  \         \____
                                      /         ____/   |   \____         \
                                     /         /        |        \         \
                                slice0     slice1     slice2     slice3     slice4
          |                       |          |          |          |          |
       PreNode                  conv0      conv1      conv2      conv3      conv4
          |                       |          |          |          |          |
          |                   Unsqueeze0 Unsqueeze1 Unsqueeze2 Unsqueeze3 Unsqueeze4
          |          slice           \         \        |        /         /
    LargeKernelConv  ======>          \         \___    |   ____/         /
          |                            \___         \   |  /          ___/
          |                                \_________concat__________/
          |                                             |
       NextNode                                     ReduceSum
          |                                             |
                                                     NextNode
                                                        |
"""


@KnowledgeFactory.register("KnowledgeSplitLargeKernel")
class KnowledgeSplitLargeKernel(KnowledgeBase):
    """Split Large Conv Kernel to speed up inference."""
    def __init__(self):
        super().__init__()
        # large kernel threshold, test with real model
        self.threshold = 176
        self.large_kernel_match = LargeKernel(self.threshold)
        self.large_kernel_pattern = Pattern() \
            .add_node("LargeKernelConv", ["Conv"], [self.large_kernel_match]) \
            .set_input("LargeKernelConv") \
            .set_output("LargeKernelConv") \
            .set_loop(MATCH_PATTERN.MATCH_ONCE)
        self._register_apply_funcs(self.large_kernel_pattern, [self._large_kernel_pattern_apply])

    def pre_process(self, graph: BaseGraph) -> bool:
        try:
            graph.infershape()
        except onnx.onnx_cpp2py_export.shape_inference.InferenceError:
            logging.info('infershape failed before optimization.')
        return True

    def post_process(self, graph: BaseGraph) -> bool:
        try:
            graph.infershape()
        except onnx.onnx_cpp2py_export.shape_inference.InferenceError:
            logging.info('infershape failed after optimization.')
        return True

    def _pads_and_slices(
        self, kslice: List[Tuple[int, int]], kshape: List[int], pads: List[int],
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        '''Calculate new pads and slice parameters.'''
        i32 = np.iinfo(np.int32)
        length = len(kslice)
        lpads, rpads = pads[:length], pads[length:]
        # this is the relative input range where kernel slice could move
        input_slices = [
            (
                -pad_l + slice_k_l,           # start
                pad_r - (size_k - slice_k_r)  # end
            )
            for pad_l, pad_r, (slice_k_l, slice_k_r), size_k in zip(lpads, rpads, kslice, kshape)
        ]
        new_lpads = [max(0, -start) for start, _ in input_slices]
        new_rpads = [max(0, end) for _, end in input_slices]
        slices_ = [
            [max(0, start), end if end < 0 else i32.max, axis - length]
            for axis, (start, end) in enumerate(input_slices)
            if start > 0 or end < 0
        ]
        return new_lpads + new_rpads, [s[0] for s in slices_], [s[1] for s in slices_], [s[2] for s in slices_]

    def _slice_kernel(self, conv: Node, graph: BaseGraph, kslice: List[Tuple[int, int]], keep_bias: bool) -> Node:
        '''Add slice and conv operators to slice kernel.'''
        kweight: Initializer = graph.get_node(conv.inputs[1], node_type=Initializer)
        pads: List[int] = conv.attrs.get('pads', [1])

        identifier = '_'.join(str(i) for i, _ in kslice)
        len_extra = len(kweight.value.shape) - len(kslice)
        sliced_weight_name = f'sliced_weight_{conv.name}_{identifier}'

        # slice kernel weight
        weight_slice = [slice(None)] * len(kweight.value.shape)
        for axes, (first, last) in enumerate(kslice):
            weight_slice[axes + len_extra] = slice(first, last)
        graph.add_initializer(
            name=sliced_weight_name,
            value=kweight.value[tuple(weight_slice)]
        )

        # slice conv input
        slice_name = f'slice_conv_{conv.name}_{identifier}'
        slice_start_name = f'{slice_name}_start'
        slice_end_name = f'{slice_name}_end'
        slice_axes_name = f'{slice_name}_axis'
        slice_output_name = f'{slice_name}_output'

        kshape: List[int] = conv.attrs.get('kernel_shape', [1])
        new_pads, start, end, axes = self._pads_and_slices(kslice, kshape, pads)
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
        # only one conv operator can keeps bias
        if keep_bias:
            new_inputs += conv.inputs[2:]
        conv_name = f'conv_{conv.name}_{identifier}'
        return graph.add_node(
            name=conv_name,
            op_type='Conv',
            inputs=new_inputs,
            outputs=[f'{conv_name}_output'],
            attrs={
                'pads': new_pads,
                'group': conv.attrs.get('group', 1),
                'kernel_shape': [end_ - start_ for start_, end_ in kslice]
            }
        )

    def _kernel_slices(self, kshape: List[int]) -> List[List[Tuple[int, int]]]:
        kslices = [[]]
        for ksize in kshape:
            n = (ksize - 1) // self.threshold + 1
            k, e = ksize // n, ksize % n
            o = (n - e) // 2
            ksizes = [k + 1 if o <= i < o + e else k for i in range(n)]
            indices = list(accumulate([0, *ksizes[:-1]]))
            kslices = [[*slc, (i, i + s)] for i, s in zip(indices, ksizes) for slc in kslices]
        return kslices

    def _split_large_kernel(self, graph: BaseGraph, matchinfo: Dict[str, List[BaseNode]]) -> bool:
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
        slices = self._kernel_slices(kshape)
        convs = [self._slice_kernel(conv0, graph, slc, idx == 0) for idx, slc in enumerate(slices)]

        outputs = []
        for conv in convs:
            unsqueeze_name = f'unsqueeze_after_{conv.name}'
            unsqueeze_node = graph.add_node(
                name=unsqueeze_name,
                op_type='Unsqueeze',
                inputs=[conv.outputs[0]],
                outputs=[f'{unsqueeze_name}_output'],
                attrs={'axes': [0]}
            )
            outputs.extend(unsqueeze_node.outputs)

        concat_node = graph.add_node(
            name=f'concat_{conv0.name}',
            op_type='Concat',
            inputs=outputs,
            outputs=[f'concat_{conv0.name}_output'],
            attrs={'axis': 0}
        )
        graph.add_node(
            name=f'reducesum_after_{concat_node.name}',
            op_type='ReduceSum',
            inputs=[concat_node.outputs[0]],
            outputs=[conv0.outputs[0]],
            attrs={'axes': [0], 'keepdims': 0}
        )
        graph.remove(conv0.name, {})
        return True

    def _large_kernel_pattern_apply(self, graph: BaseGraph, match_result: MatchResult) -> bool:
        flag = False
        for matchinfo in match_result.node_dicts:
            flag |= self._split_large_kernel(graph, matchinfo)
        return flag
