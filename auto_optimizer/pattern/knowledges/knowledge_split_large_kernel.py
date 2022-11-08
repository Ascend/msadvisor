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
                                  __________PreNode_________
                                 /          __/|\__         \
                                /        __/   |   \__       \
                               /        /      |      \       \
                             slice0  slice1  slice2  slice3  slice4
                               |       |       |       |       |
                               |       |       |       |       |
                             conv_0  conv_1  conv_2  conv_3  conv_4
                                \     /        |       \       /
       PreNode                   \   /          \       \     /
          |                      add_0           \       add_1
          |          slice          \             \       /
   LargeKernelConv  ======>          \             \     /
          |                           \             add_2
          |                            \_           _/
       NextNode                          \_       _/
                                           \     /
                                            add_3
                                              |
                                           NextNode

"""


@KnowledgeFactory.register("KnowledgeSplitLargeKernel")
class KnowledgeSplitLargeKernel(KnowledgeBase):
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
        # only one conv operator keeps bias
        if keep_bias:
            new_inputs += conv.inputs[2:]
        return graph.add_node(
            name=f'conv_{identifier}',
            op_type='Conv',
            inputs=new_inputs,
            outputs=[f'conv_{identifier}_output'],
            attrs={
                'pads': new_pads,
                'group': conv.attrs.get('group', 1),
                'kernel_shape': [end_ - start_ for start_, end_ in kslice]
            }
        )

    def _slice_kernel_shape(self, kslice: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], ...]:
        '''Slice kernel shape in half if possible, else return origin slice as a tuple.'''
        first, second = kslice[:], kslice[:]
        for idx, (start, end) in enumerate(kslice):
            if end - start > self.threshold:
                mid = (end - start) // 2
                first[idx] = start, start + mid
                second[idx] = start + mid, end
                return first, second
        return (kslice, )

    def _split_kernel(self, conv: Node, graph: BaseGraph, kslice: List[Tuple[int, int]], keep_bias: bool) -> Node:
        '''Split kernel in half recursively if possible, else slice kernel.'''
        kslices = self._slice_kernel_shape(kslice)
        if len(kslices) == 1:
            # can't split anymore, end recursion here, add slice and conv operator
            return self._slice_kernel(conv, graph, kslice, keep_bias)
        # otherwise split kernel into two halves recursively
        # the reason we use recursion is Add operator only support 2 inputs
        first = self._split_kernel(conv, graph, kslices[0], keep_bias)
        second = self._split_kernel(conv, graph, kslices[1], False)

        # add two halves back together
        # we use slices to generate unique names
        name_f = '#'.join(str(x) for x, _ in kslices[0])
        name_s = '#'.join(str(x) for x, _ in kslices[1])
        add_name = f'add_{conv.name}_f{name_f}_s{name_s}'
        return graph.add_node(
            name=add_name,
            op_type='Add',
            inputs=[first.outputs[0], second.outputs[0]],
            outputs=[f'{add_name}_output']
        )

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
        kslice = [(0, s) for s in kshape]
        last_node = self._split_kernel(conv0, graph, kslice, True)
        last_node.outputs = [conv0.outputs[0]]
        graph.remove(conv0.name, {})
        return True

    def _large_kernel_pattern_apply(self, graph: BaseGraph, match_result: MatchResult) -> bool:
        flag = False
        for matchinfo in match_result.node_dicts:
            flag |= self._split_large_kernel(graph, matchinfo)
        return flag
