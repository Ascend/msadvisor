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


import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx

from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.interface.base_node import Initializer, Node, PlaceHolder
from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory
from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase
from auto_optimizer.pattern.matcher import MatchResult
from auto_optimizer.pattern.pattern import MATCH_PATTERN, Pattern
from auto_optimizer.pattern.utils import AllNextnodesAreGather, is_lower_onnx_version

r"""
           PreNode                                   PreNode
           /  |  \                                      |
          /   |   \                                  Split_0
         /    |    \                                /   |   \
        /     |     \                              /    |    \
       /      |      \            =====>          /     |     \
      /       |       \                          /      |      \
     /        |        \                  NextNode0 NextNode1 NextNode2
 Gather_0   Gather_1   Gather_2
    |         |            |
NextNode0 NextNode1    NextNode2

"""

Indices = Union[int, Tuple[int, ...]]


@KnowledgeFactory.register()
class KnowledgeGatherToSplit(KnowledgeBase):
    '''Change Gather operators to a single split operator.'''
    def __init__(self):
        super().__init__()
        # 注册pattern的apply方法
        self.pattern_ = Pattern() \
            .add_node("PreNode", None, [AllNextnodesAreGather()]) \
            .set_loop(MATCH_PATTERN.MATCH_ONCE)
        self._register_apply_funcs(self.pattern_, [self._pattern_apply])

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

    def _cal_splits_and_out2sidx(
        self,
        out2gidx: Dict[str, Indices],
        dim: int
    ) -> Tuple[Optional[List[Tuple[int, bool]]], Optional[Dict[str, int]]]:
        gidx_lst = list(set(out2gidx.values()))
        if any(
            len(indices) == 0
            # this means every 1-d indices must be continuous
            or any(idx != indices[0] + i for i, idx in enumerate(indices))
            for indices in gidx_lst
            if isinstance(indices, tuple)
        ):
            return None, None

        def extreme_index(gidx_lst_, func):
            return func(
                indices_ if isinstance(indices_, int) else func(indices_)
                for indices_ in gidx_lst_
            )

        maxv = extreme_index(gidx_lst, max)
        minv = extreme_index(gidx_lst, min)
        if minv < 0 or maxv > dim:
            return None, None

        # dims are dimensions corresponding to gather axis of input shape
        # we iterate over gather indices and place them on dims
        # None means not occupied by any indices
        dims: List[Optional[Indices]] = [None for _ in range(dim)]
        for i, gidx in enumerate(gidx_lst):
            indices_ = (gidx, ) if isinstance(gidx, int) else gidx
            for idx in indices_:
                # already occupied by other indices, this means overlap, abort
                if dims[idx] is not None:
                    return None, None
                dims[idx] = gidx

        # split dims between different indices
        start, last = 0, dims[0]
        splits: List[Tuple[Optional[Indices], bool, int]] = []
        for i, idx in enumerate(dims[1:]):
            if idx != last:
                splits.append((last, isinstance(last, int), i + 1 - start))
                start = i + 1
            last = idx
        splits.append((dims[-1], isinstance(dims[-1], int), len(dims) - start))

        gidx2sidx = {
            gidx: i
            for i, (gidx, *_) in enumerate(splits)
            if gidx is not None  # pos i is not occupied if idx is None
        }

        return (
            # array of (split sizes, whether need to squeeze)
            [(size_, squeeze_) for _, squeeze_, size_ in splits],
            # output to split output index map, this is needed because
            # different gather operators may have the same indices and
            # we have to place gathers' output in the right order
            {out: gidx2sidx[gidx] for out, gidx in out2gidx.items()}
        )

    def _cal_splits_params(
        self,
        gathers: List[Node],
        graph: BaseGraph,
    ) -> Tuple[Optional[int], Optional[List[Tuple[int, bool]]], Optional[Dict[str, int]]]:
        axis = gathers[0].attrs.get('axis', 0)
        if not isinstance(axis, int):
            return None, None, None
        input_ = graph.get_node(gathers[0].inputs[0], node_type=PlaceHolder)
        if input_ is None or not input_.shape:
            return None, None, None

        # we need dim value of the gathered axis, because split require
        # the sum of split sizes equal to dim value of the specified axis
        dim = input_.shape[axis]
        if not isinstance(dim, int) or dim <= 0:
            return None, None, None

        out2gidx: Dict[str, Indices] = {}
        for gather in gathers:
            ini = graph.get_node(gather.inputs[1], node_type=Initializer)
            if ini is None:
                return None, None, None
            # count for 0-d indices and negative indices
            # if indices is a 0-d 'scalar', we need to squeeze it's output
            # after split because gather will 'squeeze' this dimension
            if ini.value.ndim == 0:
                val = int(ini.value)
                indices = dim + val if val < 0 else val
            elif ini.value.ndim == 1:
                indices = tuple(dim + v if v < 0 else v for v in ini.value)
            else:
                return None, None, None
            out2gidx[gather.outputs[0]] = indices

        splits, out2sidx = self._cal_splits_and_out2sidx(out2gidx, dim)
        return axis, splits, out2sidx

    def _add_squeeze(self, name: str, idx: int, input_: str, axis: int, graph: BaseGraph) -> str:
        output_ = f'squeeze_{name}_{idx}_out'
        if is_lower_onnx_version(graph, limit_version=13):
            graph.add_node(
                name=f'squeeze_{name}_{idx}',
                op_type='Squeeze',
                inputs=[input_],
                outputs=[output_],
                attrs={'axes': [axis]},
            )
        else:
            graph.add_initializer(
                name=f'squeeze_{name}_{idx}_axes',
                value=np.array([axis]),
            )
            graph.add_node(
                name=f'squeeze_{name}_{idx}',
                op_type='Squeeze',
                inputs=[input_, f'squeeze_{name}_{idx}_axes'],
                outputs=[output_],
            )
        return output_

    def _add_split(
        self,
        name: str,
        input_: str,
        axis: int,
        splits: List[Tuple[int, bool]],
        graph: BaseGraph,
    ) -> List[str]:
        split_outs = [f'{name}_output{i}' for i in range(len(splits))]
        split_sizes = [s for s, _ in splits]
        if is_lower_onnx_version(graph, limit_version=13):
            graph.add_node(
                name=name,
                op_type='Split',
                inputs=[input_],
                outputs=split_outs,
                attrs={'axis': axis, 'split': split_sizes},
            )
        else:
            graph.add_initializer(
                name=f'{name}_init',
                value=np.array(split_sizes),
            )
            graph.add_node(
                name=name,
                op_type='Split',
                inputs=[input_, f'{name}_init'],
                outputs=split_outs,
                attrs={'axis': axis},
            )
        outputs = []
        for i, (out, (_, need_squeeze)) in enumerate(zip(split_outs, splits)):
            if not need_squeeze:
                outputs.append(out)
                continue
            new_out = self._add_squeeze(name=name, idx=i, input_=out,
                                        axis=axis, graph=graph)
            outputs.append(new_out)
        return outputs

    def _match_apply(self, graph: BaseGraph, matchinfo) -> bool:
        pre_node = graph.get_node(matchinfo['PreNode'][0].name, node_type=Node)
        if pre_node is None:
            return False

        gathers = graph.get_next_nodes(pre_node.outputs[0])
        if not gathers or len(gathers) < 2:
            return False

        # extra checks, if any output of gathers is graph outputs, abort
        graph_outputs = [out.name for out in graph.outputs]
        if any(gather.outputs[0] in graph_outputs for gather in gathers):
            return False

        # calculate parameters we need to replace gathers to split
        # axis: the axis 'specified' by gather operators
        # splits: size of splits and whether we need to squeeze this output
        # out2sidx: gathers' output to new output index after split
        #           used for reorder output
        axis, splits, out2sidx = self._cal_splits_params(gathers, graph)
        if axis is None or splits is None or out2sidx is None:
            return False

        # modification starts from here
        split_name = f'Split_{pre_node.name}'
        split_outs = self._add_split(
            name=split_name,
            input_=pre_node.outputs[0],
            axis=axis,
            splits=splits,
            graph=graph,
        )

        # replace gathers with split
        for gather in gathers:
            new_output = split_outs[out2sidx[gather.outputs[0]]]
            for node in graph.get_next_nodes(gather.outputs[0]):
                node.inputs = [
                    new_output if input_ == gather.outputs[0] else input_
                    for input_ in node.inputs
                ]
            graph.remove(gather.name, {})

        graph.update_map()
        return True

    def _pattern_apply(self, graph: BaseGraph, match_result: MatchResult) -> bool:
        flag = False
        for matchinfo in match_result.node_dicts:
            flag |= self._match_apply(graph, matchinfo)
        return flag
