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

import operator
import logging

import onnx
import numpy as np

from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.interface.base_node import (
    Initializer, Node, PlaceHolder
)
from auto_optimizer.pattern.pattern import MATCH_PATTERN, Pattern
from auto_optimizer.pattern.matcher import MatchResult
from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase


# topk in NPU implemented input k & output indices as int32[],
# while ONNX defines them as int64[]
# we need fix this because ATC didn't handle this correctly in some cases
r"""
                          Node0
                            |
     Node0                 Cast(to int32, if neccesary)
       |(k)                 |(k)
     TopK_0     ====>     TopK_0
     /    \               /    \ (indices)
    /      \             /      \
  Node1   Node2        Node1    Cast(to int64)
                                 |
                                Node2
"""
pattern_topk = Pattern() \
    .add_node("TopK_0", ["TopK"]) \
    .set_loop(MATCH_PATTERN.MATCH_ONCE)


@KnowledgeFactory.register()
class KnowledgeTopkFix(KnowledgeBase):
    '''Fix some input/output dtype of TopK by cast.'''

    def __init__(self):
        super().__init__()
        self._register_apply_funcs(
            pattern_topk, [self._topk_fix])

    def _insert_cast_after_indices(self, topk: Node, graph: BaseGraph) -> None:
        indices_name = topk.outputs[1]
        topk.outputs[1] = f'{indices_name}_before_cast'
        graph.add_node(
            name=f'Cast_after_{topk.name}_indices',
            op_type='Cast',
            attrs={'to': onnx.TensorProto.INT64},
            inputs=[topk.outputs[1]],
            outputs=[indices_name],
        )

    def _insert_cast_before_k(self, topk: Node, graph: BaseGraph) -> None:
        input_k = topk.inputs[1]
        topk.inputs[1] = f'{input_k}_after_cast'
        graph.add_node(
            name=f'Cast_before_{topk.name}_K',
            op_type='Cast',
            attrs={'to': onnx.TensorProto.INT32},
            inputs=[input_k],
            outputs=[topk.inputs[1]]
        )

    def _topk_fix_k(self, topk: Node, graph: BaseGraph) -> bool:
        if len(topk.inputs) < 2:
            return False
        input_k = topk.inputs[1]
        init = graph.get_node(input_k, node_type=Initializer)
        if init is not None:
            # input k is an Initializer, then we cast it to int32
            if init.value.dtype != np.int32:
                init.value = init.value.astype(np.int32)
                return True
            return False
        ph = graph.get_node(input_k, node_type=PlaceHolder)
        need_cast_k = False
        # input is not Initializer, then we insert cast on the following conditions
        # 1. we don't have value info, and we can't acquire it's type from pre node
        # 2. we have value info and it's type is not int32
        if ph is None:
            pre_node = graph.get_prev_node(input_name=input_k)
            if not isinstance(pre_node, Node) \
                    or operator.ne(pre_node.op_type, 'Cast') \
                    or pre_node.attrs.get('to') != onnx.TensorProto.INT32:
                need_cast_k = True
        if ph is not None and ph.dtype is not np.int32:
            need_cast_k = True
        if need_cast_k:
            self._insert_cast_before_k(topk, graph)
        return need_cast_k

    def _topk_fix_indices(self, topk: Node, graph: BaseGraph) -> bool:
        indices_name = topk.outputs[1]
        need_cast_indices = False
        next_nodes = graph.get_next_nodes(indices_name)
        if indices_name in [out.name for out in graph.outputs]:
            # indices is a output of graph
            need_cast_indices = True
        if any(operator.ne(n.op_type, 'Cast') for n in next_nodes):
            # some next node didn't cast indices' type
            need_cast_indices = True
        if need_cast_indices:
            self._insert_cast_after_indices(topk, graph)
        return need_cast_indices

    def _topk_fix(self, graph: BaseGraph, match_result: MatchResult) -> bool:
        flag = False
        for matchinfo in match_result.node_dicts:
            name = matchinfo['TopK_0'][0].name
            topk = graph.get_node(name, node_type=Node)
            if topk is None:
                logging.warning(f'The matching node {name} no longer exists.')
                continue
            k_fixed = self._topk_fix_k(topk, graph)
            indices_fixed = self._topk_fix_indices(topk, graph)
            graph.update_map()
            flag |= k_fixed or indices_fixed
        return flag
