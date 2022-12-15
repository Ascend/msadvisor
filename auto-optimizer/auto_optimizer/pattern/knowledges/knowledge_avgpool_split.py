# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

import copy
import numpy as np

from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory
from auto_optimizer.pattern.pattern import Pattern, MATCH_PATTERN, MatchBase
from auto_optimizer.pattern.matcher import MatchResult
from auto_optimizer.graph_refactor.interface.base_graph import (BaseGraph, Node)
from auto_optimizer.graph_refactor.interface.base_node import BaseNode
from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase


KERNEL_MAX_SIZE = 255


class OpMatch(MatchBase):
    def __init__(self):
        super().__init__()

    def match(self, node: BaseNode, graph: BaseGraph) -> bool:
        if node is None:
            return False
        if node.attrs.get('kernel_shape') is None or \
            node.attrs.get('strides') is None:
            return False
        # kernel_shape need to equal strides
        kernel_shape = node.attrs['kernel_shape']
        strides = node.attrs['strides']
        if len(kernel_shape) != 2 or len(strides) != 2:
            return False
        h = kernel_shape[0]
        w = kernel_shape[1]
        if h != strides[0] or w != strides[1]:
            return False
        return h * w >= KERNEL_MAX_SIZE


@KnowledgeFactory.register()
class KnowledgeAvgPoolSplit(KnowledgeBase):
    """ split AvgPool to multi-concat little AvgPool if kernal size large than 255 """

    def __init__(self) -> None:
        super().__init__()

        pattern = Pattern() \
            .add_node('AvgPool', ['AveragePool'], [OpMatch()]) \
            .set_input('AvgPool') \
            .set_output('AvgPool') \
            .set_node_loop('AvgPool', MATCH_PATTERN.MATCH_ONCE)
        self._register_apply_funcs(pattern, [self._optimize_apply])

    def _calculate_mini_factor(self, n):
        '''
        calculate mini factor which number divided, if number can not be divided, then the factor is 1
        for example: 
            if n = 10, the mini factor is 2
            if n = 55, the mini factor is 5
            if n = 3, the mini factor is 1, because 3 cannot be divided
        '''
        factor = 2
        while factor * factor <= n:
            if n % factor == 0:
                return (factor, n // factor)
            factor += 1
        # cannot split n
        return (1, n)

    def _calculate_split_func(self, node: BaseNode):
        '''
        calculate kernel_shape split strategy
        '''
        kernel_shape = node.attrs['kernel_shape']
        splits = []
        tmp_h, tmp_w = 1, 1
        h, w = kernel_shape[0], kernel_shape[1]
        h_can_split, w_can_split = True, True
        while True:
            if h >= w and h_can_split:
                # split h
                factor, h = self._calculate_mini_factor(h)
                h_can_split = False if factor == 1 else True
                tmp_h *= factor
            elif w_can_split:
                # split w
                factor, w = self._calculate_mini_factor(w)
                w_can_split = False if factor == 1 else True
                tmp_w *= factor
            else:
                # split failed
                return []
            if h * w > KERNEL_MAX_SIZE:
                continue
            splits.append([h, w])
            if tmp_h * tmp_w < KERNEL_MAX_SIZE:
                splits.append([tmp_h, tmp_w])
                break
            h, w = tmp_h, tmp_w
            tmp_h, tmp_w = 1, 1
        return splits

    def _optimize_avgpool(self, graph: BaseGraph, node: BaseNode, splits):
        '''
        optimize AveragePool to multi little AveragePool by split kernel_shape and strides
        '''
        prev_node = graph.get_prev_node(node.inputs[0])
        for i, split in enumerate(splits):
            attrs = copy.deepcopy(node.attrs)
            attrs['kernel_shape'] = np.array(split, dtype = np.int64)
            attrs['strides'] = np.array(split, dtype = np.int64)
            # add new AveragePool
            new_node = graph.add_node(f'{node.name}_{i}', 'AveragePool', attrs = attrs)
            if not isinstance(prev_node, Node):
                graph.insert_node(node.name, new_node, mode = 'before', refer_index = 0)
            else:
                graph.insert_node(prev_node.name, new_node, mode = 'after', refer_index = 0)
            # old AveragePool --> AveragePool, AveragePool, ...
            prev_node = new_node
        # remove old AveragePool
        graph.remove(node.name)
        return True

    def _optimize_apply(self, graph: BaseGraph, match_result: MatchResult):
        '''
        split AveragePool
        '''
        if match_result is None or match_result.is_empty():
            return False

        optimized_result = False
        for node_dict in match_result.node_dicts:
            nodes = node_dict.get('AvgPool')
            if nodes is None or len(nodes) != 1:
                continue
            # split operator kernel_shape
            splits = self._calculate_split_func(nodes[0])
            if len(splits) <= 1:
                # split kernel_shape failed
                continue
            # split operator to multi little operator
            optimized_result |= self._optimize_avgpool(graph, nodes[0], splits)
        return optimized_result

