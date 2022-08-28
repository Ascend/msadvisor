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

from typing import List, Dict
import numpy as np
from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.interface.base_node import BaseNode
from auto_optimizer.pattern.pattern import MATCH_PATTERN
from auto_optimizer.pattern.pattern import MatchBase
from auto_optimizer.pattern.pattern import Pattern
from auto_optimizer.pattern.matcher import MatchResult
from .knowledge_base import KnowledgeBase

# continue 4 slice op
pattern0 = Pattern() \
    .add_node('Slice_0', ['Slice']) \
    .add_node('Slice_1', ['Slice']) \
    .add_node('Slice_2', ['Slice']) \
    .add_node('Slice_3', ['Slice']) \
    .add_edge('Slice_0', 'Slice_1') \
    .add_edge('Slice_1', 'Slice_2') \
    .add_edge('Slice_2', 'Slice_3') \
    .set_input('Slice_0') \
    .set_output('Slice_3') \
    .set_loop(MATCH_PATTERN.MATCH_ONECE)

# continue 3 slice op
pattern1 = Pattern() \
    .add_node('Slice_0', ['Slice']) \
    .add_node('Slice_1', ['Slice']) \
    .add_node('Slice_2', ['Slice']) \
    .add_edge('Slice_0', 'Slice_1') \
    .add_edge('Slice_1', 'Slice_2') \
    .set_input('Slice_0') \
    .set_output('Slice_2') \
    .set_loop(MATCH_PATTERN.MATCH_ONECE)

# continue 2 slice op
pattern2 = Pattern() \
    .add_node('Slice_0', ['Slice']) \
    .add_node('Slice_1', ['Slice']) \
    .add_edge('Slice_0', 'Slice_1') \
    .set_input('Slice_0') \
    .set_output('Slice_1') \
    .set_loop(MATCH_PATTERN.MATCH_ONECE)

class KnowledgeMergeContinueSlice(KnowledgeBase):
    def __init__(self):
        super().__init__()
        self._insert_op_names = set()

    def merge_intializers(self, graph, initializer1, initializer2, merged_name):
        """
        @des        merge two initializers to one initializer
        @param      graph: input onnx graph
                    initializer1: initializer need to be merged
                    initializer2: initializer need to be merged
                    merged_name: name for merged node
        @return     merged initializer
        """
        merged_data = np.append(
            initializer1.value,
            initializer2.value,
        )

        merged_node = graph.add_initializer(name=merged_name, value=merged_data)
        return merged_node

    def merge_slicedop(self, graph, node1_name, node2_name):
        """
        @des        merge two node to one node
        @param      graph: input onnx graph
                    slice_node1: slice node1 need to be merged
                    slice_node2: slice node2 need to be merged
        @return     merged graph
        """
        node1 = graph[node1_name]
        node2 = graph[node2_name]
        if node1_name == node2_name:
            return graph
        
        # modify slice_node1 -> merge_node
        node2.inputs[1] = self.merge_intializers(
            graph,
            graph[node1.inputs[1]],
            graph[node2.inputs[1]],
            '{}_1'.format(node1.name)).name
        node2.inputs[2] = self.merge_intializers(
            graph,
            graph[node1.inputs[2]],
            graph[node2.inputs[2]],
            '{}_2'.format(node1.name)).name
        node2.inputs[3] = self.merge_intializers(
            graph,
            graph[node1.inputs[3]],
            graph[node2.inputs[3]],
            '{}_3'.format(node1.name)).name
        node2.inputs[4] = self.merge_intializers(
            graph,
            graph[node1.inputs[4]],
            graph[node2.inputs[4]],
            '{}_4'.format(node1.name)).name
        graph.remove(node1.name)
        return graph

    def _build_patterns(self) -> List[Pattern]:
        """
        知识库对应多个子图
        :return: 返回多个子图定义
        """
        return [pattern0, pattern1, pattern2]

    def _build_pattern_apply_map(self) -> Dict[Pattern, List]:
        """
        构建pattern和apply的映射关系
        :return: 返回pattern和apply方法的字典
        """
        apply_dict = {
            pattern0: [self._merge_continue_slice_apply],
            pattern1: [self._merge_continue_slice_apply],
            pattern2: [self._merge_continue_slice_apply]
        }
        return apply_dict

    def is_nodes_has_same_axis(self, graph: BaseGraph, node_dict):
        axis_list = []
        for name,node in node_dict.items():
            axis = graph[graph[node[0].name].inputs[3]].value
            print("name:{} axis:{}".format(name, axis))
            if axis in axis_list:
                return True
            axis_list.append(axis)
        return False

    def _merge_continue_slice_apply(self, graph: BaseGraph, match_result: MatchResult) -> bool:
        flag = False
        for dict in match_result.node_dicts:
            if self.is_nodes_has_same_axis(graph, dict):
                print("error check nodes has same axis can not merge")
                continue
            last_node_name = dict[list(dict.keys())[-1]][0].name
            for name,node in dict.items():
                graph = self.merge_slicedop(graph, node[0].name, last_node_name)
                flag = True
        return flag

KnowledgeFactory.add_knowledge('KnowledgeMergeContinueSlice', KnowledgeMergeContinueSlice())

