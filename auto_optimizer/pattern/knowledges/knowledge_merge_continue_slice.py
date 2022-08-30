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

    def check_nodesinfo_need_to_optimize(self, graph: BaseGraph, nodesinfo):
        input3_list = []
        for name,nodes in nodesinfo.items():
            node = graph[nodes[0].name]
            input3_list.append(graph[node.inputs[3]].value)
        merge_axises = np.concatenate(input3_list)
        if np.unique(merge_axises).size != merge_axises.size:
            print("warning check nodes has same axis can not merge merge_axises:{}".format(merge_axises))
            return False

        last_node_name = nodesinfo[list(nodesinfo.keys())[-1]][0].name
        for name,nodes in nodesinfo.items():
            node = graph[nodes[0].name]
            if last_node_name != nodes[0].name:
                next_nodes = graph.get_next_nodes(node.outputs[0])
                if len(next_nodes) > 1:
                    print("warning check node:{} has >1 outputs:{} len:{}".format(node, next_nodes, len(next_nodes)))
                    return False
        return True

    def merge_slice_nodes(self, graph: BaseGraph, nodesinfo):
        if self.check_nodesinfo_need_to_optimize(graph, nodesinfo) is False:
            return False
        input1_list = []
        input2_list = []
        input3_list = []
        input4_list = []
        for name,nodes in nodesinfo.items():
            node = graph[nodes[0].name]
            input1_list.append(graph[node.inputs[1]].value)
            input2_list.append(graph[node.inputs[2]].value)
            input3_list.append(graph[node.inputs[3]].value)
            input4_list.append(graph[node.inputs[4]].value)


        last_node_name = nodesinfo[list(nodesinfo.keys())[-1]][0].name
        for name,nodes in nodesinfo.items():
            node = graph[nodes[0].name]
            if last_node_name == nodes[0].name:
                graph[node.inputs[1]].value = np.concatenate(input1_list)
                graph[node.inputs[2]].value = np.concatenate(input2_list)
                graph[node.inputs[3]].value = np.concatenate(input3_list)
                graph[node.inputs[4]].value = np.concatenate(input4_list)
            else:
                graph.remove(node.name)
                graph.remove(node.inputs[1])
                graph.remove(node.inputs[2])
                graph.remove(node.inputs[3])
                graph.remove(node.inputs[4])
        return True

    def _merge_continue_slice_apply(self, graph: BaseGraph, match_result: MatchResult) -> bool:
        flag = False
        for nodesinfo in match_result.node_dicts:
            flag |= self.merge_slice_nodes(graph, nodesinfo)
        return flag

KnowledgeFactory.add_knowledge('KnowledgeMergeContinueSlice', KnowledgeMergeContinueSlice())

