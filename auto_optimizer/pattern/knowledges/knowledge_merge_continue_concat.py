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
import logging

from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.interface.base_node import BaseNode, Node
from auto_optimizer.pattern.pattern import MATCH_PATTERN
from auto_optimizer.pattern.pattern import Pattern
from auto_optimizer.pattern.matcher import MatchResult
from .knowledge_base import KnowledgeBase


# continue 4 Concat op
pattern0 = Pattern() \
    .add_node("Concat_0", ["Concat"]) \
    .add_node("Concat_1", ["Concat"]) \
    .add_node("Concat_2", ["Concat"]) \
    .add_node("Concat_3", ["Concat"]) \
    .add_edge("Concat_0", "Concat_1") \
    .add_edge("Concat_1", "Concat_2") \
    .add_edge("Concat_2", "Concat_3") \
    .set_input("Concat_0") \
    .set_output("Concat_3") \
    .set_loop(MATCH_PATTERN.MATCH_ONCE)

# continue 3 Concat op
pattern1 = Pattern() \
    .add_node("Concat_0", ["Concat"]) \
    .add_node("Concat_1", ["Concat"]) \
    .add_node("Concat_2", ["Concat"]) \
    .add_edge("Concat_0", "Concat_1") \
    .add_edge("Concat_1", "Concat_2") \
    .set_input("Concat_0") \
    .set_output("Concat_2") \
    .set_loop(MATCH_PATTERN.MATCH_ONCE)

# continue 2 Concat op
pattern2 = Pattern() \
    .add_node("Concat_0", ["Concat"]) \
    .add_node("Concat_1", ["Concat"]) \
    .add_edge("Concat_0", "Concat_1") \
    .set_input("Concat_0") \
    .set_output("Concat_1") \
    .set_loop(MATCH_PATTERN.MATCH_ONCE)


@KnowledgeFactory.register("KnowledgeMergeContinueConcat")
class KnowledgeMergeContinueConcat(KnowledgeBase):
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
            pattern0: [self._merge_continue_concat_apply],
            pattern1: [self._merge_continue_concat_apply],
            pattern2: [self._merge_continue_concat_apply]
        }
        return apply_dict

    def check_matchinfo_need_to_optimize(self, graph: BaseGraph, nodes: List[BaseNode]) -> bool:
        """判断当前匹配的子图是否需要优化

        Args:
            graph: 整个模型的图
            nodes: 子图的节点列表

        Return:
            是否需要优化
        """
        pre_axis = None
        for idx, node in enumerate(nodes):
            if not isinstance(node, (Node, )):
                logging.info(f"Node is invalid: name {node.name} type {type(node)}")
                return False
            axis = node.attrs.get("axis", None)
            if pre_axis is not None and pre_axis != axis:
                logging.info(f"Node: {node.name} axis: {axis} not equal to pre_axis: {pre_axis}")
                return False
            pre_axis = axis
            if idx == len(nodes) - 1:
                continue
            if not node.outputs:
                logging.info(f"Node has no outputs: {node.name}")
                return False
            next_nodes = graph.get_next_nodes(node.outputs[0])
            if len(next_nodes) > 1:
                names = [node.name for node in next_nodes]
                logging.info(f"Node {node.name} has multiple outputs: {names} len {len(next_nodes)}")
                return False
        return True

    def is_concat_node(self, graph: BaseGraph, name: str) -> bool:
        pre_node = graph.get_prev_node(name)
        return pre_node is not None and pre_node.op_type == "Concat"

    def get_inputs(self, graph: BaseGraph, name: str, matchinfo: Dict[str, List[BaseNode]]) -> List[str]:
        input_list = []
        if not any(any(name == node.name for node in nodes) for nodes in matchinfo.values()):
            return input_list
        try:
            inputs = graph[name].inputs
        except (KeyError, AttributeError) as e:
            logging.info(f"Node {name} has no input: {e}")
            return input_list
        for input_name in inputs:
            pre_node = graph.get_prev_node(input_name)
            if pre_node is not None and pre_node.op_type == "Concat":
                input_list.extend(self.get_inputs(graph, pre_node.name, matchinfo))
            else:
                input_list.append(input_name)
        return input_list

    def merge_concat_nodes(self, graph: BaseGraph, matchinfo: Dict[str, List[BaseNode]]) -> bool:
        try:
            nodes = [graph[node[0].name] for node in matchinfo.values()]
        except (KeyError, IndexError, AttributeError) as e:
            logging.info(f"Failed to get node list: {e}")
            return False

        if not self.check_matchinfo_need_to_optimize(graph, nodes):
            return False

        last_node = nodes[-1]
        last_node_name = last_node.name
        input_list = self.get_inputs(graph, last_node_name, matchinfo)

        for node in nodes[:-1]:
            graph.remove(node.name)

        if not isinstance(last_node, (Node, )):
            logging.info(f"Node is invalid: name {last_node.name} type {type(last_node)}")
            return False
        last_node.inputs = input_list
        return True

    def _merge_continue_concat_apply(self, graph: BaseGraph, match_result: MatchResult) -> bool:
        flag = False
        for matchinfo in match_result.node_dicts:
            if matchinfo:
                flag |= self.merge_concat_nodes(graph, matchinfo)
        return flag
