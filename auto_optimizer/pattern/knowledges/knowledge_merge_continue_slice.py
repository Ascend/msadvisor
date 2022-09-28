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
import numpy as np

from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.interface.base_node import BaseNode, Node
from auto_optimizer.pattern.pattern import MATCH_PATTERN
from auto_optimizer.pattern.pattern import MatchBase
from auto_optimizer.pattern.pattern import Pattern
from auto_optimizer.pattern.matcher import MatchResult
from .knowledge_base import KnowledgeBase

# continue 4 slice op
pattern0 = Pattern() \
    .add_node("Slice_0", ["Slice"]) \
    .add_node("Slice_1", ["Slice"]) \
    .add_node("Slice_2", ["Slice"]) \
    .add_node("Slice_3", ["Slice"]) \
    .add_edge("Slice_0", "Slice_1") \
    .add_edge("Slice_1", "Slice_2") \
    .add_edge("Slice_2", "Slice_3") \
    .set_input("Slice_0") \
    .set_output("Slice_3") \
    .set_loop(MATCH_PATTERN.MATCH_ONCE)

# continue 3 slice op
pattern1 = Pattern() \
    .add_node("Slice_0", ["Slice"]) \
    .add_node("Slice_1", ["Slice"]) \
    .add_node("Slice_2", ["Slice"]) \
    .add_edge("Slice_0", "Slice_1") \
    .add_edge("Slice_1", "Slice_2") \
    .set_input("Slice_0") \
    .set_output("Slice_2") \
    .set_loop(MATCH_PATTERN.MATCH_ONCE)

# continue 2 slice op
pattern2 = Pattern() \
    .add_node("Slice_0", ["Slice"]) \
    .add_node("Slice_1", ["Slice"]) \
    .add_edge("Slice_0", "Slice_1") \
    .set_input("Slice_0") \
    .set_output("Slice_1") \
    .set_loop(MATCH_PATTERN.MATCH_ONCE)


@KnowledgeFactory.register("KnowledgeMergeContinueSlice")
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

    def check_matchinfo_need_to_optimize(self, graph: BaseGraph, nodes: List[BaseNode], axes: List[np.ndarray]) -> bool:
        """判断当前匹配的子图是否需要优化

        Args:
            graph: 整个模型的图
            nodes: 子图的节点列表
            axes: 全部需要合并的Slice算子操作的轴列表

        Return:
            是否需要优化
        """
        axes_to_merge = np.concatenate(axes)
        if np.unique(axes_to_merge).size != axes_to_merge.size:
            logging.info(f"Nodes has duplicate slice axis: {axes_to_merge}")
            return False

        for node in nodes[:-1]:
            if not isinstance(node, (Node, )):
                logging.info(f"Node of slice match is invalid: name {node.name} type {type(node)}")
                return False
            next_nodes = graph.get_next_nodes(node.outputs[0])
            if len(next_nodes) > 1:
                names = ", ".join([node.name for node in next_nodes])
                logging.info(f"Node {node.name} has multiple outputs: {names} len {len(next_nodes)}")
                return False
        return True

    def merge_slice_nodes(self, graph: BaseGraph, matchinfo: Dict[str, List[BaseNode]]) -> bool:
        try:
            nodes = [graph[node[0].name] for node in matchinfo.values()]
            input_lists = [[graph[node.inputs[i]].value for node in nodes] for i in range(1, 5)]
        except (KeyError, IndexError, AttributeError) as e:
            logging.info(f"Failed get node list or input list: {e}")
            return False

        if not self.check_matchinfo_need_to_optimize(graph, nodes, input_lists[2]):
            return False

        nodes_to_merge = [node.name for node in nodes]

        for node in nodes[:-1]:
            graph.remove(node.name)

        last_node = nodes[-1]
        if not isinstance(last_node, (Node, )):
            logging.info(f"Node is invalid: name {last_node.name} type {type(last_node)}")
            return False
        params = ["starts_", "ends_", "axes_", "steps_"]
        for i in range(4):
            new_input = graph.add_initializer(
                name=params[i] + "_".join(nodes_to_merge),
                value=np.concatenate(input_lists[i])
            )
            last_node.inputs[i + 1] = new_input.name
        return True

    def _merge_continue_slice_apply(self, graph: BaseGraph, match_result: MatchResult) -> bool:
        flag = False
        for matchinfo in match_result.node_dicts:
            if matchinfo:
                flag |= self.merge_slice_nodes(graph, matchinfo)
        return flag
