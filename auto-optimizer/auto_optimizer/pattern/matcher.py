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

import copy
import types
from typing import Callable, List, Dict, Optional, Set
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.interface.base_node import Node
from .pattern import Pattern, PatternNode


class MatchResult(object):
    def __init__(self, pattern: Pattern) -> None:
        self.pattern: Pattern = pattern
        self.node_dicts: List[Dict[str, List[Node]]] = []

    def connected(self, match_result: 'MatchResult') -> bool:
        """
        判断两个子图之间是否可以连接，如果可以连接则返回True，否则返回False
        """
        l_node_inputs = set()
        l_node_outputs = set()
        r_node_inputs = set()
        r_node_outputs = set()

        for node_dict in self.node_dicts:
            for nodes in node_dict.values():
                for node in nodes:
                    l_node_inputs.update(node.inputs)
                    l_node_outputs.update(node.outputs)
        for node_dict in match_result.node_dicts:
            for nodes in node_dict.values():
                for node in nodes:
                    r_node_inputs.update(node.inputs)
                    r_node_outputs.update(node.outputs)
        return bool(l_node_inputs & r_node_outputs) or bool(l_node_outputs & r_node_inputs)

    def merge(self, match_result: 'MatchResult') -> None:
        """
        合并子图，将不存在包含关系的子图合并到一个列表
        """
        cur_del_indexs = []
        for node_dict in match_result.node_dicts:
            can_insert = True
            for i, cur_node_dict in enumerate(self.node_dicts):
                if self._include(node_dict, cur_node_dict):
                    cur_del_indexs.append(i)
                    continue
                if self._include(cur_node_dict, node_dict):
                    can_insert = False
                    break
            cur_del_indexs.reverse()
            for i in cur_del_indexs:
                self.node_dicts.pop(i)
            cur_del_indexs.clear()
            if can_insert:
                self.node_dicts.append(node_dict)

    def add_node_dict(self, node_dict: Dict[str, List[Node]]) -> None:
        """
        添加子图匹配到的节点数据
        :param node_dict:子图匹配后的所有节点，字典key是算子名，value是实际算子节点
        """
        self.node_dicts.append(node_dict)

    def is_empty(self) -> bool:
        """
        判断当前匹配结果是否为空
        :return 匹配结果为空则返回True，否则返回False
        """
        if len(self.node_dicts) == 0:
            return True
        for node_dict in self.node_dicts:
            if len(node_dict) != 0:
                return False
        return True

    def _include(
        self,
        l_node_dict: Dict[str, List[Node]],
        r_node_dict: Dict[str, List[Node]]
    ) -> bool:
        """
        检查子图是否包含另一个子图，如果包含返回True，否则返回False
        """
        for name, r_nodes in r_node_dict.items():
            if l_node_dict.get(name) is None:
                return False
            if len(r_nodes) > len(l_node_dict.get(name)):
                return False
            for r_node in r_nodes:
                is_exist = False
                for l_node in l_node_dict.get(name):
                    if l_node.name == r_node.name:
                        is_exist = True
                        break
                if not is_exist:
                    return False
        return True


class Matcher(object):
    def __init__(self, graph: BaseGraph, pattern: Pattern) -> None:
        self._graph: BaseGraph = graph
        self._pattern: Pattern = pattern
        self._visit_direction: int = 0 # 0: visit from up to down; 1: visit from down to up

    def get_candidate_nodes(self) -> List[Node]:
        """
        根据定义的子图的输入/输出节点，匹配计算图中所有候选节点
        :return: 返回候选节点列表
        """
        start_pattern_node = self._pattern.get_start_node()
        if start_pattern_node is None:
            return []

        ret: List[Node] = []
        hash_set: Set[str] = set()
        for node in self._graph.nodes:
            if not start_pattern_node.match(node, self._graph):
                continue
            if node.name in hash_set:
                continue
            ret.append(node)
            hash_set.add(node.name)
        return ret

    def __get_prev_nodes(self, cur_node: Node) -> List[Node]:
        """
        根据节点输入名，获取所有该节点的前置节点
        """
        prev_nodes: Set[Node] = set()
        for input_name in cur_node.inputs:
            node = self._graph.get_prev_node(input_name)
            if node is None:
                continue
            prev_nodes.add(node)
        return list(prev_nodes)

    def __get_next_nodes(self, cur_node: Node) -> List[Node]:
        """
        获取所有子节点
        """
        next_nodes: Set[Node] = set()
        for output_ in cur_node.outputs:
            nodes = self._graph.get_next_nodes(output_)
            next_nodes = next_nodes.union(nodes)
        return list(next_nodes)

    def __get_prev_pattern_nodes(
        self,
        pattern_node: PatternNode
    ) -> List[PatternNode]:
        """
        获取节点前置节点
        :param pattern_node: 算子节点模板
        """
        return pattern_node.inputs

    def __get_next_pattern_nodes(
        self,
        pattern_node: PatternNode
    ) -> List[PatternNode]:
        """
        获取节点后置节点
        :param pattern_node: 算子节点模板
        """
        return pattern_node.outputs

    def __nodes_group_dfs(
        self,
        nodes: List[Node],
        pattern_nodes: List[PatternNode],
        nodes_map: Dict[PatternNode, Optional[Node]],
        nodes_map_group: List[Dict[PatternNode, Optional[Node]]],
        get_next_func: Callable[[PatternNode], List[PatternNode]]
    ) -> None:
        """
        匹配nodes和pattern_nodes，生成所有能匹配的组合
        :param nodes: 实际算子节点列表
        :param pattern_nodes: 算子节点列表，知识库中定义的算子节点
        :param nodes_map: nodes和pattern_nodes中匹配的节点
        :param nodes_map_group: nodes和pattern_nodes可以匹配的组合的一个集合
        :param get_next_func: 获取前置或者后置节点的方法
        """
        if len(nodes_map) == len(pattern_nodes):
            nodes_map_group.append(copy.deepcopy(nodes_map))
            return
        for pattern_node in pattern_nodes:
            if pattern_node in nodes_map:
                continue
            for node in nodes:
                if node in nodes_map.values():
                    continue
                if not pattern_node.match(node, self._graph):
                    continue
                nodes_map[pattern_node] = node
                self.__nodes_group_dfs(
                    nodes,
                    pattern_nodes,
                    nodes_map,
                    nodes_map_group,
                    get_next_func
                )
            if pattern_node in nodes_map:
                nodes_map.pop(pattern_node)
                continue
            if pattern_node.can_match_zero_time():
                # 节点没有成功匹配过，但节点可以匹配0次
                nodes_map[pattern_node] = None
                if not isinstance(get_next_func, types.MethodType):
                    continue
                # 根据回调函数获取pattern_node的前置节点或者后置节点
                new_pattern_nodes: List[PatternNode] = get_next_func(pattern_node)
                if len(new_pattern_nodes) != 0:
                    pattern_nodes.extend(new_pattern_nodes)
                self.__nodes_group_dfs(
                    nodes,
                    pattern_nodes,
                    nodes_map,
                    nodes_map_group,
                    get_next_func
                )
                nodes_map.pop(pattern_node)
                for nd in new_pattern_nodes:
                    pattern_nodes.remove(nd)

    def __match_nodes(
        self,
        nodes: List[Node],
        pattern_nodes: List[PatternNode],
        result: Dict[str, List[Node]],
        get_next_func: Callable[[PatternNode], List[PatternNode]]
    ) -> bool:
        """
        对nodes和pattern_nodes进行匹配，并基于这些节点往上或者往下继续遍历
        :param nodes: 实际算子节点列表
        :param pattern_nodes: 算子节点列表，知识库中定义的算子节点
        :param result: 匹配结果
        :param get_next_func: 获取前置或者后置节点的方法
        :return: 匹配成功则返回True，失败返回False
        """
        if len(pattern_nodes) == 0:
            return True
        # 计算nodes和pattern_nodes所有可能存在的组合
        nodes_map: Dict[PatternNode, Optional[Node]] = {}
        nodes_map_group: List[Dict[PatternNode, Optional[Node]]] = []
        self.__nodes_group_dfs(
            nodes,
            pattern_nodes,
            nodes_map,
            nodes_map_group,
            get_next_func
        )
        if len(nodes_map_group) == 0:
            return False
        # 逐个尝试nodes_groups匹配组合，只要有能成功完成子图匹配的组合，则匹配成功并且返回
        for nodes_map in nodes_map_group:
            if len(nodes_map) == 0:
                continue
            ret = True
            for pattern_node, node in nodes_map.items():
                if node is None:
                    continue
                ret = self.__graph_bfs(node, pattern_node, result)
                if not ret:
                    break
            if ret:
                return True
        return False

    def __match_prev_nodes(
        self,
        node: Node,
        pattern_node: PatternNode,
        result: Dict[str, List[Node]],
        visited: Set[Node]
    ) -> bool:
        """
        匹配node前置节点
        :param node: 实际算子节点
        :param pattern_node: 算子节点模板
        :param result: 匹配结果
        :return: 匹配成功则返回True，否则返回False
        """
        if pattern_node.can_match_more_time():
            root_pattern_node = self._pattern.get_start_node()
            if pattern_node is not root_pattern_node:
                prev_nodes = self.__get_prev_nodes(node)
                for prev_node in prev_nodes:
                    if not pattern_node.match(prev_node, self._graph):
                        continue
                    if prev_node in visited:
                        continue
                    result[pattern_node.op_name].append(prev_node)
                    if self.__match_prev_nodes(
                        prev_node,
                        pattern_node,
                        result,
                        visited = visited
                    ):
                        return True
                    visited.add(prev_node)
                    result[pattern_node.op_name].pop()
        if len(pattern_node.inputs) == 0:
            root_pattern_node = self._pattern.get_start_node()
            prev_pattern_nodes = root_pattern_node.inputs
            prev_nodes = self.__get_prev_nodes(result[root_pattern_node.op_name][0])
        else:
            prev_pattern_nodes = pattern_node.inputs
            prev_nodes = self.__get_prev_nodes(node)
        return self.__match_nodes(
            prev_nodes,
            prev_pattern_nodes,
            result,
            self.__get_prev_pattern_nodes
        )

    def __match_next_nodes(
        self,
        node: Node,
        pattern_node: PatternNode,
        result: Dict[str, List[Node]],
        visited: Set[Node]
    ) -> bool:
        """
        匹配node后置节点
        :param node: 实际算子节点
        :param pattern_node: 算子节点模板
        :param result: 匹配结果
        :return: 匹配成功则返回True，否则返回False
        """
        if pattern_node.can_match_more_time():
            next_nodes = self.__get_next_nodes(node)
            for next_node in next_nodes:
                if not pattern_node.match(next_node, self._graph):
                    continue
                if next_node in visited:
                    continue
                result[pattern_node.op_name].append(next_node)
                if self.__match_next_nodes(
                    next_node,
                    pattern_node,
                    result,
                    visited = visited
                ):
                    return True
                visited.add(next_node)
                result[pattern_node.op_name].pop()
        if len(pattern_node.outputs) == 0:
            root_pattern_node = self._pattern.get_start_node()
            next_pattern_nodes = root_pattern_node.outputs
            next_nodes = self.__get_next_nodes(result[root_pattern_node.op_name][-1])
        else:
            next_pattern_nodes = pattern_node.outputs
            next_nodes = self.__get_next_nodes(node)
        return self.__match_nodes(
            next_nodes,
            next_pattern_nodes,
            result,
            self.__get_next_pattern_nodes
        )

    def __graph_bfs(
        self,
        node: Node,
        pattern_node: PatternNode,
        result: Dict[str, List[Node]]
    ) -> bool:
        """
        从node开始匹配子图，应用广度优先算法
        :param node: 实际算子节点
        :param pattern_node: 算子节点模板
        :param result: 匹配结果
        :return: 匹配成功则返回True，否则返回False
        """
        if not pattern_node.match(node, self._graph):
            return False
        if pattern_node is self._pattern.get_start_node():
            if pattern_node.op_name not in result:
                result[pattern_node.op_name] = [node]
        else:
            if pattern_node.op_name in result:
                for idx, item in enumerate(result[pattern_node.op_name]):
                    if node.name == item.name:
                        return idx == 0
                return False
            result[pattern_node.op_name] = [node]

        if self._visit_direction == 0:
            if not self.__match_next_nodes(
                node,
                pattern_node,
                result,
                visited = set()
            ):
                result.pop(pattern_node.op_name)
                return False
        else:
            if not self.__match_prev_nodes(
                node,
                pattern_node,
                result,
                visited = set()
            ):
                result.pop(pattern_node.op_name)
                return False
        return True

    def get_match_map(self, node: Node) -> MatchResult:
        """
        获取匹配的节点列表
        :param node: 子图遍历起始节点
        :return: 匹配结果
        """
        result = MatchResult(self._pattern)

        start_pattern_node = self._pattern.get_start_node()
        if not start_pattern_node.match(node, self._graph):
            return result

        match_nodes: Dict[str, List[Node]] = {}
        if len(start_pattern_node.inputs) != 0:
            # visit from down to up
            self._visit_direction = 1
            if not self.__graph_bfs(node, start_pattern_node, match_nodes):
                return result
            for nodes in match_nodes.values():
                nodes.reverse()
        # visit from up to down
        self._visit_direction = 0
        if not self.__graph_bfs(node, start_pattern_node, match_nodes):
            return result
        result.add_node_dict(match_nodes)
        return result
