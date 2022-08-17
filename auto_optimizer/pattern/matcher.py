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
import operator as op
from enum import Enum
from typing import List, Dict
from magiconnx.interface import BaseGraph as GraphBase
from magiconnx.interface import BaseNode as NodeBase
from .pattern import PatternNode
from .pattern import VISIT_DIRECTION


class MatchResult(object):
    def __init__(self, pattern):
        self.pattern = pattern
        self.node_dicts = []

    def add_node_dict(self, node_dict: Dict[str, List[NodeBase]]):
        """
        添加子图匹配到的节点数据
        :param node_dict:子图匹配后的所有节点，字典key是算子名，value是实际算子节点
        """
        visit_direction = self.pattern.get_match_direction()
        if visit_direction == VISIT_DIRECTION.DOWN_UP:
            self.node_dicts.insert(0, node_dict)
        else:
            self.node_dicts.append(node_dict)

    def count(self):
        """
        返回匹配结果的数量
        """
        return len(self.node_dicts)

    def is_empty(self):
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

    def include(self, match_result):
        """
        比较两个匹配的结果是否存在包含关系
        :param match_result: 其他子图匹配的结果
        :return :如果包含match_result，则返回True，否则返回False
        """
        if len(self.node_dicts) < len(match_result.node_dicts):
            return False
        for other_node_dict in match_result.node_dicts:
            # 比较两个子图是否相等
            exist_same_sub_graph = False
            for node_dict in self.node_dicts:
                if len(other_node_dict) != len(node_dict):
                    continue

                is_equal = True
                for op_name, nodes in node_dict.items():
                    other_nodes = other_node_dict.get(op_name)
                    if len(nodes) != len(other_nodes):
                        is_equal = False
                        break
                    node_names = set()
                    for node in nodes:
                        node_names.add(node.name)
                    for node in other_nodes:
                        if node.name not in node_names:
                            is_equal = False
                            break
                    if not is_equal:
                        break
                if is_equal:
                    exist_same_sub_graph = True
            if not exist_same_sub_graph:
                return False
        return True


class Matcher(object):
    def __init__(self, graph, pattern):
        self._graph = graph
        self._pattern = pattern

    def get_candidate_nodes(self) -> List[NodeBase]:
        """
        遍历子图，通过匹配Pattern输入或者输出节点，获取候选节点
        当前主要针对单节点输入或者单节点输出
        :return: 返回候选节点列表
        """
        start_pattern_node = self._pattern.get_start_node()
        if start_pattern_node is None:
            return []

        ret = [] # List[NodeBase]
        hash_set = set()
        for node in self._graph._all_ops_map.values():
            if not start_pattern_node.match(node, self._graph):
                continue
            if node.name in hash_set:
                continue
            ret.append(node)
            hash_set.add(node.name)
        return ret

    def __get_prev_nodes(self, cur_node: NodeBase):
        """
        根据节点输入名，获取所有该节点的前置节点
        """
        prev_nodes = set()
        for node in self._graph._all_ops_map.values():
            if len(node.outputs) == 0:
                continue
            for prev_output_name in node.outputs:
                for input_name in cur_node.inputs:
                    if op.eq(prev_output_name, input_name):
                        prev_nodes.add(node)
                        break
        return list(prev_nodes)

    def __get_next_nodes(self, cur_node: NodeBase):
        """
        根据节点输出，获取所有该节点的后置节点
        """
        next_nodes = set()
        for node in self._graph._all_ops_map.values():
            if len(node.inputs) == 0:
                continue
            for next_input_name in node.inputs:
                for output_name in cur_node.outputs:
                    if op.eq(next_input_name, output_name):
                        next_nodes.add(node)
                        break
        return list(next_nodes)

    def __get_prev_pattern_nodes(self, pattern_node):
        """
        获取节点前置节点
        """
        return pattern_node.inputs

    def __get_next_pattern_nodes(self, pattern_node):
        """
        获取节点后置节点
        """
        return pattern_node.outputs

    def __nodes_group_dfs(self, nodes, pattern_nodes, visited, nodes_group, callback):
        """
        匹配nodes和pattern_nodes，生成所有能匹配的组合
        :param nodes_group: List[Dict{PatternNode, node}]
        """
        if len(visited) == len(pattern_nodes):
            nodes_group.append(copy.deepcopy(visited))
            return
        for pattern_node in pattern_nodes:
            if pattern_node in visited:
                continue
            for node in nodes:
                if node in visited.values():
                    continue
                if not pattern_node.match(node, self._graph):
                    continue
                visited[pattern_node] = node
                self.__nodes_group_dfs(nodes, pattern_nodes, visited, nodes_group, callback)
            if pattern_node in visited:
                visited.pop(pattern_node)
                continue
            if self._pattern.node_cann_match_zero(pattern_node.op_name):
                # 节点没有成功匹配过，且节点是可以匹配零次
                visited[pattern_node] = None
                if not isinstance(callback, types.MethodType):
                    print('callback is not method type.')
                    continue
                # 根据回调函数获取pattern_node的前置节点或者后置节点
                new_pattern_nodes = callback(pattern_node)
                if len(new_pattern_nodes) != 0:
                    pattern_nodes.extend(new_pattern_nodes)
                self.__nodes_group_dfs(nodes, pattern_nodes, visited, nodes_group, callback)
                visited.pop(pattern_node)

    def __match_nodes(self, nodes, pattern_nodes, result, callback) -> bool:
        """
        匹配输入的节点
        """
        if len(pattern_nodes) == 0:
            return True
        # 计算nodes和pattern_nodes可能存在的组合
        visited = {}
        nodes_group = []
        self.__nodes_group_dfs(nodes, pattern_nodes, visited, nodes_group, callback)
        if len(nodes_group) == 0:
            return False
        # 尝试nodes和pattern_nodes所有的组合，找出能成功匹配子图的组合
        for group in nodes_group:
            if len(group) == 0:
                continue
            match_result = True
            for pattern_node, node in group.items():
                if node is None:
                    continue
                match_result = self.__graph_bfs(node, pattern_node, result)
                if not match_result:
                    break
            if match_result:
                return True
        return False

    def __match_continuous_nodes(self, nodes: List[NodeBase], pattern_node: PatternNode, result, callback) -> bool:
        """
        匹配连续的节点
        """
        if not self._pattern.node_cann_match_more(pattern_node.op_name):
            # 节点不支持连续匹配
            return False
        match_continuous_nodes = []
        for node in nodes:
            if not pattern_node.match(node, self._graph):
                continue
            if pattern_node.op_name not in result:
                result[pattern_node.op_name] = [node]
            else:
                result[pattern_node.op_name].append(node)
            if not isinstance(callback, types.MethodType):
                return False
            if callback(node, pattern_node, result):
                match_continuous_nodes.append(node)
            else:
                result[pattern_node.op_name].remove(node)
        return len(match_continuous_nodes) == len(nodes)

    def __match_prev_nodes(self, node: NodeBase, pattern_node: PatternNode, result: Dict[str, List[NodeBase]]) -> bool:
        """
        匹配前置节点
        """
        prev_nodes = self.__get_prev_nodes(node)
        if not self.__match_continuous_nodes(prev_nodes, pattern_node, result, self.__match_prev_nodes):
            return self.__match_nodes(prev_nodes, pattern_node.inputs, result, self.__get_prev_pattern_nodes)
        return True

    def __match_next_nodes(self, node: NodeBase, pattern_node: PatternNode, result: Dict[str, List[NodeBase]]) -> bool:
        """
        匹配后置节点
        """
        next_nodes = self.__get_next_nodes(node)
        if not self.__match_continuous_nodes(next_nodes, pattern_node, result, self.__match_next_nodes):
            return self.__match_nodes(next_nodes, pattern_node.outputs, result, self.__get_next_pattern_nodes)
        return True

    def __graph_bfs(self, node: NodeBase, pattern_node: PatternNode, result: Dict[str, List[NodeBase]]) -> bool:
        """
        从node开始匹配子图，应用广度优先算法
        """
        if not pattern_node.match(node, self._graph):
            return False
        if pattern_node.op_name in result:
            return True
        result[pattern_node.op_name] = [node]

        # 遍历前置节点
        match_prev_result = True
        if len(pattern_node.inputs) != 0 or self._pattern.node_cann_match_more(pattern_node.op_name):
            match_prev_result = self.__match_prev_nodes(node, pattern_node, result)
        # 遍历后置节点
        match_next_result = True
        if len(pattern_node.outputs) != 0 or self._pattern.node_cann_match_more(pattern_node.op_name):
            match_next_result = self.__match_next_nodes(node, pattern_node, result)
        # 结果处理
        if not match_prev_result or not match_next_result:
            result.pop(pattern_node.op_name)
            return False
        return True

    def __find_next_start_node(self, start_pattern_node, match_nodes):
        """
        获取下个子图遍历的起始节点
        根据遍历的顺序不同，起始节点不同
        """
        start_nodes = []
        visit_direction = self._pattern.get_match_direction()
        if visit_direction == VISIT_DIRECTION.DOWN_UP:
            pattern_prev_node = self._pattern.in_nodes[0]
            while pattern_prev_node.op_name not in match_nodes:
                if not self._pattern.node_cann_match_zero(pattern_prev_node.op_name):
                    break
                if len(pattern_prev_node.outputs) != 0:
                    return []
                pattern_prev_node = pattern_prev_node.outputs[0]
            if pattern_prev_node.op_name not in match_nodes:
                return []
            prev_nodes = self.__get_prev_nodes(match_nodes[pattern_prev_node.op_name][-1])
            for prev_node in prev_nodes:
                if start_pattern_node.match(prev_node, self._graph):
                    start_nodes.append(prev_node)
        elif visit_direction == VISIT_DIRECTION.UP_DOWN:
            pattern_out_node = self._pattern.out_nodes[0]
            while pattern_out_node.op_name not in match_nodes:
                if not self._pattern.node_cann_match_zero(pattern_out_node.op_name):
                    # 如果节点可以匹配0个，所以没有匹配到也是正常的
                    break
                if len(pattern_out_node.inputs) != 0:
                    return []
                pattern_out_node = pattern_out_node.inputs[0]
            if pattern_out_node.op_name not in match_nodes:
                return []
            next_nodes = self.__get_next_nodes(match_nodes[pattern_out_node.op_name][-1])
            for next_node in next_nodes:
                if start_pattern_node.match(next_node, self._graph):
                    start_nodes.append(next_node)
        return start_nodes

    def get_match_map(self, node: NodeBase) -> MatchResult:
        """
        获取匹配的节点列表
        :param node: 子图遍历起始节点
        :return: 匹配结果
        """
        match_result = MatchResult(self._pattern)
        start_pattern_node = self._pattern.get_start_node()
        if not start_pattern_node.match(node, self._graph):
            return match_result

        start_nodes = [node]
        while self._pattern.can_match_again(match_result.count()):
            if len(start_nodes) == 0:
                break
            start_node = start_nodes.pop(0)
            match_nodes = {}
            # 子图遍历
            print('start node: {}'.format(start_node.name))
            if not self.__graph_bfs(start_node, start_pattern_node, match_nodes):
                continue
            match_result.add_node_dict(match_nodes)
            # 根据匹配的结果，查找下一个符合条件的起始节点
            new_start_nodes = self.__find_next_start_node(start_pattern_node, match_nodes)
            if len(new_start_nodes) != 0:
                start_nodes.extend(new_start_nodes)
        return match_result

