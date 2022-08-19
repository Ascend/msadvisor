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
from abc import abstractmethod
from enum import Enum
from typing import List, Dict
from magiconnx.interface import BaseGraph as GraphBase
from magiconnx.interface import BaseNode as NodeBase


class MATCH_PATTERN(Enum):
    # 不循环，只匹配一次
    MATCH_ONECE = 1
    # 循环无上限，匹配一次或者多次
    MATCH_ONECE_OR_MORE = -1
    # 循环无上限，匹配零次或者多次
    MATCH_ZERO_OR_MORE = -2
    # 循环有上限，匹配一次或者最多固定次数
    MATCH_ONECE_OR_LIMITED_TIME = -3
    # 循环有上限，匹配零次或者最多固定次数
    MATCH_ZERO_OR_LIMITED_TIME = -4


# 子图遍历方向
class VISIT_DIRECTION(Enum):
    # 从上往下遍历
    UP_DOWN = 1
    # 从下往上遍历
    DOWN_UP = 2
    # 方向不明确
    NO_DIRECTION = 3


class MatchBase(object):
    def __init__(self):
        pass

    @abstractmethod
    def match(self, node: NodeBase, graph: GraphBase) -> bool:
        return False


class PatternNode(object):
    def __init__(self, op_name: str, op_types: List[str], op_matchs: List[MatchBase] = None):
        self.op_name = op_name
        self.op_types = op_types
        self.op_matchs = op_matchs
        self.inputs = []
        self.outputs = []

    def match(self, node: NodeBase, graph: GraphBase) -> bool:
        if node is None:
            return False
        if self.op_types.count(node.op_type) == 0:
            return False
        if self.op_matchs is None:
            return True
        for match_class in self.op_matchs:
            if not match_class().match(node, graph):
                return False
        return True

    def set_input(self, inputs: List[str] or str):
        self.inputs.append(inputs)

    def set_output(self, output: List[str] or str):
        self.outputs.append(output)


class Pattern(object):
    def __init__(self):
        self.node_dict = {} # Dict[str, PatternNode]
        self.in_nodes = [] # PatternNode
        self.out_nodes = [] # PatternNode
        self.node_match_pattern_dict = {}
        self.graph_match_pattern = MATCH_PATTERN.MATCH_ONECE
        self.graph_loop = -1

    def add_node(self, op_name: str, op_types: List[str], op_match: List[MatchBase] = None):
        """
        创建PatternNode，并增加到节点列表
        :param op_name: 算子节点名
        :param op_types: 支持的算子类型列表
        :param op_match: 算子match列表
        :return: 返回Builder实例
        """
        if self.node_dict.get(op_name) is not None:
            raise RuntimeError('Operator({}) has bean existed.'.format(op_name))
        self.node_dict[op_name] = PatternNode(op_name, op_types, op_match)
        return self

    def add_edge(self, prev_op_name: str, next_op_name: str):
        prev_node = self.node_dict.get(prev_op_name)
        next_node = self.node_dict.get(next_op_name)
        if prev_node is None:
            raise RuntimeError('Operator({}) has not bean added.'.format(prev_op_name))
        if next_node is None:
            raise RuntimeError('Operator({}) has not bean added.'.foramt(next_op_name))
        next_node.set_input(prev_node)
        prev_node.set_output(next_node)
        return self

    def set_input(self, op_name: str):
        """
        设置子图的输入节点
        :param op_name: 算子节点名
        :return: 返回Builder实例
        """
        in_node = self.node_dict.get(op_name)
        if in_node is None:
            raise RuntimeError('Operator({}) has not bean added.'.format(op_name))
        if len(in_node.inputs) > 0:
            raise RuntimeError('Operator({}) is not output node.'.format(op_name))
        self.in_nodes.append(in_node)
        return self

    def set_output(self, op_name: str):
        """
        设置子图的输出节点
        :param op_name: 输出节点名
        :return: 返回Builder实例
        """
        out_node = self.node_dict.get(op_name)
        if out_node is None:
            raise RuntimeError('Operator({}) has not bean added.'.format(op_name))
        if len(out_node.outputs) > 0:
            raise RuntimeError('Operator({}) is not output node.'.format(op_name))
        self.out_nodes.append(out_node)
        return self

    def set_node_loop(self, op_name: str, match_pattern: MATCH_PATTERN, loop: int = -1):
        """
        设置算子节点匹配模式
        :param op_name: 算子节点名称
        :param match_pattern: 匹配模式
        :param loop: 针对需要固定匹配次数的场景下，需要明确匹配次数
        :return: 返回Builder实例
        """
        node = self.node_dict.get(op_name)
        if node is None:
            raise RuntimeError('Operator({}) has not bean added.'.format(op_name))
        self.node_match_pattern_dict[op_name] = match_pattern
        return self

    def set_loop(self, match_pattern: MATCH_PATTERN, loop: int = -1):
        """
        设置子图循环次数，默认不限循环次数
        单输入多输出、多输入单输出不支持子图循环匹配
        :param match_pattern: 子图匹配模式
        :param loop: 针对需要固定匹配次数的场景下，需要明确匹配次数
        :return: 返回Builder实例
        """
        self.graph_match_pattern = match_pattern
        if match_pattern == MATCH_PATTERN.MATCH_ONECE_OR_MORE:
            if len(self.in_nodes) != len(self.out_nodes):
                raise RuntimeError('.')
        return self

    def get_match_direction(self):
        """
        1、多输入单输出，从下往上遍历
        2、单输出多输出，从上往下遍历
        3、单输入单输出，从上往下遍历
        4、多输入多输出，暂时不支持
        """
        if len(self.in_nodes) == 1:
            # 单输入，则从上往下遍历
            return VISIT_DIRECTION.UP_DOWN
        if len(self.out_nodes) == 1:
            # 单输出，从下往上遍历
            return VISIT_DIRECTION.DOWN_UP
        # 多输入多输出场景，没法确定遍历方向，暂时不支持
        return VISIT_DIRECTION.NO_DIRECTION

    def get_start_node(self):
        """
        获取起始节点
        """
        visit_direction = self.get_match_direction()
        if visit_direction == VISIT_DIRECTION.UP_DOWN:
            # 单输入
            return self.in_nodes[0]
        if visit_direction == VISIT_DIRECTION.DOWN_UP:
            # 单输出
            return self.out_nodes[0]
        # 多输入多输出不支持
        return None

    def node_cann_match_more(self, op_name) -> bool:
        if op_name not in self.node_match_pattern_dict:
            return False
        node_match_pattern = self.node_match_pattern_dict[op_name]
        return node_match_pattern == MATCH_PATTERN.MATCH_ONECE_OR_MORE or \
            node_match_pattern == MATCH_PATTERN.MATCH_ZERO_OR_MORE

    def node_cann_match_zero(self, op_name) -> bool:
        if op_name not in self.node_match_pattern_dict:
            return False
        node_match_pattern = self.node_match_pattern_dict[op_name]
        return node_match_pattern == MATCH_PATTERN.MATCH_ZERO_OR_MORE or \
            node_match_pattern == MATCH_PATTERN.MATCH_ZERO_OR_LIMITED_TIME

    def can_match_again(self, loop_num: int) -> bool:
        """
        判断是否可以继续循环
        """
        if self.graph_match_pattern == MATCH_PATTERN.MATCH_ONECE_OR_MORE:
            return True
        if self.graph_match_pattern == MATCH_ONECE:
            return loop_num == 0
        if self.graph_match_pattern == MATCH_ONECE_OR_LIMITED_TIME:
            if loop_num < self.loop:
                return True
        return False
