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

from abc import abstractmethod
from enum import Enum, unique
from typing import Dict, List, Optional
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.interface.base_node import BaseNode


@unique
class MATCH_PATTERN(Enum):
    # 不循环，只匹配一次
    MATCH_ONCE = 1
    # 循环无上限，匹配一次或者多次
    MATCH_ONCE_OR_MORE = 2
    # 循环无上限，匹配零次或者多次
    MATCH_ZERO_OR_MORE = 3


class MatchBase(object):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def match(self, node: BaseNode, graph: BaseGraph) -> bool:
        return False


class PatternNode(object):
    def __init__(
        self,
        op_name: str,
        op_types: Optional[List[str]],
        op_matchs: Optional[List[MatchBase]] = None
    ) -> None:
        self._op_name: str = op_name
        self._op_types: List[str] = [] if op_types is None else op_types
        self._op_matchs: List[MatchBase] = [] if op_matchs is None else op_matchs
        self._inputs: List['PatternNode'] = []
        self._outputs: List['PatternNode'] = []
        self._match_pattern: MATCH_PATTERN = MATCH_PATTERN.MATCH_ONCE

    @property
    def op_name(self) -> str:
        return self._op_name

    @property
    def inputs(self) -> List:
        return self._inputs

    @property
    def outputs(self) -> List:
        return self._outputs

    def match(self, node: BaseNode, graph: BaseGraph) -> bool:
        """
        算子节点匹配
        :param node: 实际算子节点
        :param graph: 实际子图对象
        :return: 匹配成功返回True，失败返回False
        """
        if node is None:
            return False
        if self._op_types and self._op_types.count(node.op_type) == 0:
            return False
        if self._op_matchs is None:
            return True
        for op_match in self._op_matchs:
            if not op_match.match(node, graph):
                return False
        return True

    def add_parent(self, parent: 'PatternNode') -> None:
        self._inputs.append(parent)

    def add_child(self, child: 'PatternNode') -> None:
        self._outputs.append(child)

    def set_match_pattern(self, match_pattern: MATCH_PATTERN) -> None:
        if isinstance(match_pattern, MATCH_PATTERN):
            self._match_pattern = match_pattern

    def can_match_more_time(self) -> bool:
        return self._match_pattern == MATCH_PATTERN.MATCH_ONCE_OR_MORE or \
            self._match_pattern == MATCH_PATTERN.MATCH_ZERO_OR_MORE

    def can_match_zero_time(self) -> bool:
        return self._match_pattern == MATCH_PATTERN.MATCH_ZERO_OR_MORE


class Pattern(object):
    def __init__(self) -> None:
        self._nodes: Dict[str, PatternNode] = {}
        self._inputs: List[PatternNode] = []
        self._outputs: List[PatternNode] = []
        self._match_pattern = MATCH_PATTERN.MATCH_ONCE
        self._root: PatternNode = None

    @property
    def node_dict(self) -> Dict[str, PatternNode]:
        return self._nodes

    def add_node(
        self,
        op_name: str,
        op_types: Optional[List[str]],
        op_matchs: Optional[List[MatchBase]] = None
    ) -> 'Pattern':
        """
        创建PatternNode，并增加到节点列表
        :param op_name: 算子节点名
        :param op_types: 支持的算子类型列表
        :param op_matchs: 算子匹配规则列表
        :return: 返回实例
        """
        if self._nodes.get(op_name) is not None:
            raise RuntimeError(f'Operator({op_name}) already exists.')
        node = PatternNode(op_name, op_types, op_matchs)
        self._nodes[op_name] = node
        self._inputs.append(node)
        self._outputs.append(node)
        return self

    def add_edge(self, op_name: str, next_op_name: str) -> 'Pattern':
        cur_node = self._nodes.get(op_name)
        if cur_node is None:
            raise RuntimeError(f'Operator({op_name}) not exists.')
        next_node = self._nodes.get(next_op_name)
        if next_node is None:
            raise RuntimeError(f'Operator({next_op_name}) not exists.')
        next_node.add_parent(cur_node)
        cur_node.add_child(next_node)
        # update graph inputs and outputs
        if next_node in self._inputs:
            self._inputs.remove(next_node)
        if cur_node in self._outputs:
            self._outputs.remove(cur_node)
        return self

    def set_node_loop(
        self,
        op_name: str,
        match_pattern: MATCH_PATTERN
    ) -> 'Pattern':
        """
        设置算子节点匹配模式
        :param op_name: 算子节点名称
        :param match_pattern: 匹配模式
        :return: 返回实例
        """
        node = self._nodes.get(op_name)
        if node is None:
            raise RuntimeError(f'Operator({op_name}) not exists.')
        if isinstance(match_pattern, MATCH_PATTERN):
            node.set_match_pattern(match_pattern)
        return self

    def set_loop(self, match_pattern: MATCH_PATTERN) -> 'Pattern':
        """
        设置子图循环模式
        单输入多输出、多输入单输出、多输入多输出不支持子图循环匹配
        :param match_pattern: 子图匹配模式
        :return: 返回实例
        """
        if isinstance(match_pattern, MATCH_PATTERN):
            self._match_pattern = match_pattern
        return self

    def get_start_node(self) -> Optional[PatternNode]:
        """
        获取子图匹配遍历的起始节点
        :return: 返回子图遍历的起始节点
        """
        if len(self._inputs) == 0:
            raise RuntimeError('Graph is invalid, no input node.')
        if len(self._inputs) == 1:
            return self._inputs[0]
        if len(self._outputs) == 1:
            return self._outputs[0]
        if self._root is not None:
            return self._root
        for node in self._nodes.values():
            if len(node.inputs) <= 1:
                continue
            visited: Set[PatternNode] = set([node])
            queue: List[PatternNode] = [node]
            while len(queue) != 0:
                node_ = queue.pop(0)
                queue.extend(node_.inputs)
                visited.add(node_)
            queue.clear()
            queue.append(node)
            while len(queue) != 0:
                node_ = queue.pop(0)
                queue.extend(node_.outputs)
                visited.add(node_)
            if len(visited) == len(self._nodes):
                self._root = node
                return self._root
        raise RuntimeError('Get root node failed from graph.')

    def can_match_more(self) -> bool:
        """
        判断子图是否可以循环匹配
        :return: 能匹配则返回True，否则返回False
        """
        return self._match_pattern == MATCH_PATTERN.MATCH_ONCE_OR_MORE or \
            self._match_pattern == MATCH_PATTERN.MATCH_ZERO_OR_MORE
