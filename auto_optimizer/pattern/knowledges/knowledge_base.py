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
from abc import abstractmethod
from typing import Dict, Tuple, List
from auto_optimizer.pattern.pattern import Pattern
from auto_optimizer.pattern.pattern import DIRECTION
from auto_optimizer.pattern.matcher import MatchResult
from auto_optimizer.pattern.matcher import Matcher
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph


class UnionFind(object):
    def __init__(self):
        self.uf = []

    def find(self, cur):
        if self.uf[cur] != cur:
            return self.find(self.uf[cur])
        return cur

    def union(self, pos0, pos1):
        u0 = self.find(pos0)
        u1 = self.find(pos1)
        if u0 != u1:
            self.uf[pos1] = u0

    def expand(self):
        last_index = len(self.uf)
        self.uf.append(last_index)


class KnowledgeBase(object):
    def __init__(self):
        self._patterns = self._build_patterns()
        self._pattern_apply_dict = self._build_pattern_apply_map()

        self._pattern_idx = -1
        self._apply_idx = -1

    def __get_current_pattern(self):
        if len(self._patterns) == 0:
            return None
        if self._pattern_idx == -1:
            return self._patterns[0]
        if self._pattern_idx < len(self._patterns):
            return self._patterns[self._pattern_idx]
        return None

    def has_next_pattern(self) -> bool:
        if len(self._patterns) == 0:
            return False
        if self._pattern_idx == -1:
            return True
        if self._pattern_idx + 1 < len(self._patterns):
            return True
        return False

    def next_pattern(self):
        if not self.has_next_pattern():
            return None
        self._pattern_idx += 1
        self._apply_idx = -1
        return self._patterns[self._pattern_idx]

    def __get_current_apply_method(self):
        pattern = self.__get_current_pattern()
        apply_methods = self._pattern_apply_dict[pattern]
        if len(apply_methods) == 0:
            return None
        if self._apply_idx == -1:
            return apply_methods[0]
        if self._apply_idx < len(apply_methods):
            return apply_methods[self._apply_idx]
        return None

    def has_next_apply(self):
        pattern = self.__get_current_pattern()
        apply_methods = self._pattern_apply_dict.get(pattern)
        if apply_methods is None or len(apply_methods) == 0:
            return False
        if self._apply_idx == -1:
            return True
        if self._apply_idx + 1 < len(apply_methods):
            return True
        return False

    def next_apply(self):
        if not self.has_next_apply():
            return
        self._apply_idx += 1

    def get_apply_ids(self) -> int:
        """
        返回当前pattern对应的所有apply_id
        """
        pattern = self.__get_current_pattern()
        apply_methods = self._pattern_apply_dict.get(pattern)
        if apply_methods is None:
            return []
        return [ i for i, _ in enumerate(apply_methods) ]

    def set_apply_id(self, apply_id) -> bool:
        """
        基于当前pattern，设置apply_id
        """
        pattern = self.__get_current_pattern()
        apply_methods = self._pattern_apply_dict.get(pattern)
        if apply_methods is None:
            return False
        return True if apply_id < len(apply_methods) and apply_id >= 0 else False

    @abstractmethod
    def __build_patterns(self) -> List[Pattern]:
        """
        知识库对应多个子图
        :return: 返回多个子图定义
        """
        return []

    @abstractmethod
    def __build_pattern_apply_map(self) -> Dict[Pattern, List]:
        """
        构建pattern和apply的映射关系
        :return: 返回pattern和apply方法的字典
        """
        return {}

    def __is_sub_graph_connection(self, match_result, cur_node) -> bool:
        """
        判断两个子图之间是否存在连接
        :param match_result: 子图匹配的结果
        :param cur_node: 当前子图的关键节点
        :return: 存在连接，则返回True，否则返回False
        """
        pattern = self.__get_current_pattern()
        direction = pattern.get_visit_direction()
        node_dict = match_result.node_dicts[0]
        for nodes in node_dict.values():
            if direction == DIRECTION.UP_DOWN:
                for node in nodes:
                    if node.outputs[0] in cur_node.inputs:
                        return True
            elif direction == DIRECTION.DOWN_UP:
                for node in nodes:
                    if cur_node.outputs[0] in node.inputs:
                        return True
            else:
                return False
        return False

    def get_candidate_sub_graphs(self, graph: BaseGraph, top_ops_names: List[str] = None) -> List[MatchResult]:
        """
        匹配所有子图
        :param graph: 计算图
        :param top_ops_names: topn算子列表，性能差的算子
        :return: 成功返回匹配的结果，失败返回空数组
        """
        if graph is None:
            return []
        pattern = self.__get_current_pattern()
        matcher = Matcher(graph, pattern)
        candidate_nodes = matcher.get_candidate_nodes()
        direction = pattern.get_visit_direction()
        if direction == DIRECTION.DOWN_UP:
            candidate_nodes.reverse() # 从下往上遍历，遍历结果排序取反

        uf = UnionFind()
        all_match_result = []
        for node in candidate_nodes:
            match_result = matcher.get_match_map(node)
            if match_result.is_empty():
                continue
            all_match_result.append(match_result)
            uf.expand()

            if not pattern.can_match_more():
                continue

            # 上下存在连接的子图，通过union-find建立关联
            for index, match_res in enumerate(all_match_result):
                if index == len(all_match_result) - 1:
                    continue
                if self.__is_sub_graph_connection(match_res, node):
                    uf.union(index, len(all_match_result) - 1)

        # 合并
        for index, match_result in enumerate(all_match_result):
            pindex = uf.find(index)
            if pindex == index:
                continue
            all_match_result[pindex].add_node_dict(match_result.node_dicts[0])
        # 拷贝结果
        result = []
        for index, match_result in enumerate(all_match_result):
            if uf.find(index) == index:
                result.append(copy.deepcopy(match_result))
        return result

    def apply(self, graph: BaseGraph, match_result: MatchResult) -> bool:
        """
        改图
        :param graph: 计算图
        :param match_result: 子图匹配的结果
        :return: 成功返回True，失败返回False
        """
        if graph is None or match_result is None:
            return False
        apply_method = self.__get_current_apply_method()
        if apply_method is None:
            return False
        if isinstance(apply_method, types.MethodType):
            if len(match_result.node_dicts) == 0:
                return False
            return apply_method(graph, match_result)
        return False
