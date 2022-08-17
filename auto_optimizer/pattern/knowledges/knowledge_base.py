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
from pattern.pattern import Pattern
from pattern.pattern import VISIT_DIRECTION
from pattern.matcher import MatchResult
from pattern.matcher import Matcher
from magiconnx.interface import BaseGraph as GraphBase
from magiconnx.interface import BaseNode as NodeBase


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

    def _has_next_pattern(self) -> bool:
        if len(self._patterns) == 0:
            return False
        if self._pattern_idx == -1:
            return True
        if self._pattern_idx + 1 < len(self._patterns):
            return True
        return False

    def _next_pattern(self):
        if not self._has_next_pattern():
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

    def _has_next_apply(self):
        pattern = self.__get_current_pattern()
        apply_methods = self._pattern_apply_dict.get(pattern)
        if apply_methods is None or len(apply_methods) == 0:
            return False
        if self._apply_idx == -1:
            return True
        if self._apply_idx + 1 < len(apply_methods):
            return True
        return False

    def _next_apply(self):
        if not self._has_next_apply():
            return
        self._apply_idx += 1

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

    def get_candidate_sub_graphs(self, graph: GraphBase, top_ops_names: List[str] = None) -> List[MatchResult]:
        """
        匹配所有子图
        """
        all_match_result = []
        pattern = self.__get_current_pattern()
        if pattern is None:
            return []
        matcher = Matcher(graph, pattern)
        candidate_nodes = matcher.get_candidate_nodes()
        visit_direction = pattern.get_match_direction()
        if visit_direction == VISIT_DIRECTION.DOWN_UP:
            candidate_nodes.reverse() # 从下往上遍历，遍历结果排序取反
        for node in candidate_nodes:
            match_result = matcher.get_match_map(node)
            if match_result.is_empty():
                continue
            is_sub_collection = False
            for other_match_result in all_match_result:
                if other_match_result.include(match_result):
                    is_sub_collection = True
                    break
            if not is_sub_collection:
                all_match_result.append(match_result)
        return all_match_result

    def apply(self, graph: GraphBase, match_result: MatchResult) -> bool:
        """
        修改子图
        :param graph: 整图
        :param node: 算子节点
        :return: 成功返回True，失败返回False
        """
        apply_method = self.__get_current_apply_method()
        if apply_method is None:
            return False
        if isinstance(apply_method, types.MethodType):
            if len(match_result.node_dicts) == 0:
                return False
            return apply_method(graph, match_result)
        return False
