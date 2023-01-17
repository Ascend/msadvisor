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
from typing import Dict, Optional, List, Callable
from collections import defaultdict

from auto_optimizer.pattern.pattern import Pattern
from auto_optimizer.pattern.matcher import MatchResult
from auto_optimizer.pattern.matcher import Matcher
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph

ApplyFunc = Callable[[BaseGraph, MatchResult], bool]
ApplyFuncs = List[ApplyFunc]


class UnionFind(object):
    def __init__(self, size: int = 0) -> None:
        self.uf: List[int] = [i for i in range(size)]

    def find(self, cur: int):
        if self.uf[cur] != cur:
            return self.find(self.uf[cur])
        return cur

    def union(self, pos0: int, pos1: int) -> None:
        u0 = self.find(pos0)
        u1 = self.find(pos1)
        if u0 < u1:
            self.uf[u1] = u0
        elif u0 > u1:
            self.uf[u0] = u1


class KnowledgeBase(object):
    def __init__(self) -> None:
        self._pattern_apply_dict: Dict[Pattern, ApplyFuncs] = defaultdict(list)
        self.reset()

    def reset(self) -> None:
        self._pattern_idx = -1
        self._apply_idx = -1

    @property
    def _patterns(self) -> List[Pattern]:
        return [*self._pattern_apply_dict]

    def _register_apply_funcs(self, pattern: Pattern, apply_funcs: ApplyFuncs) -> bool:
        '''
        注册pattern的apply方法
        '''
        if not isinstance(pattern, Pattern) or \
                not isinstance(apply_funcs, List):
            return False
        if not all(callable(func) for func in apply_funcs):
            return False
        self._pattern_apply_dict[pattern].extend(apply_funcs)
        return True

    def __get_current_pattern(self) -> Optional[Pattern]:
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

    def next_pattern(self) -> Optional[Pattern]:
        if not self.has_next_pattern():
            return None
        self._pattern_idx += 1
        self._apply_idx = -1
        return self._patterns[self._pattern_idx]

    def __get_current_apply_method(self) -> Optional[ApplyFunc]:
        pattern = self.__get_current_pattern()
        if pattern is None:
            return None
        apply_methods = self._pattern_apply_dict[pattern]
        if len(apply_methods) == 0:
            return None
        if self._apply_idx == -1:
            return apply_methods[0]
        if self._apply_idx < len(apply_methods):
            return apply_methods[self._apply_idx]
        return None

    def has_next_apply(self) -> bool:
        pattern = self.__get_current_pattern()
        if pattern is None:
            return False
        apply_methods = self._pattern_apply_dict.get(pattern)
        if apply_methods is None or len(apply_methods) == 0:
            return False
        if self._apply_idx == -1:
            return True
        if self._apply_idx + 1 < len(apply_methods):
            return True
        return False

    def next_apply(self) -> None:
        if not self.has_next_apply():
            return
        self._apply_idx += 1

    def get_apply_ids(self) -> List[int]:
        """
        返回当前pattern对应的所有apply_id
        """
        pattern = self.__get_current_pattern()
        if pattern is None:
            return []
        apply_methods = self._pattern_apply_dict.get(pattern)
        if apply_methods is None:
            return []
        return [i for i, _ in enumerate(apply_methods)]

    def set_apply_id(self, apply_id) -> bool:
        """
        基于当前pattern，设置apply_id
        """
        pattern = self.__get_current_pattern()
        if pattern is None:
            return False
        apply_methods = self._pattern_apply_dict.get(pattern)
        if apply_methods is None:
            return False
        if apply_id >= len(apply_methods) or apply_id < 0:
            return False
        self._apply_idx = apply_id
        return True

    @abstractmethod
    def pre_process(self, graph: BaseGraph) -> bool:
        """
        整图预处理方法
        :param graph: 预处理的整图
        """
        return True

    @abstractmethod
    def post_process(self, graph: BaseGraph) -> bool:
        """
        整图后处理方法
        :param graph: 后处理的整图
        """
        return True

    def match_pattern(self, graph: BaseGraph, top_ops_names: Optional[List[str]] = None) -> List[MatchResult]:
        """
        匹配所有子图
        :param graph: 计算图
        :param top_ops_names: topn算子列表，性能差的算子
        :return: 成功返回匹配的结果，失败返回空数组
        """
        if graph is None:
            return []
        pattern = self.__get_current_pattern()
        if pattern is None:
            return []
        matcher = Matcher(graph, pattern)
        candidate_nodes = matcher.get_candidate_nodes()

        match_results: List[MatchResult] = []
        for node in candidate_nodes:
            match_result = matcher.get_match_map(node)
            if match_result.is_empty():
                continue
            match_results.append(match_result)
        if len(match_results) == 0:
            return []

        if not pattern.can_match_more():
            return match_results

        uf = UnionFind(len(match_results))
        for i, match_result in enumerate(match_results):
            for j, other_match_result in enumerate(match_results[i + 1:]):
                j += i + 1
                if uf.find(i) == uf.find(j):
                    continue
                if match_result.connected(other_match_result):
                    uf.union(i, j)
        for i, match_result in enumerate(match_results):
            pi = uf.find(i)
            if pi != i:
                match_results[pi].merge(match_result)
        return [copy.deepcopy(match_result)
            for i, match_result in enumerate(match_results)
                if uf.find(i) == i]

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
        if callable(apply_method):
            if len(match_result.node_dicts) == 0:
                return False
            return apply_method(graph, match_result)
        return False
