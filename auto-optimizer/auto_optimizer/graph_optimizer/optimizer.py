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

from copy import deepcopy
from typing import Callable, Dict, List

from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase
from auto_optimizer import KnowledgeFactory
from auto_optimizer.pattern.knowledges import *


class GraphOptimizer:
    def __init__(self, knowledges: List[str]) -> None:
        registered_knowledges: Dict[str, KnowledgeBase] = KnowledgeFactory.get_knowledge_pool()
        for idx, name in enumerate(registered_knowledges):
            knowledges = [name if v == str(idx) else v for v in knowledges]
        knowledges = list(dict.fromkeys(knowledges))
        knowledge_dict = {
            name: registered_knowledges.get(name, KnowledgeBase())
            for name in knowledges
            if name in registered_knowledges
        }
        if len(knowledge_dict) == 0:
            raise ValueError('No valid knowledge provided.')
        self.knowledges: Dict[str, KnowledgeBase] = knowledge_dict

    def load_config(self):
        pass

    @staticmethod
    def evaluate(graph: BaseGraph, knowledge: KnowledgeBase) -> bool:
        graph_copy = deepcopy(graph)
        while knowledge.has_next_pattern():
            knowledge.next_pattern()
            match_results = knowledge.match_pattern(graph_copy)
            if match_results is None or len(match_results) == 0:
                continue
            while knowledge.has_next_apply():
                knowledge.next_apply()
                for match_result in match_results:
                    if knowledge.apply(graph_copy, match_result):
                        return True
        return False

    @staticmethod
    def optimize(graph: BaseGraph, knowledge: KnowledgeBase) -> bool:
        res = False
        while knowledge.has_next_pattern():
            knowledge.next_pattern()
            match_results = knowledge.match_pattern(graph)
            if match_results is None or len(match_results) == 0:
                continue
            while knowledge.has_next_apply():
                knowledge.next_apply()
                for match_result in match_results:
                    res |= knowledge.apply(graph, match_result)
        return res

    def _apply_action(
        self,
        graph: BaseGraph,
        action: Callable[[BaseGraph, KnowledgeBase], bool]
    ) -> List[str]:
        applied_knowledges = []
        for name, knowledge in self.knowledges.items():
            knowledge.reset()
            if not knowledge.pre_process(graph):
                continue
            if action(graph, knowledge):
                applied_knowledges.append(name)
            knowledge.post_process(graph)
        return applied_knowledges

    def evaluate_knowledges(self, graph: BaseGraph) -> List[str]:
        return self._apply_action(graph, GraphOptimizer.evaluate)

    def apply_knowledges(self, graph: BaseGraph) -> List[str]:
        return self._apply_action(graph, GraphOptimizer.optimize)


if __name__ == "__main__":
    knowledges = KnowledgeFactory.get_knowledge_pool()
    graph_opt = GraphOptimizer(list(knowledges.keys()))
