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

from typing import List, Dict, Union

from auto_optimizer.pattern.knowledges import KnowledgeConv1d2Conv2d, KnowledgeMergeContinueSlice

knowledge_map = {'KnowledgeConv1d2Conv2d': KnowledgeConv1d2Conv2d}

class GraphOptimizer:

    def __init__(self, knowledges: List[str] = None):
        self.knowldges = knowledges

    def load_config(self):
        pass

    @staticmethod
    def optimize(graph, knowledge):
        res = True
        while knowledge.has_next_pattern():
            knowledge.next_pattern()
            match_results = knowledge.get_candidate_sub_graphs(graph)
            if match_results is None or len(match_results) == 0:
                continue
            while knowledge.has_next_apply():
                knowledge.next_apply()
                for match_result in match_results:
                    res &= knowledge.apply(graph, match_result)
        return res

    def apply_knowledges(self, graph):
        for knowldge in self.knowldges:
            GraphOptimizer.optimize(graph, knowledge_map[knowldge]())

        return graph

if __name__ == "__main__":
    graph_opt = GraphOptimizer(['KnowledgeConv1d2Conv2d'])