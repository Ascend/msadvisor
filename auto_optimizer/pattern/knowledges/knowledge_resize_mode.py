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

from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory
from auto_optimizer.pattern.pattern import MATCH_PATTERN
from auto_optimizer.pattern.pattern import MatchBase
from auto_optimizer.pattern.pattern import Pattern
from auto_optimizer.pattern.matcher import MatchResult
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.interface.base_node import BaseNode
from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase


class ResizeModeOptimize:
    def __init__(self):
        self._mode_from = ['linear', 'cubic']
        self._mode_to   = 'nearest'

    @property
    def mode_from(self):
        return self._mode_from

    @property
    def mode_to(self):
        return self._mode_to


class ResizeOpMatch(MatchBase):
    def __init__(self):
        super().__init__()

    def match(self, node: BaseNode, graph: BaseGraph) -> bool:
        if node is None:
            return False
        if not node.op_type == 'Resize':
            return False
        if node['mode'] in ResizeModeOptimize().mode_from:
            return True
        return False


class ResizeOpPattern(Pattern):
    def __init__(self):
        super().__init__()
        self.add_node('resize_operator', ['Resize'], [ResizeOpMatch()]) \
            .set_input('resize_operator') \
            .set_output('resize_operator') \
            .set_node_loop('resize_operator', MATCH_PATTERN.MATCH_ONCE) \
            .set_loop(MATCH_PATTERN.MATCH_ONCE)


@KnowledgeFactory.register("KnowledgeResizeMode")
class KnowledgeResizeMode(KnowledgeBase):
    def __init__(self):
        super().__init__()
        self._register_apply_funcs(ResizeOpPattern(), self._resize_mode_apply)

    def _resize_mode_apply(self, graph: BaseGraph, match_result: MatchResult) -> bool:
        """ Resize 模型转换应用方法
        :param graph       : 整图
        :param match_result: 子图匹配结果
        :return            : 模式转换是否应用成功
        """
        mode = ResizeModeOptimize()
        for node_dict in match_result.node_dicts:
            for nodes in node_dict.values():
                for node in nodes:
                    if node.op_type != 'Resize':
                        continue
                    if node['mode'] not in mode.mode_from:
                        continue
                    node['mode'] = mode.mode_to
                    if mode.mode_to == 'nearest':
                        node['nearest_mode'] = 'round_prefer_floor'
        
        return True