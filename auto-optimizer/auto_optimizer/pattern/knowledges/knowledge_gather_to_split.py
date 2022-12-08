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

from typing import List, Dict
import logging
import operator as op

import numpy as np
import onnx

from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.interface.base_node import BaseNode, Initializer, Node, PlaceHolder
from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase
from auto_optimizer.pattern.utils import NextNodeCount
from auto_optimizer.pattern.pattern import MATCH_PATTERN
from auto_optimizer.pattern.pattern import MatchBase
from auto_optimizer.pattern.pattern import Pattern
from auto_optimizer.pattern.matcher import MatchResult


r"""
          PreNode                                      PreNode
          /  |  \                                         |            
         /   |   \                                     Split _0          
        /    |    \                                   /   |    \  
       /     |     \                                 /    |     \ 
      /      |      \            ==========>        /     |      \              
     /       |       \                             /      |       \ 
    /        |        \                          Selu_1  Conv_0   Conv_0 
Gather_0   Gather_1   Gather_2
    |        |         |        
  Selu_1   Conv_0   Conv_0                 


"""
pattern = Pattern() \
    .add_node("PreNode", ["Transpose"]) \
    .add_node("Gather_0", ["Gather"], [NextNodeCount(1)]) \
    .add_node("Gather_1", ["Gather"], [NextNodeCount(1)]) \
    .add_node("Gather_2", ["Gather"], [NextNodeCount(1)]) \
    .add_edge("PreNode", "Gather_0") \
    .add_edge("PreNode", "Gather_1") \
    .add_edge("PreNode", "Gather_2") \
    .set_input("PreNode") \
    .set_output("Gather_0") \
    .set_output("Gather_1") \
    .set_output("Gather_2") \
    .set_loop(MATCH_PATTERN.MATCH_ONCE)
    
r"""
         PreNode                                      PreNode
          /  |                                           |
         /   |                                        Split _0
        /    |                                       /   |    
       /     |                                      /    |     
      /      |                 ==========>         /     |      
     /       |                                    /      |       
    /        |                                  Selu_1  Conv_0   
Gather_0   Gather_1 
    |        |         
  Selu_1   Conv_0   


"""
pattern1 = Pattern() \
    .add_node("PreNode", ["Transpose"]) \
    .add_node("Gather_0", ["Gather"], [NextNodeCount(1)]) \
    .add_node("Gather_1", ["Gather"], [NextNodeCount(1)]) \
    .add_edge("PreNode", "Gather_0") \
    .add_edge("PreNode", "Gather_1") \
    .set_input("PreNode") \
    .set_output("Gather_0") \
    .set_output("Gather_1") \
    .set_loop(MATCH_PATTERN.MATCH_ONCE)


@KnowledgeFactory.register()
class KnowledgeGatherToSplit(KnowledgeBase):

    def __init__(self):
        super().__init__()
        # 注册pattern的apply方法
        self._register_apply_funcs(pattern, [self._split_pattern_apply])
        self._register_apply_funcs(pattern1, [self._split_pattern_apply])
        
    def _get_gather_nodes_indices(self, nodes: List[Node], graph: BaseGraph) -> List[int]:
        """
        获取Gather算子取的全部下标
        :param nodes: gather算子列表
        :param graph: 完整的图结构
        :return: 返回按gather节点顺序排列的下标列表
        """
        indices = []
        for n in nodes:
            indice = graph.get_node(n.inputs[1], node_type=Initializer)
            if indice is None or indice.value.size > 1:
                return []
            indices.append(int(indice.value))
        return indices

    def _split_match_apply(self, graph: BaseGraph, matchinfo: Dict[str, List[Node]]) -> bool:
        # make sure nodes of matching subgraph still exist in case some previous apply functions modified graph
        if any(graph.get_node(node.name, node_type=Node) is None for nodes in matchinfo.values() for node in nodes):
            logging.info("Some matching node have been removed or renamed, failed to optimizd.")
            return False
        preNode = graph.get_node(matchinfo['PreNode'][0].name, node_type=Node)
        gather_to_remove = [
            graph.get_node(v[0].name, node_type=Node) for k, v in matchinfo.items() if k != 'PreNode'
        ]

        indices = self._get_gather_nodes_indices(gather_to_remove, graph)
        indices = sorted(indices)

        for i in range(len(indices)-1):
            if (indices[i] + 1 < indices[i + 1]):
                return False
        output_value = []
        axis=0
        for node in gather_to_remove:
            print(node.name)
            output_value.append(node.outputs)
            axis=node.attrs.get('axis', None)
            graph.remove(node.name)
        split_value = []
        for i in range(len(indices)):
            if (indices[i] == 0):
                split_value.append(1)
            else :
                split_value.append(indices[i] - indices[i - 1])
        # add split node after PreNode
        split = graph.add_node(
            name=f'Split_{preNode.name}_pattern',
            op_type='Split',
            inputs=[matchinfo['PreNode'][0].outputs],
            outputs=output_value,
            attrs={
                'axis': axis,
                'split': split_value
            }
        )
        graph.insert_node(preNode.name, split, 0, 'after')
        graph.update_map()
        return True

    def _split_pattern_apply(self, graph: BaseGraph, match_result: MatchResult) -> bool:
        flag = False
        for matchinfo in match_result.node_dicts:
            if matchinfo:
                flag |= self._split_match_apply(graph, matchinfo)

        return flag

