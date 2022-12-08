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

import operator as op

from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.interface.base_node import BaseNode, Node, Initializer
from auto_optimizer.pattern.pattern import MATCH_PATTERN, Pattern, MatchBase
from auto_optimizer.pattern.matcher import MatchResult
from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase


class EmptySlice(MatchBase):
    def match(self, node: BaseNode, graph: BaseGraph) -> bool:
        if not isinstance(node, Node) or len(node.inputs) < 3:
            return False
        starts = graph.get_node(node.inputs[1], node_type=Initializer)
        ends = graph.get_node(node.inputs[2], node_type=Initializer)
        if starts is None or ends is None:
            return False
        if len(starts.value) != len(ends.value):
            return False
        return any(s == e for s, e in zip(starts.value, ends.value))


r'''
if concat has more than 1 input after remove empty slice,
then it won't be removed

    |        |
   Node  EmptySlice            |
     \      /                 Node
      \    /                   |
      Concat         ====>     |
        |                   NextNode
        |
     NextNode

'''

# empty slice pattern
# we need this because ATC didn't support empty tensor in some cases
pattern_empty_slice = Pattern() \
    .add_node("Slice_0", ["Slice"], [EmptySlice()]) \
    .add_node("Concat_0", ["Concat"]) \
    .add_edge("Slice_0", "Concat_0") \
    .set_input("Slice_0") \
    .set_output("Concat_0") \
    .set_loop(MATCH_PATTERN.MATCH_ONCE)


@KnowledgeFactory.register()
class KnowledgeEmptySliceFix(KnowledgeBase):
    '''Remove slice operator which output empty tensor.'''
    def __init__(self):
        super().__init__()
        self._register_apply_funcs(pattern_empty_slice, [self._empty_slice_fix])

    def _disconnect_useless_concat(self, node: Node, graph: BaseGraph) -> None:
        if node.outputs[0] not in [out.name for out in graph.outputs]:
            # outputs of graph does not contain output of concat
            # then we just reconnect next_node and pre_node to skip concat
            for nnext in graph.get_next_nodes(node.outputs[0]):
                nnext.inputs = [
                    inp if inp != node.outputs[0] else node.inputs[0]
                    for inp in nnext.inputs
                ]
            return
        # outputs of graph contain output of concat
        # we can't change name of outputs of graph,
        # so we change output name of pre_node to output of graph
        pre_node = graph.get_prev_node(node.inputs[0])
        if pre_node is not None:
            idx_out = pre_node.get_output_id(node.inputs[0])
            pre_node.outputs[idx_out] = node.outputs[0]
        elif node.inputs[0] in [inp.name for inp in graph.inputs]:
            # in case concat connect inputs and outputs of a graph, although not likely
            # we add Unsqueeze/Squeeze combination to prevent change input/output
            graph.add_node(
                name=f'unsqueeze_{node.name}',
                op_type='Unsqueeze',
                inputs=[node.inputs[0]],
                outputs=[f'unsqueeze_{node.name}_output'],
                attrs={'axes': [0]}
            )
            graph.add_node(
                name=f'squeeze_{node.name}',
                op_type='Squeeze',
                inputs=[f'unsqueeze_{node.name}_output'],
                outputs=[node.outputs[0]],
                attrs={'axes': [0]}
            )

    def _empty_slice_fix(self, graph: BaseGraph, match_result: MatchResult) -> bool:
        flag = False
        for matchinfo in match_result.node_dicts:
            slice_empty = graph.get_node(matchinfo['Slice_0'][0].name, node_type=Node)
            concat = graph.get_node(matchinfo['Concat_0'][0].name, node_type=Node)
            if slice_empty is None or concat is None:
                continue
            output_name = slice_empty.outputs[0]
            for node in graph.get_next_nodes(output_name):
                if op.ne(node.op_type, 'Concat'):
                    # next node is not concat, do nothing
                    continue
                # remove the output of empty slice from it's inputs
                node.inputs = list(filter(lambda x: x != output_name, node.inputs))
                if len(node.inputs) != 1:
                    continue
                # number of concat's inputs equal to 1 after removal
                # we can remove this concat since it does nothing
                self._disconnect_useless_concat(node, graph)
                graph.remove(node.name, {})
            # should use graph.remove_unused_nodes instead, but this API have some issue unfixed
            graph.remove(slice_empty.name, {})
            graph.update_map()
            flag = True
        return flag
