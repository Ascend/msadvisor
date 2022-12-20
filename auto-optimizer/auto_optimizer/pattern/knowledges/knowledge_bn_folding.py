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

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from auto_optimizer.graph_refactor.interface import (
    BaseGraph, Initializer, Node
)
from auto_optimizer.graph_refactor.interface.base_node import BaseNode
from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory
from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase
from auto_optimizer.pattern.matcher import MatchResult
from auto_optimizer.pattern.pattern import MATCH_PATTERN, MatchBase, Pattern
from auto_optimizer.pattern.utils import NextNodeCount

# when certain conditions are met, tr/bn/tr structure
# can be replaced by mul/add
r"""
        PreNode
           |                   PreNode
       Transpose0                 |
           |                     Mul
   BatchNormalization  =====>     |
           |                     Add
       Transpose1                 |
           |                  NextNode
       NextNode
"""


class ConstBNMatch(MatchBase):
    def match(self, node: BaseNode, graph: BaseGraph) -> bool:
        if not (isinstance(node, Node) and node.op_type == 'BatchNormalization'):
            return False
        if node.attrs.get('training_mode', 0) != 0:
            return False
        scale = graph.get_node(node.inputs[1], node_type=Initializer)
        bias = graph.get_node(node.inputs[2], node_type=Initializer)
        mean = graph.get_node(node.inputs[3], node_type=Initializer)
        var = graph.get_node(node.inputs[4], node_type=Initializer)
        if (scale is None or bias is None or mean is None or var is None):
            return False
        return True


@KnowledgeFactory.register()
class KnowledgeBNFolding(KnowledgeBase):
    '''BatchNormalization constants folding.'''
    def __init__(self):
        super().__init__()
        self.pattern_ = Pattern() \
            .add_node('tr0', ['Transpose'], [NextNodeCount(1)]) \
            .add_node('bn0', ['BatchNormalization'], [NextNodeCount(1), ConstBNMatch()]) \
            .add_node('tr1', ['Transpose']) \
            .add_edge('tr0', 'bn0') \
            .add_edge('bn0', 'tr1') \
            .set_loop(MATCH_PATTERN.MATCH_ONCE)
        self._register_apply_funcs(self.pattern_, [self._apply])

    def _constant_folding(self, scale: NDArray, bias: NDArray, mean: NDArray,
                          var: NDArray, epsilon: float,) -> Tuple[NDArray, NDArray]:
        common_divisor = np.sqrt(var + epsilon)
        mul_init = scale / common_divisor
        add_init = bias - scale * mean / common_divisor
        return mul_init, add_init

    def _bn_folding(self, tr0: Node, bn: Node, tr1: Node,
                    graph: BaseGraph) -> Tuple[Optional[NDArray], Optional[NDArray]]:
        scale = graph.get_node(bn.inputs[1], node_type=Initializer)
        bias = graph.get_node(bn.inputs[2], node_type=Initializer)
        mean = graph.get_node(bn.inputs[3], node_type=Initializer)
        var = graph.get_node(bn.inputs[4], node_type=Initializer)
        if (scale is None or bias is None or mean is None or var is None):
            return None, None
        epsilon = bn.attrs.get('epsilon', 1e-5)
        mul_init, add_init = self._constant_folding(
            scale=scale.value,
            bias=bias.value,
            mean=mean.value,
            var=var.value,
            epsilon=epsilon
        )
        # 计算出Mul/Add的系数后，还需要考虑Channel维度对齐和Transpose
        perm0: List[int] = tr0.attrs.get('perm', [])
        perm1: List[int] = tr1.attrs.get('perm', [])
        if not (perm0 and perm1 and len(perm0) == len(perm1)):
            return None, None
        # 前后的两个Transpose必须能抵消
        # 比如: [0, 1, 3, 2] + [0, 1, 3, 2] => [0, 1, 2, 3]
        if [perm0[idx] for idx in perm1] != list(range(len(perm0))):
            return None, None
        # 这些系数是形状为[C]的张量，需要和输入维度 [N, C, D1, D2, ..., Dn]
        # 对齐为 [1, C, 1, 1, ..., 1]
        shape_ = [1] * len(perm0)
        shape_[1] = mul_init.shape[0]
        mul_init = mul_init.reshape(shape_).transpose(perm1)
        add_init = add_init.reshape(shape_).transpose(perm1)
        return mul_init, add_init

    def _apply(self, graph: BaseGraph, match_res: MatchResult) -> bool:
        flag = False
        for dicts in match_res.node_dicts:
            bn = graph.get_node(dicts['bn0'][0].name, node_type=Node)
            tr0 = graph.get_node(dicts['tr0'][0].name, node_type=Node)
            tr1 = graph.get_node(dicts['tr1'][0].name, node_type=Node)
            if bn is None or tr0 is None or tr1 is None:
                continue
            mul_init, add_init = self._bn_folding(tr0, bn, tr1, graph)
            if mul_init is None or add_init is None:
                continue
            # 确定可以做优化，进行子图替换
            graph.add_initializer(
                name=f'{bn.name}_mul_init',
                value=mul_init,
            )
            graph.add_initializer(
                name=f'{bn.name}_add_init',
                value=add_init,
            )
            graph.add_node(
                name=f'{bn.name}_mul',
                op_type='Mul',
                inputs=[tr0.inputs[0], f'{bn.name}_mul_init'],
                outputs=[f'{bn.name}_mul_out'],
            )
            graph.add_node(
                name=f'{bn.name}_add',
                op_type='Add',
                inputs=[f'{bn.name}_mul_out', f'{bn.name}_add_init'],
                outputs=[tr1.outputs[0]],
            )
            graph.remove(bn.name, {})
            graph.remove(tr0.name, {})
            graph.remove(tr1.name, {})
            graph.update_map()
            flag = True
        return flag
