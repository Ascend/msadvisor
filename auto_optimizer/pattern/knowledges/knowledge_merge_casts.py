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

from typing import Set
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from onnx.onnx_cpp2py_export.shape_inference import InferenceError

from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory
from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase
from auto_optimizer.pattern.pattern import MATCH_PATTERN
from auto_optimizer.pattern.pattern import Pattern
from auto_optimizer.pattern.matcher import MatchResult
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.interface.base_node import BaseNode


class MergeCastsPattern(Pattern):
    """ 可进行 Cast 算子合并的子图匹配模式
    可进行 Cast 算子合并的子图模式为以任意算子为根节点，其余节点全部为 Cast 算子的一颗树：

                           Add
                           / \
                          /   \
                        Cast Cast
                        /    /  \
                     Cast  Cast Cast

    Cast 算子合并一共可归纳为 3 种方法：
    1. 同属性的兄弟 Cast 算子合并

                Add                  Add   
                / \                   |    
               /   \                  |    
             Cast Cast     -->      Cast
              |     |               /  \ 
             Mul   Sub            Mul   Sub

    2. 单分支路径上的父子 Cast 算子合并

                Add
                 |                   Add
               Cast1       -->        |
                 |                  Cast2
               Cast2

    3. 根节点后的 Cast 算子如果与输出类型相同可以消除

        Add
         |  int32               Add
         |                -->    |  int32
        Cast (to: int32)         |
         |                      Mul
        Mul
    """

    def __init__(self):
        super().__init__()
        self.add_node('root_node', None) \
            .add_node('cast_node', 'Cast') \
            .set_input('root_node') \
            .set_output('cast_node') \
            .set_node_loop('root_node', MATCH_PATTERN.MATCH_ONCE) \
            .set_node_loop('cast_node', MATCH_PATTERN.MATCH_ONCE_OR_MORE) \
            .set_loop(MATCH_PATTERN.MATCH_ONCE)


@KnowledgeFactory.register()
class KnowledgeMergeCasts(KnowledgeBase):
    def __init__(self):
        super().__init__()
        self._register_apply_funcs(MergeCastsPattern(), [self._apply_method])

    def pre_process(self, graph: BaseGraph) -> bool:
        try:
            graph.infershape()
        except InferenceError as e:
            return False
        return True

    def _is_cast_node(self, node: BaseNode) -> bool:
        """ 判断节点是否为 Cast 节点
        :param node: 待判断节点
        """
        return node.op_type == 'Cast'

    def _apply_method(self, graph: BaseGraph, match_result: MatchResult) -> bool:
        """ Cast 节点合并应用方法
        :param graph       : 整图
        :param match_result: 子图匹配结果
        :return            : 类型转换是否应用成功
        """
        edge_type_dict = self._make_edge_type_dict(graph)
        for root_output in self._build_root_outputs(graph, match_result):
            for cast_node in filter(self._is_cast_node, graph.get_next_nodes(root_output)):
                self._merge_cast_tree(graph, cast_node, root_output, 10)
            # 递归结束后合并根节点下相同类型的兄弟 Cast 节点
            self._merge_brother_casts(graph, root_output)
            if root_output not in edge_type_dict:
                continue
            # 设法移除根节点下方的 Cast 节点
            output_type = NP_TYPE_TO_TENSOR_TYPE.get(edge_type_dict[root_output], 0)
            self._remove_cast_after_root(graph, root_output, output_type)
        return True

    def _make_edge_type_dict(self, graph: BaseGraph):
        """ 生成图边类型信息
        :param graph: 整图
        :return     : 图边信息字典
        """
        edge_type_dict = {}
        for edge in graph.value_infos:
            edge_type_dict[edge.name] = edge.dtype
        for input_node in graph.inputs:
            edge_type_dict[input_node.name] = input_node.dtype
        for output_node in graph.outputs:
            edge_type_dict[output_node.name] = output_node.dtype
        for initializer in graph.initializers:
            edge_type_dict[initializer.name] = initializer.value.dtype
        return edge_type_dict

    def _build_root_outputs(self, graph: BaseGraph, match_result: MatchResult) -> Set[str]:
        """ 构建根节点输出集合
        :param graph       : 整图
        :param match_result: 子图匹配结果
        :return            : 根节点输出集合
        """
        root_outputs = set()
        # 根节点主要由两部分组成：
        # 1. 匹配结果中的 root_node
        # 2. 整图输入
        for node_dict in match_result.node_dicts:
            if 'root_node' not in node_dict:
                continue
            for node in node_dict['root_node']:
                root_outputs.update(node.outputs)
        for graph_input in graph.inputs:
            root_outputs.add(graph_input.name)
        return root_outputs

    def _merge_brother_casts(self, graph: BaseGraph, node_output):
        """ 合并 node_output 后继的兄弟 Cast 节点
        :param graph      : 整图
        :param node_output: 进行后续节点搜索的输出
        """
        # node_output 后续的 Cast 节点中，每个类型的 Cast 节点都记录一个唯一的实例
        cast_node_map = {}
        for cast_node in filter(self._is_cast_node, graph.get_next_nodes(node_output)):
            to_type = cast_node['to']
            # cast_node_map 中已存在同类的 Cast 节点，则将两个 Cast 节点进行合并
            if to_type in cast_node_map:
                for next_node in graph.get_next_nodes(cast_node.outputs[0]):
                    input_index = next_node.inputs.index(cast_node._outputs[0])
                    next_node.inputs[input_index] = cast_node_map[to_type].outputs[0]
                graph.update_map()
                graph.remove(cast_node.name)
            else:
                cast_node_map[to_type] = cast_node

    def _transfer_cast_to_root(self, graph: BaseGraph, node_output, root_output):
        """ 将 node_output 后续的 Cast 节点迁移至 root_output
        :param graph      : 整图
        :param node_output: 进行后续节点搜索的输出
        :param root_output: 根节点输出
        """
        next_nodes = graph.get_next_nodes(node_output)
        if not next_nodes:
            return
        # 如果后续节点全部都是 Cast 节点则留下一个用于父子合并
        if all(map(self._is_cast_node, next_nodes)):
            nodes_to_transfer = next_nodes[1:]
        # 否则将所有 Cast 节点迁移至 root_output
        else:
            nodes_to_transfer = filter(self._is_cast_node, next_nodes)
        for node in nodes_to_transfer:
            node.inputs[0] = root_output
        graph.update_map()

    def _remove_parent_cast(self, graph: BaseGraph, node: BaseNode):
        """ 单分支路径上的父子 Cast 算子合并
        :param graph: 整图
        :param node : Cast 节点
        """
        next_nodes = graph.get_next_nodes(node.outputs[0])
        if len(next_nodes) == 1 and next_nodes[0].op_type == 'Cast':
            graph.remove(node.name)

    def _remove_cast_after_root(self, graph: BaseGraph, root_output, output_type):
        """ 根节点后的 Cast 算子如果与输出类型相同则进行消除
        :param graph      : 整图
        :param root_output: 根节点输出
        :param output_type: 输出类型
        """
        for next_node in graph.get_next_nodes(root_output):
            if next_node.op_type == 'Cast' and next_node['to'] == output_type:
                graph.remove(next_node.name)

    def _merge_cast_tree(self, graph: BaseGraph, cast_node: BaseNode, root_output, max_recursion):
        """ Cast 树型合并递归方法
        :param graph        : 整图
        :param cast_node    : cast 节点
        :param root_output  : 根节点输出
        :param max_recursion: 递归深度控制
        """
        if max_recursion <= 0:
            return
        for node in filter(self._is_cast_node, graph.get_next_nodes(cast_node.outputs[0])):
            self._merge_cast_tree(graph, node, root_output, max_recursion - 1)
        self._merge_brother_casts(graph, cast_node.outputs[0])
        self._transfer_cast_to_root(graph, cast_node.outputs[0], root_output)
        self._remove_parent_cast(graph, cast_node)
