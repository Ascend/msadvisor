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
import numpy as np
from typing import List, Dict
from knowledge_base import KnowledgeBase
from pattern.knowledge_factory import KnowledgeFactory
from magiconnx.interface import BaseGraph as GraphBase
from magiconnx.interface import BaseNode as NodeBase
from pattern.pattern import MATCH_PATTERN
from pattern.pattern import MatchBase
from pattern.pattern import Pattern
from pattern.matcher import MatchResult


class Conv1dMatch(MatchBase):
    def __init__(self):
        super().__init__()

    def match(self, node: NodeBase, graph: GraphBase) -> bool:
        if node is None:
            return False
        if not op.eq(node.op_type, 'Conv'):
            return False
        if len(node.inputs) > 1:
            weight = graph[node.inputs[1]]
            return len(weight.value.shape) == 3
        return False


# element_wise允许出现0次，或者多次
pattern = Pattern() \
    .add_node('Conv', ['Conv'], [Conv1dMatch]) \
    .add_node('element_wise', ['Mul', 'Add', 'Sub', 'Div', 'BatchNormalization', 'LeakyRelu', 'Relu']) \
    .add_edge('Conv', 'element_wise') \
    .set_input('Conv') \
    .set_output('element_wise') \
    .set_node_loop('element_wise', MATCH_PATTERN.MATCH_ZERO_OR_MORE) \
    .set_loop(MATCH_PATTERN.MATCH_ONECE_OR_MORE)

class KnowledgeConv1d2Conv2d(KnowledgeBase):
    def __init__(self):
        super().__init__()

    def _build_patterns(self) -> List[Pattern]:
        """
        知识库对应多个子图
        :return: 返回多个子图定义
        """
        return [pattern]

    def _build_pattern_apply_map(self) -> Dict[Pattern, List]:
        """
        构建pattern和apply的映射关系
        :return: 返回pattern和apply方法的字典
        """
        apply_dict = {
            pattern: [self._conv1d2conv2d_apply]
        }
        return apply_dict

    def _expand_conv_input_dims(self, graph: GraphBase, conv: NodeBase, mode: str) -> bool:
        """
        通过增加Unsqueeze算子，将conv1d的输入从3维扩展到4维
        :param graph: 整图
        :param conv: 卷积算子，在该算子前面插入Unsqueeze算子
        :return: True： 操作成功； False： 操作失败
        """
        us = graph.add_node('Unsqz_%d' % id, 'Unsqueeze', {'axes': [2]})
        graph.insert_node(conv, us, mode=mode)
        return True

    def _conv1d_to_conv2d(self, graph: GraphBase, conv: NodeBase) -> bool:
        """
        将conv1d转换成conv2d，修改conv1d属性和W
        :param graph: 整图
        :param conv: 卷积算子
        :return: True：操作成功；False：操作失败
        """
        attrs = ('dilations', 'kernel_shape', 'strides')
        for attr in attrs:
            if attr in conv.attrs.keys():
                val = conv[attr][0]
                conv[attr] = [1, val]

        if 'pads' in conv.attrs.keys():
            pds = conv['pads'][0]
            conv['pads'] = [0, pds, 0, pds]

        conv_w = graph[conv.inputs[1]].value
        conv_w = np.expand_dims(conv_w, axis=-2)
        graph[conv.inputs[1]].value = conv_w
        return True

    def _reduce_output_dims(self, graph: GraphBase, node: NodeBase, mode: str) -> bool:
        """
        降低维度
        :param graph: 整图
        :param node:
        :return: True：操作成功；False：操作失败
        """
        sq = graph.add_node('Sqz_%d' % id, 'Squeeze', {'axes': [2]})
        graph.insert_node(node, sq, mode=mode)
        return True

    def __get_next_nodes(self, graph: GraphBase, cur_node: NodeBase):
        """
        根据节点输出，获取所有该节点的后置节点
        """
        next_nodes = set()
        for node in graph._all_ops_map.values():
            if len(node.inputs) == 0:
                continue
            for next_input_name in node.inputs:
                for output_name in cur_node.outputs:
                    if op.eq(next_input_name, output_name):
                        next_nodes.add(node)
                        break
        return list(next_nodes)

    def _conv1d2conv2d_apply(self, graph: GraphBase, match_result: MatchResult) -> bool:
        input_node = match_result.node_dicts[0].get('Conv')[0]
        # print('insert node before {}'.format(input_node.name))
        self.expand_conv_input_dims(graph, input_node, 'before')

        for node_dict in match_result.node_dicts:
            conv1d = node_dict.get('Conv')[0]
            # print('modify conv1d {}.'.format(conv1d.name))
            self._conv1d_to_conv2d(graph, conv1d)

        # 考虑分叉的场景
        for node_dict in match_result.node_dicts:
            conv = node_dict.get('Conv')[0]
            element_wises = node_dict.get('element_wise')

            conv_pattern = pattern.node_dict.get('Conv')
            element_wise_pattern = pattern.node_dict.get('element_wise')

            conv_next_nodes = self.__get_next_nodes(graph, conv)
            for next_node in conv_next_nodes:
                if not conv_pattern.match(next_node, graph) and \
                    not element_wise_pattern.match(next_node, graph):
                    # print('insert Squeeze before {}'.format(next_node.name))
                    self._reduce_output_dims(graph, next_node, 'before')

            if element_wises is not None:
                for element_wise in element_wises:
                    next_nodes = self.__get_next_nodes(graph, element_wise)
                    for next_node in next_nodes:
                        if not conv_pattern.match(next_node, graph) and \
                            not element_wise_pattern.match(next_node, graph):
                            # print('insert Squeeze before {}'.format(next_node.name))
                            self._reduce_output_dims(graph, next_node, 'before')
        return True

KnowledgeFactory.add_knowledge('KnowledgeConv1d2Conv2d', KnowledgeConv1d2Conv2d())


def evaluate(data_path, parameter = None):
    if parameter is None:
        return False
    onnx_file = parameter.get('file_name')
    onnx_path = data_path + '/' + onnx_file
    print('onnx path: {}'.format(onnx_path))
    from magiconnx import OnnxGraph
    graph = OnnxGraph(onnx_path)
    conv1d2conv2d = KnowledgeConv1d2Conv2d()
    while conv1d2conv2d._has_next_pattern():
        conv1d2conv2d._next_pattern()
        # 遍历计算图，找出所有能匹配的子图
        match_results = conv1d2conv2d.get_candidate_sub_graphs(graph)
        if match_results is None or len(match_results) == 0:
            continue
        # 尝试不同的修改方法
        while conv1d2conv2d._has_next_apply():
            conv1d2conv2d._next_apply()
            for match_result in match_results:
                res = conv1d2conv2d.apply(graph, match_result)
    return True

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1]
    onnx_file = sys.argv[2]
    parameter = {'file_name': onnx_file}
    ret = evaluate(data_path, parameter)

