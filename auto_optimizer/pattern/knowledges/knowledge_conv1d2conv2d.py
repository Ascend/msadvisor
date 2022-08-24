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
import operator as op
import numpy as np
import onnx
import os

from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory
from auto_optimizer.pattern.pattern import MATCH_PATTERN
from auto_optimizer.pattern.pattern import MatchBase
from auto_optimizer.pattern.pattern import Pattern
from auto_optimizer.pattern.matcher import MatchResult
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.interface.base_node import BaseNode
from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase
from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph

class Conv1dMatch(MatchBase):
    def __init__(self):
        super().__init__()

    def match(self, node: BaseNode, graph: BaseGraph) -> bool:
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
        self._insert_op_names = set()

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

    def __is_lower_onnx_version(self) -> bool:
        """
        判断当前onnx版本是否小于1.12.0
        :return: onnx版本小于1.12.0，则返回True，否则返回False
        """
        limit_version = '1.12.0'
        onnx_version = onnx.version.version

        limit_versions = limit_version.split('.')
        for i, v in enumerate(onnx_version.split('.')):
            if i == len(limit_versions):
                return False
            if v < limit_versions[i]:
                return True
        return False

    def _expand_conv_input_dims(self, graph, conv, mode: str) -> bool:
        """
        通过增加Unsqueeze算子，将conv1d的输入从3维扩展到4维
        :param graph: 整图
        :param conv: 卷积算子，在该算子前面插入Unsqueeze算子
        :return: True： 操作成功； False： 操作失败
        """
        op_name = 'Unsqueeze_%s_%s' % (mode, conv.name)
        if op_name in self._insert_op_names:
            return True
        self._insert_op_names.add(op_name)
        if self.__is_lower_onnx_version():
            us = graph.add_node(op_name, 'Unsqueeze', {'axes': np.array([2], dtype=np.int64)})
            graph.insert_node(conv.name, us, mode=mode)
        else:
            us = graph.add_node(op_name, 'Unsqueeze')
            axes_name = '%s_axes' % op_name
            graph.add_initializer(axes_name, np.array([2], dtype=np.int64))
            graph.insert_node(conv.name, us, mode=mode)
            us.inputs.append(axes_name)
        graph.update_map()
        return True

    def _conv1d_to_conv2d(self, graph, conv) -> bool:
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

    def _reduce_output_dims(self, graph, node, mode: str) -> bool:
        """
        降低维度
        :param graph: 整图
        :param node:
        :return: True：操作成功；False：操作失败
        """
        op_name = 'Squeeze_%s_%s' % (mode, node.name)
        if op_name in self._insert_op_names:
            return True
        self._insert_op_names.add(op_name)
        if self.__is_lower_onnx_version():
            sq = graph.add_node(op_name, 'Squeeze', {'axes': np.array([2], dtype=np.int64)})
            graph.insert_node(node.name, sq, mode=mode)
        else:
            sq = graph.add_node(op_name, 'Squeeze')
            axes_name = '%s_axes' % op_name
            graph.add_initializer(axes_name, np.array([2], dtype=np.int64))
            graph.insert_node(node.name, sq, mode=mode)
            sq.inputs.append(axes_name)
        graph.update_map()
        return True

    def _conv1d2conv2d_apply(self, graph, match_result: MatchResult) -> bool:
        input_node = match_result.node_dicts[0].get('Conv')[0]
        self._expand_conv_input_dims(graph, input_node, 'before')

        # 考虑分叉的场景
        for node_dict in match_result.node_dicts:
            conv = node_dict.get('Conv')[0]
            element_wises = node_dict.get('element_wise')

            conv_pattern = pattern.node_dict.get('Conv')
            element_wise_pattern = pattern.node_dict.get('element_wise')

            conv_next_nodes = graph.get_next_nodes(conv.outputs[0])
            if len(conv_next_nodes) == 0:
                self._reduce_output_dims(graph, conv, 'after')
            for next_node in conv_next_nodes:
                if op.eq(next_node.op_type, 'Reshape'):
                    continue
                if not conv_pattern.match(next_node, graph) and \
                    not element_wise_pattern.match(next_node, graph):
                    self._reduce_output_dims(graph, next_node, 'before')

            if element_wises is not None:
                for element_wise in element_wises:
                    next_nodes = graph.get_next_nodes(element_wise.outputs[0])
                    if len(next_nodes) == 0:
                        self._reduce_output_dims(graph, element_wise, 'after')
                    for next_node in next_nodes:
                        if op.eq(next_node.op_type, 'Reshape'):
                            continue
                        if not conv_pattern.match(next_node, graph) and \
                            not element_wise_pattern.match(next_node, graph):
                            self._reduce_output_dims(graph, next_node, 'before')

        for node_dict in match_result.node_dicts:
            conv1d_name = node_dict.get('Conv')[0].name
            conv1d = graph[conv1d_name]
            self._conv1d_to_conv2d(graph, conv1d)
        return True

KnowledgeFactory.add_knowledge('KnowledgeConv1d2Conv2d', KnowledgeConv1d2Conv2d())


def need_to_optimize(graph, knowledge):
    while knowledge.has_next_pattern():
        knowledge.next_pattern()
        match_results = knowledge.get_candidate_sub_graphs(graph)
        if match_results is None or len(match_results) == 0:
            continue
        else:
            return True
    return False

def optimize(graph, knowledge):
    res = False
    while knowledge.has_next_pattern():
        knowledge.next_pattern()
        match_results = knowledge.get_candidate_sub_graphs(graph)
        if match_results is None or len(match_results) == 0:
            continue
        while knowledge.has_next_apply():
            knowledge.next_apply()
            for match_result in match_results:
                res |= knowledge.apply(graph, match_result)
    return res

def evaluate(data_path, parameter = None):
    if parameter is None:
        return False
    onnx_file = parameter.get('file_name')
    onnx_path = os.path.join(data_path, onnx_file)
    mode = parameter.get('mode')
    onnx_graph = OnnxGraph.parse(onnx_path)
    if onnx_graph is None:
        print('onnx model open failed, onnx path: {}'.format(onnx_path))
        return False
    if len(onnx_graph.inputs) == 0 and len(onnx_graph.outputs) == 0:
        print('Invalid onnx model, onnx path: {}'.format(onnx_path))
        return False

    knowledge = KnowledgeConv1d2Conv2d()

    if mode == "evaluate":
        res =  need_to_optimize(onnx_graph, knowledge)
        if res:
            print("The current model need to be optimized")
        else:
            print('The current model does not need to be optimized')
    elif mode == "optimize":
        res = optimize(onnx_graph, knowledge)
        if res:
            out_file = os.path.join(data_path, "{}_optimize.onnx".format(os.path.splitext(onnx_file)[0]))
            onnx_graph.save(out_file)
            print("The current model need to be optimized, the optimized model path is: {}".format(out_file))
        else:
            print('The current model is not optimized')
    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print('Parameter size should be 2 or 3')
        exit(1)
    data_path = sys.argv[1]
    onnx_file = sys.argv[2]
    mode = 'optimize' if len(sys.argv) <= 3 else sys.argv[3]
    if mode not in ['optimize', 'evaluate']:
        print('Invalid parameter {}'.format(mode))
        exit(1)
    parameter = {'file_name': onnx_file, 'mode': mode}
    ret = evaluate(data_path, parameter)

