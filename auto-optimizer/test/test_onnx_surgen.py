# Copyright 2023 Huawei Technologies Co., Ltd
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

import os
from typing import List

from auto_optimizer.graph_refactor.interface.base_node import BaseNode
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase
from auto_optimizer.pattern.matcher import Matcher, MatchResult
from auto_optimizer.pattern.pattern import MATCH_PATTERN, Pattern, MatchBase


class D(KnowledgeBase):

    def __init__(self) -> None:
        super().__init__()
        res = self._register_apply_funcs(pattern, [self._apply])
        print(f'register result {res}')

    def _apply(self):
        pass


class ConvMatch(MatchBase):

    def __init__(self) -> None:
        super().__init__()

    def match(self, node: BaseNode, graph: BaseGraph) -> bool:
        return True


# get subgraph inputs by pattern
def get_input_op_name_list(pattern: Pattern) -> List[str]:
    input_op_name_list: List[str] = []
    for PatternNode in pattern._inputs:
        input_op_name_list.append(PatternNode._op_name)
    return input_op_name_list


# get subgraph outputs by pattern
def get_output_op_name_list(pattern: Pattern) -> List[str]:
    output_op_name_list: List[str] = []
    for PatternNode in pattern._outputs:
        output_op_name_list.append(PatternNode._op_name)
    return output_op_name_list


def get_subgraph(onnxpath: str, pattern: Pattern) -> None:
    d = D()
    graph = OnnxGraph.parse(onnxpath)
    # 根据定义的子图，在graph中查找匹配，返回一组MatchResult实例
    match_results = d.match_pattern(graph)
    if match_results is None or len(match_results) == 0:
        print('No subgraph is matched.')
        return

    input_op_name_list = get_input_op_name_list(pattern)
    output_op_name_list = get_output_op_name_list(pattern)

    for i, match_result in enumerate(match_results):
        try:
            # 指定截取后模型onnx文件的保存路径
            new_model_path = f'new_model_{i}.onnx'
            # 模型截取的输入边
            input_name_list: List[str] = []
            for input_op_name in input_op_name_list:
                input_name_list.append(match_result.node_dicts[0]
                                       .get(input_op_name)[0].inputs[0])
            # 模型截取的输出边
            output_name_list: List[str] = []
            for output_op_name in output_op_name_list:
                output_name_list.append(match_result.node_dicts[0]
                                        .get(output_op_name)[0].outputs[0])
            # 模型截断后导出
            graph.extract(new_model_path, input_name_list, output_name_list)
        except Exception as err:
            print('Failed to extract subgraph, error: {}'.format(err))
    return


if __name__ == '__main__':

    # 定义子图
    pattern = Pattern() \
        .add_node('Unsqueeze_0', 'Unsqueeze', [ConvMatch()]) \
        .add_node('Expand_0', 'Expand') \
        .add_node('Transpose_0', 'Transpose') \
        .add_node('Mul_0', 'Mul') \
        .add_node('MatMul_0', 'MatMul') \
        .add_node('Add_0', 'Add') \
        .add_edge('Unsqueeze_0', 'Expand_0') \
        .add_edge('Expand_0', 'Transpose_0') \
        .add_edge('Expand_0', 'Mul_0') \
        .add_edge('Transpose_0', 'Mul_0') \
        .add_edge('Mul_0', 'MatMul_0') \
        .add_edge('MatMul_0', 'Add_0') \
        .set_loop(MATCH_PATTERN.MATCH_ONCE)

    # 源onnx路径
    onnx_path = './onnx/aasist_bs1_ori.onnx'
    get_subgraph(onnx_path, pattern)
