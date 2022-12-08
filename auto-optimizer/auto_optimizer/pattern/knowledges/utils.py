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

from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph, Node
from auto_optimizer.graph_refactor.interface.base_node import BaseNode

def is_lower_onnx_version(graph: BaseGraph, limit_version = 13) -> bool:
    """
    check current onnx version is lower than limit version
    """
    def domain_check(domain): return domain == '' or domain == 'ai.onnx'
    opset_versions = [opset.version for opset in graph.opset_imports if domain_check(opset.domain)]
    return len(opset_versions) == 0 or opset_versions[0] < limit_version

def insert_unsqueeze(graph: BaseGraph, node: BaseNode, attrs, mode: str, refer_index) -> BaseNode:
    if attrs.get('axes') is None:
        raise RuntimeError('insert unsqueeze failed, invalid axes.')
    op_name = f'Unsqueeze_before_{node.name}'
    if not graph.get_node(op_name, Node) is None:
        return None
    if is_lower_onnx_version(graph, 13):
        us = graph.add_node(op_name, 'Unsqueeze', attrs = attrs)
        graph.insert_node(node.name, us, mode=mode, refer_index=refer_index)
    else:
        us = graph.add_node(op_name, 'Unsqueeze')
        axes_name = f'{op_name}_axes'
        graph.add_initializer(axes_name, attrs.get('axes'))
        graph.insert_node(node.name, us, mode=mode, refer_index=refer_index)
        us.inputs.append(axes_name)
    graph.update_map()
    return us

def insert_squeeze(graph: BaseGraph, node: BaseNode, attrs, mode: str, refer_index) -> BaseNode:
    if attrs.get('axes') is None:
        raise RuntimeError('Insert squeeze failed, invalid axes.')
    op_name = f'Squeeze_{mode}_{node.name}_{refer_index}'
    if not graph.get_node(op_name, Node) is None:
        return None
    if is_lower_onnx_version(graph, 13):
        sq = graph.add_node(op_name, 'Squeeze', attrs = attrs)
        graph.insert_node(node.name, sq, mode=mode, refer_index=refer_index)
    else:
        sq = graph.add_node(op_name, 'Squeeze')
        axes_name = f'{op_name}_axes'
        graph.add_initializer(axes_name, attrs.get('axes'))
        graph.insert_node(node.name, sq, mode=mode, refer_index=refer_index)
        sq.inputs.append(axes_name)
    graph.update_map()
    return sq
