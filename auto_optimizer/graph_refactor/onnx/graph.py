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

import copy
from itertools import chain
from collections import deque
from typing import List, Dict, Union

import numpy as np
import onnx
from onnx import helper, GraphProto, ModelProto, OperatorSetIdProto

from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.onnx.node import PlaceHolder, Initializer, Node

class OnnxGraph(BaseGraph):

    def __init__(
        self,
        nodes: List[Node] = None,
        inputs: List[PlaceHolder] = None,
        outputs: List[PlaceHolder] = None,
        initializers: List[Initializer] = None,
        value_infos: List[PlaceHolder] = None,
        name: str = None,
        **kwargs: Dict[str, object]
    ):
        super(OnnxGraph, self).__init__(nodes, inputs, outputs, initializers, value_infos, name, **kwargs)

    @classmethod
    def parse(cls, path_or_bytes: Union[str, ModelProto, GraphProto]) -> 'OnnxGraph':
        if isinstance(path_or_bytes, str):
            onnx_model = onnx.load(path_or_bytes)
        if isinstance(path_or_bytes, ModelProto):
            onnx_model = path_or_bytes
        if isinstance(path_or_bytes, GraphProto):
            onnx_graph = path_or_bytes
            meta = {}
        else:
            onnx_graph = onnx_model.graph
            meta = {
                    'ir_version': onnx_model.ir_version,
                    'domain': onnx_model.domain,
                    'model_version': onnx_model.model_version,
                    'doc_string': onnx_model.doc_string,
                    'opset_imports': onnx_model.opset_import
            }

        inputs = [PlaceHolder.parse(i) for i in onnx_graph.input]
        outputs = [PlaceHolder.parse(o) for o in onnx_graph.output]
        initializers = [Initializer.parse(i) for i in onnx_graph.initializer]
        value_infos = [PlaceHolder.parse(v) for v in onnx_graph.value_info]

        # TODO: Constant Node to Initializer
        nodes = []
        for node in onnx_graph.node:
            nodes.append(Node.parse(node))

        graph = cls(nodes, inputs, outputs, initializers, value_infos, onnx_graph.name, **meta)
        return graph

    def add_placeholder(self, name, dtype, shape, ph_type='input'):
        pass

    def add_initializer(self, name, value):
        pass

    def add_node(self, name, op_type, attrs=None, domain=None):
        pass

    def insert_node(self, refer_name, insert_node, refer_io_index=0, mode='after'):
        pass

    def get_nodes(self, op_type):
        pass

    def remove(self, name, maps=None):
        pass

    def __getitem__(self, key):
        pass

    def __setitem__(self, key, value):
        pass

    @property
    def inputs(self) -> List[PlaceHolder]:
        return self._inputs

    @property
    def outputs(self) -> List[PlaceHolder]:
        return self._outputs

    @property
    def nodes(self) -> List[Node]:
        return self._nodes

    @property
    def initializers(self) -> List[Initializer]:
        return self._initializers

    def proto(self) -> GraphProto:
        self.toposort()
        return helper.make_graph(nodes=[node.proto() for node in self._nodes],
                                name=self.name,
                                inputs=[input.proto() for input in self._inputs],
                                outputs=[output.proto() for output in self._outputs],
                                initializer=[ini.proto() for ini in self._initializers],
                                value_info=[val.proto() for val in self._value_infos]
        )

    def model(self) -> ModelProto:
        return helper.make_model(self.proto(), **self._meta)

    def save(self, path: str):
        onnx.save(self.model(), path)

    def get_prev_node(self, input_name: str) -> Union[Node, PlaceHolder, Initializer]:
        # TODO: raise exception
        return self._prev_map.get(input_name, None)

    def get_next_nodes(self, output_name: str) -> Union[List[Node], List[PlaceHolder], List[Initializer]]:
        # TODO: raise exception
        return self._next_map.get(output_name, [])

    def toposort(self):
        def visited_all_prev_nodes(node, visited):
            for input_name in node.inputs:
                prev_node = self.get_prev_node(input_name)
                if prev_node not in visited and prev_node:
                    return False
            return True

        queue = deque()
        visited = set()
        for node in self._nodes:
            if visited_all_prev_nodes(node, visited):
                queue.append(node)
        
        sorted_nodes = []
        while queue:
            node = queue.popleft()
            if visited_all_prev_nodes(node, visited):
                sorted_nodes.append(node)
                visited.add(node)
                for output_name in node.outputs:
                    for next_node in self.get_next_nodes(output_name):
                        if next_node not in queue and next_node not in visited:
                            queue.append(next_node)
            else:
                queue.append(node)
        
        self._nodes = sorted_nodes