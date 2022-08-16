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
from typing import List, Dict, Union

import numpy as np
import onnx
from onnx import helper, GraphProto, ModelProto, OperatorSetIdProto

from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.onnx.node import PlaceHolder, Initializer, Node

class OnnxGraph(BaseGraph):
    def __init__(
        self,
        nodes: List[Node] = [],
        inputs: List[PlaceHolder] = [],
        outputs: List[PlaceHolder] = [],
        initializers: List[Initializer] = [],
        value_infos: List[PlaceHolder] = [],
        name: str = None,
        **kwargs: Dict[str, object]
    ):
        super().__init__()
        self._nodes = nodes
        self._inputs = inputs
        self._outputs = outputs
        self._initializers = initializers
        self._value_infos = value_infos
        self.name = name

        self._node_map = {}
        self._prev_map = {}
        self._next_map = {}

        self._meta = {'ir_version': kwargs.get('ir_version', 4),
                    'producer_name': kwargs.get('producer_name', 'AutoOptimizer'),
                    'producer_version': kwargs.get('producer_version', 'beta'),
                    'domain': kwargs.get('domain', ''),
                    'model_version': kwargs.get('model_version', 0),
                    'opset_imports': kwargs.get('opset_imports', None)
        }

        self.update_map()

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
            meta = {'ir_version': onnx_model.ir_version,
                    'domain': onnx_model.domain,
                    'model_version': onnx_model.model_version,
                    'doc_string': onnx_model.doc_string,
                    'opset_imports': onnx_model.opset_import
            }

        inputs = [PlaceHolder.parse(i) for i in onnx_graph.input]
        outputs = [PlaceHolder.parse(o) for o in onnx_graph.output]
        initializers = [Initializer.parse(i) for i in onnx_graph.initializer]
        value_infos = [PlaceHolder.parse(v) for v in onnx_graph.value_info]

        nodes = []
        for node in onnx_graph.node:
            nodes.append(Node.parse(node))

        graph = cls(nodes, inputs, outputs, initializers, value_infos, onnx_graph.name, **meta)
        return graph

    def update_map(self):
        # clear map first
        self._node_map = {}
        self._prev_map = {}
        self._next_map = {}

        for n in chain(self._inputs, self._outputs, self._nodes, self._initializers, self._value_infos):
            self._node_map[n.name] = n
        # update prev node info
        for n in self._nodes:
            for o in n.outputs:
                # if output name not in map
                if not self._prev_map.get(o):
                    self._prev_map[o] = n
                else:
                    # TODO: ERROR: duplicate output names
                    pass
        # update next node info
        for n in self._nodes:
            for i in n.inputs:
                if not self._next_map.get(i):
                    self._next_map[i] = [n]
                else:
                    self._next_map[i].append(n)

    def add_placeholder(self, name, dtype, shape, ph_type='input'):
        pass

    def add_initializer(self, name, value):
        pass

    def add_node(self, name, op_type, attrs={}, domain=None):
        pass

    def insert_node(self, refer_name, insert_node, refer_io_index=0, mode='after'):
        pass

    def get_nodes(self, op_type):
        pass

    def remove(self, name, maps={0:0}):
        pass

    def __getitem__(self, key):
        pass

    def __setitem__(self, key, value):
        pass

    @property
    def inputs(self):
        pass

    @property
    def outputs(self):
        pass

    def to_graph(self) -> GraphProto:
        self.toposort()
        return helper.make_graph(nodes=[node.to_proto() for node in self._nodes],
                                name=self.name,
                                inputs=[input.to_proto() for input in self._inputs],
                                outputs=[output.to_proto() for output in self._outputs],
                                initializer=[ini.to_proto() for ini in self._initializers]
        )

    def to_model(self) -> ModelProto:
        return helper.make_model(self.to_graph(), **self._meta)

    def save(self, path: str):
        onnx.save(self.to_model(), path)

    def toposort(self):
        def visited_all_prev_nodes(node, visited):
            for input_name in node.inputs:
                prev_node = self.get_prev_node(input_name)
                if not visited.get(prev_node, False) and prev_node:
                    return False
            return True

        queue = []
        visited = {}
        for node in self._nodes:
            if visited_all_prev_nodes(node, visited):
                queue.append(node)
        
        sorted_nodes = []
        while queue:
            node = queue.pop(0)
            if visited_all_prev_nodes(node, visited):
                sorted_nodes.append(node)
                visited[node] = True
                for output_name in node.outputs:
                    for next_node in self.get_next_nodes(output_name):
                        if next_node not in queue and not visited.get(next_node):
                            queue.append(next_node)
            else:
                queue.append(node)
        
        self._nodes = sorted_nodes