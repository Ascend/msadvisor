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

from abc import ABC, abstractmethod
from collections import deque
from itertools import chain
from typing import List, Dict, Union

import numpy as np

from auto_optimizer.graph_refactor.onnx.node import PlaceHolder, Initializer, Node

class BaseGraph(ABC):

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
        self._nodes = nodes if nodes else []
        self._inputs = inputs if inputs else []
        self._outputs = outputs if outputs else []
        self._initializers = initializers if initializers else []
        self._value_infos = value_infos if value_infos else []
        self.name = name

        self._node_map = {}
        self._prev_map = {}
        self._next_map = {}

        self._meta = {
                    'ir_version': kwargs.get('ir_version', 4),
                    'producer_name': kwargs.get('producer_name', 'AutoOptimizer'),
                    'producer_version': kwargs.get('producer_version', 'alpha'),
                    'domain': kwargs.get('domain', ''),
                    'model_version': kwargs.get('model_version', 0),
                    'opset_imports': kwargs.get('opset_imports', None)
        }

        self.update_map()
    
    def update_map(self):
        # clear map first
        self._node_map = {}
        self._prev_map = {}
        self._next_map = {}

        self._node_map = {n.name: n for n in 
                        chain(self._inputs, self._outputs, self._nodes, self._initializers, self._value_infos)}
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

    @classmethod
    @abstractmethod
    def parse(cls, model):
        pass

    def add_input(self, name, dtype, shape) -> PlaceHolder:
        dtype = np.dtype(dtype)
        input = PlaceHolder(name, dtype, shape)
        self._node_map[name] = input
        self._inputs.append(input)
        return input

    def add_output(self, name, dtype, shape) -> PlaceHolder:
        dtype = np.dtype(dtype)
        output = PlaceHolder(name, dtype, shape)
        self._node_map[name] = output
        self._outputs.append(output)
        return output

    def add_initializer(self, name, value) -> Initializer:
        initializer = Initializer(name, value)
        self._node_map[name] = initializer
        self._initializers.append(initializer)
        return initializer

    def add_node(self, name, op_type, attrs=None, domain=None) -> Node:
        node = Node(name, op_type, attrs=attrs, domain=domain)
        self._node_map[name] = node
        self._nodes.append(node)
        return node

    def insert_node(self, refer_name, insert_node, refer_index=0, mode='after'):
        # TODO: exception: name not exists in graph
        refer_node = self._node_map[refer_name]
        if refer_node.op_type == 'PlaceHolder':
            raise RuntimeError(
                'Please use another mode with appropriate reference node or other insert methods.')
        
        if len(insert_node.inputs) > 1 or len(insert_node.outputs) > 1:
            raise RuntimeError(
                'Only support inserting node with single input and output.')

        if mode == 'after':
            refer_out_name = refer_node.outputs[refer_index]
            new_out_name = f'{refer_node.name}/{insert_node.name}'
            # connect insert node
            refer_node.outputs[refer_index] = new_out_name
            insert_node.inputs = [new_out_name]
            insert_node.outputs = [refer_out_name]
            # update prev and next map for new output of reference node
            self._prev_map[new_out_name] = refer_node
            self._next_map[new_out_name] = [insert_node]
            # update prev map for original output of reference node
            self._prev_map[refer_out_name] = insert_node
        elif mode == 'before':
            refer_in_name = refer_node.inputs[refer_index]
            new_in_name = f'{insert_node.name}/{refer_node.name}'
            # connect insert node
            refer_node.inputs[refer_index] = new_in_name
            insert_node.inputs = [refer_in_name]
            insert_node.outputs = [new_in_name]
            # update prev and next map for new input of reference node
            self._prev_map[new_in_name] = insert_node
            self._next_map[new_in_name] = [refer_node]
            # update next map for original input of reference node
            self._next_map[refer_in_name].append(insert_node)
            self._next_map[refer_in_name].remove(refer_node)            
        else:
            raise ValueError(
                f'The value for mode argument should be "after" or "before", but got "{mode}"')
        
        self._node_map[insert_node.name] = insert_node

    def get_nodes(self, op_type):
        nodes = []
        for node in self._node_map.values():
            if node.op_type == op_type:
                nodes.append(node)
        return nodes

    def remove(self, name, maps=None):
        maps = maps if maps else {0:0}
        # TODO: exception: name not exist in graph
        node = self._node_map[name]
        self._node_map.pop(name, None)
        if node in self._inputs:
            self._inputs.remove(node)
            self._next_map.pop(name, None)
            return True
        if node in self._outputs:
            self._outputs.remove(node)
            self._prev_map.pop(name, None)
            return True
        if node in self._initializers:
            self._initializers.remove(node)
            self._next_map.pop(name, None)
            return True
        if node in self._nodes:
            self._nodes.remove(node)
            for in_id, in_name in enumerate(node.inputs):
                # update next map, node is no longer a next node
                self._next_map[in_name].remove(node)
                out_id = maps.get(in_id, None)
                # out_id exists, do connection
                if out_id is not None:
                    out_name = node.outputs[out_id]
                    for next_node in self.get_next_nodes(out_name):
                        next_node_in_id = next_node.get_input_id(out_name)
                        next_node.inputs[next_node_in_id] = in_name
                        # update next map, prev node has new next node
                        self._next_map[in_name].append(next_node)
            # update prev and next map, outputs of node no long exist
            for out_name in node.outputs:
                self._prev_map.pop(out_name, None)
                self._next_map.pop(out_name, None)
            return True
        return False

    def __getitem__(self, key):
        return self._node_map[key]

    def __setitem__(self, key, value):
        # TODO
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
    
    @property
    def value_infos(self) -> List[PlaceHolder]:
        return self._value_infos

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

    @abstractmethod
    def save(self, path):
        pass
