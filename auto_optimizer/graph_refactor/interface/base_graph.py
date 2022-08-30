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

from .base_node import PlaceHolder, Initializer, Node   

class BaseGraph(ABC):

    def __init__(
        self,
        nodes: List[Node] = None,
        inputs: List[PlaceHolder] = None,
        outputs: List[PlaceHolder] = None,
        initializers: List[Initializer] = None,
        value_infos: List[PlaceHolder] = None,
        name: str = None
    ):
        self._nodes = nodes or []
        self._inputs = inputs or []
        self._outputs = outputs or []
        self._initializers = initializers or []
        self._value_infos = value_infos or []
        self.name = name

        self._node_map = {}
        self._prev_map = {}
        self._next_map = {}

        self.update_map()
    
    def update_map(self):
        # clear map first
        self._node_map = {}
        self._prev_map = {}
        self._next_map = {}

        self._node_map = {n.name: n for n in 
                        chain(self._inputs, self._outputs, self._nodes, self._initializers, self._value_infos)}
        
        for n in self._nodes:
            # update prev node info
            for o in n.outputs:
                # if output name not in map
                if not self._prev_map.get(o):
                    self._prev_map[o] = n
                else:
                    # TODO: ERROR: duplicate output names
                    pass
            # update next node info
            for i in set(n.inputs):
                if not self._next_map.get(i):
                    self._next_map[i] = [n]
                else:
                    self._next_map[i].append(n)

    @classmethod
    @abstractmethod
    def parse(cls, model):
        pass

    @abstractmethod
    def add_input(self, name, dtype, shape) -> PlaceHolder:
        pass

    @abstractmethod
    def add_output(self, name, dtype, shape) -> PlaceHolder:
        pass
    
    @abstractmethod
    def add_initializer(self, name, value) -> Initializer:
        pass

    @abstractmethod
    def add_node(self, name, op_type, attrs=None, domain=None) -> Node:
        pass

    def _add_input(self, graph_input) -> PlaceHolder:
        # TODO: ERROR: duplicate names
        self._node_map[graph_input.name] = graph_input
        self._inputs.append(graph_input)
        return graph_input

    def _add_output(self, graph_output) -> PlaceHolder:
        # TODO: ERROR: duplicate names
        self._node_map[graph_output.name] = graph_output
        self._outputs.append(graph_output)
        return graph_output

    def _add_initializer(self, initializer) -> Initializer:
        # TODO: ERROR: duplicate names
        self._node_map[initializer.name] = initializer
        self._initializers.append(initializer)
        return initializer

    def _add_node(self, node) -> Node:
        # TODO: ERROR: duplicate names
        self._node_map[node.name] = node
        self._nodes.append(node)
        return node

    def insert_node(self, refer_name, insert_node, refer_index=0, mode='after'):
        """Insert a node with single input and output

        Args:
            refer_name: reference node
            insert_node: the node to be inserted
            refer_index: specifies the inserting position within reference node's output id when mode='after'; 
                         specifies the inserting position within reference node's input id when mode='before';
                         Default 0.
            mode: insert the node before or after the reference node. Default 'after'.
        """        
        # TODO: parameter checking with decorator
        # single input and output
        # the value for mode argument

        if refer_name not in self._node_map.keys():
            raise KeyError(
                f'The node name"{refer_name}" not exists in graph')
        else:
            refer_node = self._node_map[refer_name]

        # reference node is input or initializer: convert to inserting node before the next node
        input_flag = False
        if isinstance(refer_node, Initializer) or refer_node in self._inputs:
            if mode == 'before':
                raise RuntimeError(
                    f'Can not insert node before {refer_node.name}.')
            name = refer_node.name
            refer_node = self._next_map[name][0]
            refer_index = refer_node.inputs.index(name)
            mode = 'before'
            input_flag = True
                
        # reference node is output: convert to inserting node after the prev node
        output_flag = False
        if refer_node in self._outputs:
            if mode == 'after':
                raise RuntimeError(
                    f'Can not insert node after {refer_node.name}.')
            name = refer_node.name
            refer_node = self._prev_map[name]
            refer_index = refer_node.outputs.index(name)
            mode = 'after'
            output_flag = True

        # reference node is operator node
        if mode == 'after':
            refer_out_name = refer_node.outputs[refer_index]
            new_out_name = f'{refer_node.name}/{insert_node.name}'
            # connect insert node
            refer_node.outputs[refer_index] = new_out_name
            insert_node.inputs = [new_out_name]
            insert_node.outputs = [refer_out_name]
            # update prev and next map
            self._prev_map[new_out_name] = refer_node
            self._next_map[new_out_name] = [insert_node]
            self._prev_map[refer_out_name] = insert_node
            # deal with situation of inserting node before output
            if output_flag and self._next_map[refer_out_name]:
                for node in self._next_map[refer_out_name]:
                    index = node.get_input_id(refer_out_name)
                    node.inputs[index] = new_out_name
                    self._next_map[new_out_name].append(node)
                    self._next_map[refer_out_name] = []
        elif mode == 'before':
            refer_in_name = refer_node.inputs[refer_index]
            new_in_name = f'{insert_node.name}/{refer_node.name}'
            # connect insert node
            refer_node.inputs[refer_index] = new_in_name
            insert_node.inputs = [refer_in_name]
            insert_node.outputs = [new_in_name]
            # update prev and next map
            self._prev_map[new_in_name] = insert_node
            self._next_map[new_in_name] = [refer_node]
            self._next_map[refer_in_name].append(insert_node)
            self._next_map[refer_in_name].remove(refer_node)
            if input_flag:
                # deal with situation of inserting node before output
                for node in self._next_map[refer_in_name]:
                    if node.name != insert_node.name:
                        index = node.get_input_id(refer_in_name)
                        node.inputs[index] = new_in_name
                        self._next_map[new_in_name].append(node)
                self._next_map[refer_in_name] = [insert_node]

        self._node_map[insert_node.name] = insert_node

    def get_nodes(self, op_type):
        nodes = [node for node in self._node_map.values() if node.op_type == op_type]
        return nodes

    def remove(self, name, maps=None):
        """Remove a specific node from graph
        If map is not provided, it will simply connect the previous node of first input and next nodes of first output.

        Args:
            name: name of the node to be removed
            maps: auto connection map, Default = {0:0}. Keys should be input ids of current node and values should
                  be output ids of current node.

        Return:
            True if remove succeeds, otherwise False
        """
        maps = maps or {0:0}
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
        if isinstance(node, Initializer):
            self._initializers.remove(node)
            self._next_map.pop(name, None)
            return True
        if isinstance(node, Node):
            self._nodes.remove(node)
            for in_id, in_name in enumerate(node.inputs):
                # update next map, node is no longer a next node
                if self._next_map.get(in_name, None):
                    self._next_map[in_name].remove(node)
                    if not self._next_map[in_name]:
                        self._next_map.pop(in_name, None)
                out_id = maps.get(in_id, None)
                # out_id exists, do connection
                if out_id is not None:
                    out_name = node.outputs[out_id]
                    for next_node in self.get_next_nodes(out_name):
                        next_node_in_id = next_node.get_input_id(out_name)
                        next_node.inputs[next_node_in_id] = in_name
                        # update next map, prev node has new next node
                        if not self._next_map.get(in_name):
                            self._next_map[in_name] = [next_node]
                        else:
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
