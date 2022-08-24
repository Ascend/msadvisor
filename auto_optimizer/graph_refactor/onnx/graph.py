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

from typing import List, Dict, Union

import numpy as np
import onnx
from onnx import helper, GraphProto, ModelProto, OperatorSetIdProto

from .. import BaseGraph
from .. import PlaceHolder, Initializer, Node
from .node import OnnxPlaceHolder, OnnxInitializer, OnnxNode

class OnnxGraph(BaseGraph):

    def __init__(
        self,
        nodes: List[OnnxNode] = None,
        inputs: List[OnnxPlaceHolder] = None,
        outputs: List[OnnxPlaceHolder] = None,
        initializers: List[OnnxInitializer] = None,
        value_infos: List[OnnxPlaceHolder] = None,
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

        inputs = [OnnxPlaceHolder.parse(i) for i in onnx_graph.input]
        outputs = [OnnxPlaceHolder.parse(o) for o in onnx_graph.output]
        initializers = [OnnxInitializer.parse(i) for i in onnx_graph.initializer]
        value_infos = [OnnxPlaceHolder.parse(v) for v in onnx_graph.value_info]

        # TODO: Constant Node to Initializer
        nodes = []
        for node in onnx_graph.node:
            nodes.append(OnnxNode.parse(node))

        graph = cls(nodes, inputs, outputs, initializers, value_infos, onnx_graph.name, **meta)
        return graph

    def add_input(self, name, dtype, shape) -> OnnxPlaceHolder:
        dtype = np.dtype(dtype)
        graph_input = OnnxPlaceHolder(name, dtype, shape)
        return self._add_input(graph_input)

    def add_output(self, name, dtype, shape) -> OnnxPlaceHolder:
        dtype = np.dtype(dtype)
        graph_output = OnnxPlaceHolder(name, dtype, shape)
        return self._add_output(graph_output)

    def add_initializer(self, name, value) -> OnnxInitializer:
        initializer = OnnxInitializer(name, value)
        return self._add_initializer(initializer)

    def add_node(self, name, op_type, attrs=None, domain=None) -> OnnxNode:
        node = OnnxNode(name, op_type, attrs=attrs, domain=domain)
        return self._add_node(node)

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

    def infershape(self):
        model = self.model()
        # TODO: exception
        inferred_model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        graph = inferred_model.graph
        self._value_infos = [OnnxPlaceHolder.parse(v) for v in graph.value_info]
        for n in self._value_infos:
            self._node_map[n.name] = n