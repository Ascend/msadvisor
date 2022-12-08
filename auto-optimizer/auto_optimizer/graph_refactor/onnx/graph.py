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

import warnings
import os
from typing import List, Dict, Union, Sequence, Optional

import onnx
import numpy as np
from onnx import helper, GraphProto, ModelProto, OperatorSetIdProto, version_converter

from .. import BaseGraph
from .node import OnnxPlaceHolder, OnnxInitializer, OnnxNode


class OnnxGraph(BaseGraph):

    def __init__(
        self,
        name: str,
        nodes: Optional[List[OnnxNode]] = None,
        inputs: Optional[List[OnnxPlaceHolder]] = None,
        outputs: Optional[List[OnnxPlaceHolder]] = None,
        initializers: Optional[List[OnnxInitializer]] = None,
        value_infos: Optional[List[OnnxPlaceHolder]] = None,
        **kwargs: Dict[str, object]
    ):
        super(OnnxGraph, self).__init__(name, nodes, inputs, outputs, initializers, value_infos)

        opsets = kwargs.get('opset_imports', 11)
        if isinstance(opsets, int):
            opset_imports = onnx.OperatorSetIdProto()
            opset_imports.version = opsets
            opset_imports = [opset_imports]
        elif isinstance(opsets, Sequence):
            opset_imports = [op for op in opsets if not op.domain or op.domain == '']
            if len(opset_imports) < len(opsets):
                warnings.warn(
                    f'Only one domain version is allowed, keep opset with domain "ai.onnx"')
        else:
            opset_imports = opsets

        self._meta = {
            'ir_version': kwargs.get('ir_version', 4),
            'producer_name': kwargs.get('producer_name', 'AutoOptimizer'),
            'producer_version': kwargs.get('producer_version', 'alpha'),
            'domain': kwargs.get('domain', ''),
            'model_version': kwargs.get('model_version', 0),
            'opset_imports': opset_imports
        }

    @classmethod
    def parse(cls, path_or_bytes: Union[str, ModelProto, GraphProto], add_name_suffix: bool = False) -> 'OnnxGraph':
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

        nodes = []
        useless_value_infos = set()
        for node in onnx_graph.node:
            if node.op_type == 'Constant':
                initializers.append(OnnxInitializer.parse(node))
                useless_value_infos.add(node.output[0])
            else:
                nodes.append(OnnxNode.parse(node, add_name_suffix))

        value_infos = []
        for value_info in onnx_graph.value_info:
            if value_info.name not in useless_value_infos:
                value_infos.append(OnnxPlaceHolder.parse(value_info))

        graph = cls(onnx_graph.name, nodes, inputs, outputs, initializers, value_infos, **meta)
        return graph

    def add_input(self, name: str, dtype: str, shape: Sequence[Union[int, str]]) -> OnnxPlaceHolder:
        dtype = np.dtype(dtype)
        graph_input = OnnxPlaceHolder(name, dtype, shape)
        return self._add_input(graph_input)

    def add_output(self, name: str, dtype, shape) -> OnnxPlaceHolder:
        dtype = np.dtype(dtype)
        graph_output = OnnxPlaceHolder(name, dtype, shape)
        return self._add_output(graph_output)

    def add_initializer(self, name: str, value: np.ndarray) -> OnnxInitializer:
        initializer = OnnxInitializer(name, value)
        return self._add_initializer(initializer)

    def add_node(
        self,
        name: str,
        op_type: str,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        attrs: Optional[Dict[str, object]] = None,
        domain: str = ''
    ) -> OnnxNode:
        node = OnnxNode(name, op_type, inputs, outputs, attrs=attrs, domain=domain)
        self.update_map()
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

    def save(self, path: str) -> None:
        onnx.save(self.model(), path)

    def infershape(self) -> None:
        # clear value_infos
        self._value_infos = []
        self._value_map = {}
        model = self.model()
        # TODO: exception
        inferred_model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        graph = inferred_model.graph
        self._value_infos = [OnnxPlaceHolder.parse(v) for v in graph.value_info]
        self._value_map = {v.name: v for v in self._value_infos}

    def extract(
        self,
        new_model_save_path: str,
        input_name_list: List[str],
        output_name_list: List[str],
        enable_model_check: bool = True
    ) -> 'OnnxGraph':
        # TODO: reimplement
        def check_model(model):
            pass
        if not enable_model_check:
            onnx.checker.check_model = check_model

        print('Begin to extract the model.')
        old_model_save_path = '{}_tmp.onnx'.format(new_model_save_path.split('.')[0])
        self.save(old_model_save_path)
        onnx.utils.extract_model(
            old_model_save_path, new_model_save_path, input_name_list, output_name_list)
        os.remove(old_model_save_path)

        print('Extract the model completed, model saved in {}.'.format(
            new_model_save_path))

        return OnnxGraph.parse(new_model_save_path)

    def simplify(self, **kwargs) -> 'OnnxGraph':
        try:
            from onnxsim import simplify
        except ImportError:
            raise RuntimeError("No module named 'onnxsim'")

        model = self.model()
        model_sim, check = simplify(model, **kwargs)
        if not check:
            raise RuntimeError("Simplified ONNX model could not be validated")

        return OnnxGraph.parse(model_sim)

    @property
    def opset_imports(self) -> Optional[Sequence[OperatorSetIdProto]]:
        return self._meta['opset_imports']

    @opset_imports.setter
    def opset_imports(self, opset: Union[int, None]) -> None:
        if not opset:
            self._meta['opset_imports'] = None
        else:
            opset_imports = OperatorSetIdProto()
            opset_imports.version = opset
            model = self.model()
            converted_model = version_converter.convert_version(model, opset)
            self.graph = OnnxGraph.parse(converted_model)
            self._meta['opset_imports'] = [opset_imports]

    def _run(self, model, datas):
        import onnxruntime as rt
        if isinstance(datas, np.ndarray):
            datas = [datas]
        sess = rt.InferenceSession(model)
        inputs = [node.name for node in sess.get_inputs()]
        outputs = [out.name for out in sess.get_outputs()]
        return sess.run(outputs, {name: data for name, data in zip(inputs, datas)})

    def dump(self, data, path = 'dump', outputs = []):
        try:
            from skl2onnx.helpers.onnx_helper import (select_model_inputs_outputs, enumerate_model_node_outputs)
        except ImportError:
            raise RuntimeError('import sk2onnx failed, please install first.')
        ori_model = self.model()
        if len(outputs) == 0:
            outputs = [name for name in enumerate_model_node_outputs(ori_model)]
        new_model = select_model_inputs_outputs(ori_model, outputs)
        new_model_byte = new_model.SerializeToString()
        arrs = self._run(new_model_byte, data)
        if not os.path.exists(path):
            os.makedirs(path, mode = 0o700)
        idx = 0
        for node in ori_model.graph.node:
            for i, output in enumerate(node.output):
                fname = f'{node.name}_{i}.npy'
                np.save(os.path.join(path, fname), arrs[idx])
                idx += 1
