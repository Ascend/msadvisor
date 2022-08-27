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
from onnx import NodeProto, TensorProto, ValueInfoProto, helper, numpy_helper
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE, NP_TYPE_TO_TENSOR_TYPE

from .. import Node, PlaceHolder, Initializer

class OnnxNode(Node):
    def __init__(
        self,
        name: str = None,
        op_type: str = None,
        inputs: List[str] = [],
        outputs: List[str] = [],
        attrs: Dict[str, object] = {},
        domain: str = None
    ):
        super(OnnxNode, self).__init__(name, op_type, inputs, outputs, attrs, domain)
    
    @classmethod
    def parse(cls, node:NodeProto):
        return cls(
            name = node.name, 
            op_type = node.op_type, 
            inputs = list(node.input), 
            outputs = list(node.output), 
            attrs = {attr.name: helper.get_attribute_value(attr)
                     for attr in node.attribute}, 
            domain = node.domain
        )
    
    def proto(self) -> NodeProto:
        return helper.make_node(
            self.op_type, 
            self.inputs, 
            self.outputs, 
            name=self.name, 
            domain=self.domain, 
            **self.attrs
        )


class OnnxInitializer(Initializer):
    def __init__(
        self,
        name: str = None,
        value: np.ndarray = None
    ):
        super(OnnxInitializer, self).__init__(name, value)

    @classmethod
    def parse(cls, node:Union[NodeProto, TensorProto]):
        if hasattr(node, 'op_type') and node.op_type == 'Constant':
            name = node.output[0]
            value = numpy_helper.to_array(node.attribute[0].t)
        else:
            name = node.name
            value = numpy_helper.to_array(node)
        return cls(
            name = name, 
            value = value
        )
    
    def proto(self) -> TensorProto:
        return helper.make_tensor(
            self._name,
            NP_TYPE_TO_TENSOR_TYPE[self._value.dtype],
            self._value.shape,
            self._value.flatten()
        )


class OnnxPlaceHolder(PlaceHolder):
    def __init__(
        self,
        name: str = None,
        dtype: np.dtype = None,
        shape: List[int] = None
    ):
        super(OnnxPlaceHolder, self).__init__(name, dtype, shape)

    @classmethod
    def parse(cls, node:ValueInfoProto):
        tensor_type = node.type.tensor_type
        dtype = TENSOR_TYPE_TO_NP_TYPE[tensor_type.elem_type]
        shape = [
            dim.dim_value if dim.dim_value > 0 else -1
            for dim in tensor_type.shape.dim
        ]
        return cls(
            name = node.name, 
            dtype = dtype,
            shape = shape
        )
    
    def proto(self) -> ValueInfoProto:
        return helper.make_tensor_value_info(
            self._name,
            NP_TYPE_TO_TENSOR_TYPE[self._dtype],
            self._shape
        )