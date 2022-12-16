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

from typing import List, Dict, Optional, Sequence, Union, cast

import numpy as np
from onnx import NodeProto, TensorProto, ValueInfoProto, helper, numpy_helper

from .. import Node, PlaceHolder, Initializer

try:
    tensor_dtype_to_np_dtype = helper.tensor_dtype_to_np_dtype
    np_dtype_to_tensor_dtype = helper.np_dtype_to_tensor_dtype

except AttributeError:
    # onnx.__version__ before '1.13.0'
    from onnx import mapping

    def tensor_dtype_to_np_dtype(tensor_dtype: int) -> np.dtype:
        return mapping.TENSOR_TYPE_TO_NP_TYPE[tensor_dtype]

    def np_dtype_to_tensor_dtype(np_dtype: np.dtype) -> int:
        return cast(int, mapping.NP_TYPE_TO_TENSOR_TYPE[np_dtype])


class OnnxNode(Node):
    def __init__(
        self,
        name: str,
        op_type: str,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        attrs: Optional[Dict[str, object]] = None,
        domain: str = ''
    ) -> None:
        super(OnnxNode, self).__init__(name, op_type, inputs, outputs, attrs, domain)

    @classmethod
    def parse(cls, node: NodeProto, add_name_suffix: bool = False) -> 'OnnxNode':

        if not node.name:
            node.name = '{}_{}'.format(node.op_type, node.output[0])

        if add_name_suffix:
            name = '{}_{}'.format(node.name, 'op')
        else:
            name = node.name

        return cls(
            name=name,
            op_type=node.op_type,
            inputs=list(node.input),
            outputs=list(node.output),
            attrs={attr.name: helper.get_attribute_value(attr)
                   for attr in node.attribute},
            domain=node.domain
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
        name: str,
        value: Optional[np.ndarray] = None
    ) -> None:
        super(OnnxInitializer, self).__init__(name, value)

    @classmethod
    def parse(cls, node: Union[NodeProto, TensorProto]) -> 'OnnxInitializer':
        if isinstance(node, NodeProto):
            if hasattr(node, 'op_type') and node.op_type == 'Constant':
                name = node.output[0]
                value = numpy_helper.to_array(node.attribute[0].t)
            else:
                raise RuntimeError('Can\'t parse a Non-Constant Node into a OnnxInitializer.')
        else:
            name = node.name
            value = numpy_helper.to_array(node)
        return cls(
            name=name,
            value=value
        )

    def proto(self) -> TensorProto:
        tensor_proto = numpy_helper.from_array(self.value)
        tensor_proto.name = self._name
        return tensor_proto


class OnnxPlaceHolder(PlaceHolder):
    def __init__(
        self,
        name: str,
        dtype: np.dtype = np.dtype('int64'),
        shape: Optional[Sequence[Union[int, str]]] = None
    ) -> None:
        super(OnnxPlaceHolder, self).__init__(name, dtype, shape)

    @classmethod
    def parse(
        cls,
        node: ValueInfoProto,
        dtype: np.dtype = np.dtype('int64'),
        shape: Optional[Sequence[Union[int, str]]] = None
    ) -> 'OnnxPlaceHolder':
        if node.HasField('type'):
            tensor_type = node.type.tensor_type
            dtype = tensor_dtype_to_np_dtype(tensor_type.elem_type)
            if tensor_type.HasField('shape'):
                shape = [
                    dim.dim_value if dim.HasField('dim_value') else dim.dim_param
                    for dim in tensor_type.shape.dim
                ]
        return cls(
            name=node.name,
            dtype=dtype,
            shape=shape
        )

    def proto(self, dtype: int = 1, shape: Optional[Sequence[Union[int, str]]] = None) -> ValueInfoProto:
        if self.shape:
            shape = ['-1' if dim == -1 else dim for dim in self.shape]
        if self.dtype:
            dtype = np_dtype_to_tensor_dtype(self._dtype)
        return helper.make_tensor_value_info(
            self._name,
            dtype,
            shape
        )
