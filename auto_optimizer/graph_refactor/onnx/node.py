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
import warnings

import numpy as np
from onnx import NodeProto, TensorProto, ValueInfoProto, helper, numpy_helper
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE, NP_TYPE_TO_TENSOR_TYPE

from auto_optimizer.graph_refactor.interface.base_node import BaseNode


class Node(BaseNode):
    def __init__(
        self,
        name: str = None,
        op_type: str = None,
        inputs: List[str] = [],
        outputs: List[str] = [],
        attrs: Dict[str, object] = {},
        domain: str = None
    ):
        """
        A node represents a computation operator in a graph.

        Args:
            name(str): The name of this node.
            op_type(str): The operaton type of this node.
            attrs (Dict[str, object]): A dictionary that maps attribute names to their values.
            inputs (List[Tensor]): A list of zero or more input names.
            outputs (List[Tensor]): A list of zero or more output names.
        """
        self._name = name
        self._op_type = op_type
        self._inputs = inputs
        self._outputs = outputs
        self._attrs = attrs
        self._domain = domain  
    
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

    @property
    def op_type(self):
        return self._op_type

    @property
    def inputs(self):
        return self._inputs
    
    @inputs.setter
    def inputs(self, inputs:List[str]):
        self._inputs = inputs
    
    def get_input_id(self, input:str):
        if input not in self._inputs:
            raise RuntimeError(
                f'Name of input should be one of {self._inputs}')
        else:
            return self._inputs.index(input)

    @property
    def outputs(self):
        return self._outputs
    
    @outputs.setter
    def outputs(self, outputs:List[str]):
        self._outputs = outputs
    
    def get_output_id(self, output:str):
        if output not in self._outputs:
            raise RuntimeError(
                f'Name of output should be one of {self._outputs}')
        else:
            return self._outputs.index(output)
    
    @property
    def attrs(self):
        return self._attrs
    
    def __getitem__(self, key):
        if key not in self._attrs:
            raise KeyError(
                f'Node({self.name}) do not have {key} attribute.')
        return self._attrs[key]

    def __setitem__(self, key, value):
        if key not in self._attrs:
            warnings.warn(
                f'Node({self.name}) do not have {key} attribute.')
        self._attrs[key] = value

    @property
    def domain(self):
        return self._domain

    def __str__(self) -> str:
        return f'Node({self.name}): \n\tinputs={self.inputs}\n\toutputs={self.outputs}\n\tattrs = {self.attrs}\n'

    def __repr__(self) -> str:
        return self.__str__()


class Initializer(BaseNode):
    def __init__(
        self,
        name: str = None,
        value: np.ndarray = None
    ):
        """
        An initializer represents a tensor which specifies for a graph input or a constant node.

        Args:
            name(str): The name of this initializer.
            value(np.ndarray): The constant value of this initializer.
        """
        self._name = name
        self._op_type = 'Initializer'
        self._value = value

    @classmethod
    def parse(cls, node:Union[NodeProto, TensorProto]):
        if hasattr(node, 'op_type') and node.op_type == 'Constant':
            value = numpy_helper.to_array(node.attribute[0].t)
        else:
            value = numpy_helper.to_array(node)
        return cls(
            name = node.name, 
            value = value
        )
    
    def proto(self) -> TensorProto:
        return helper.make_tensor(
            self._name,
            NP_TYPE_TO_TENSOR_TYPE[self._value.dtype],
            self._value.shape,
            self._value.flatten()
        )

    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, value:np.ndarray):
        self._value = value
    
    def __str__(self) -> str:
        return f'{self.op_type}({self.name}): (shape={self._value.shape}, dtype={self._value.dtype})\n'

    def __repr__(self) -> str:
        return self.__str__()


class PlaceHolder(BaseNode):
    def __init__(
        self,
        name: str = None,
        dtype: np.dtype = None,
        shape: List[int] = None
    ):
        """
        A placeholder used to store the type and shape information.

        Args:
            name(str): The name of this placeHolder.
            dtype(np.dtype): The data type of this placeHolder.
            shape(List[int]): The shape of this placeHolder.
        """
        self._name = name
        self._op_type = 'PlaceHolder'
        self._dtype = dtype
        self._shape = shape

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

    @property
    def dtype(self):
        return self._dtype
    
    @dtype.setter
    def dtype(self, dtype:np.dtype):
        self._dtype = dtype

    @property
    def shape(self):
        return self._shape
    
    @shape.setter
    def shape(self, shape:List[int]):
        self._shape = shape
    
    def __str__(self) -> str:
        return f'{self.op_type}({self.name}): (shape={self.shape}, dtype={self.dtype})\n'

    def __repr__(self) -> str:
        return self.__str__()