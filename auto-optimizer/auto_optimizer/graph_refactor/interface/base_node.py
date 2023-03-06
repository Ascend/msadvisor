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
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Sequence, Union

import numpy as np


class BaseNode(ABC):
    def __init__(self, name: str, op_type: str) -> None:
        super().__init__()
        self._name: str = name
        self._op_type: str = op_type

    @classmethod
    @abstractmethod
    def parse(cls, _) -> 'BaseNode':
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def op_type(self) -> str:
        return self._op_type

    def __eq__(self, rhs: 'BaseNode') -> bool:
        if not isinstance(rhs, BaseNode):
            return False
        return self.name == rhs.name and self.op_type == rhs.op_type

    def __hash__(self) -> int:
        return hash((self.name, self.op_type))


class Node(BaseNode):
    def __init__(
        self,
        name: str,
        op_type: str,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        attrs: Optional[Dict[str, object]] = None,
        domain: str = ''
    ) -> None:
        """
        A node represents a computation operator in a graph.

        Args:
            name(str): The name of this node.
            op_type(str): The operaton type of this node.
            attrs (Dict[str, object]): A dictionary that maps attribute names to their values.
            inputs (List[Tensor]): A list of zero or more input names.
            outputs (List[Tensor]): A list of zero or more output names.
        """
        super().__init__(name, op_type)
        self._inputs: List[str] = [] if inputs is None else inputs
        self._outputs: List[str] = [] if outputs is None else outputs
        self._attrs: Dict[str, object] = {} if attrs is None else attrs
        self._domain: str = domain if domain is not None else ''

    def __eq__(self, rhs: 'Node') -> bool:
        if not isinstance(rhs, Node):
            return False
        return self.inputs == rhs.inputs \
                and self.outputs == rhs.outputs \
                and self.attrs == rhs.attrs \
                and self.domain == rhs.domain \
                and super().__eq__(rhs)

    def __hash__(self) -> int:
        return super().__hash__()

    @classmethod
    def parse(cls, _) -> 'Node':
        raise NotImplementedError()

    @property
    def inputs(self) -> List[str]:
        return self._inputs

    @inputs.setter
    def inputs(self, inputs: List[str]) -> None:
        self._inputs = inputs

    def get_input_id(self, node_input: str) -> int:
        if node_input not in self._inputs:
            raise RuntimeError(
                f'Name of input should be one of {self._inputs}')
        return self._inputs.index(node_input)

    def get_input_ids(self, node_input: str) -> List[int]:
        return [idx for idx, name in enumerate(self._inputs) if name == node_input]

    @property
    def outputs(self) -> List[str]:
        return self._outputs

    @outputs.setter
    def outputs(self, outputs: List[str]) -> None:
        self._outputs = outputs

    def get_output_id(self, output: str) -> int:
        if output not in self._outputs:
            raise RuntimeError(
                f'Name of output should be one of {self._outputs}')
        return self._outputs.index(output)

    @property
    def attrs(self) -> Dict[str, object]:
        return self._attrs

    def __getitem__(self, key: str) -> object:
        if key not in self._attrs:
            raise KeyError(
                f'Node({self.name}) do not have {key} attribute.')
        return self._attrs[key]

    def __setitem__(self, key: str, value: object) -> None:
        if key not in self._attrs:
            warnings.warn(
                f'Node({self.name}) do not have {key} attribute.')
        self._attrs[key] = value

    @property
    def domain(self) -> str:
        return self._domain
    
    @domain.setter
    def domain(self, domain: str) -> None:
        self._domain = domain

    def __str__(self) -> str:
        return f'Node({self.name}): \n\tinputs={self.inputs}\n\toutputs={self.outputs}\n\tattrs = {self.attrs}\n'

    def __repr__(self) -> str:
        return self.__str__()


class Initializer(BaseNode):
    def __init__(
        self,
        name: str,
        value: Optional[np.ndarray] = None
    ) -> None:
        """
        An initializer represents a tensor which specifies for a graph input or a constant node.

        Args:
            name(str): The name of this initializer.
            value(np.ndarray): The constant value of this initializer.
        """
        super().__init__(name, 'Initializer')
        self._value: np.ndarray = value if value is not None else np.array([])

    def __eq__(self, rhs: 'Initializer') -> bool:
        if not isinstance(rhs, Initializer):
            return False
        return self.value.dtype == rhs.value.dtype \
                and np.array_equal(self.value, rhs.value, equal_nan=True) \
                and super().__eq__(rhs)

    def __hash__(self) -> int:
        return super().__hash__()

    @classmethod
    def parse(cls, _) -> 'Initializer':
        raise NotImplementedError()

    @property
    def value(self) -> np.ndarray:
        return self._value

    @value.setter
    def value(self, value: np.ndarray) -> None:
        self._value = value

    def __str__(self) -> str:
        return f'{self.op_type}({self.name}): (shape={self._value.shape}, dtype={self._value.dtype})\n'

    def __repr__(self) -> str:
        return self.__str__()


class PlaceHolder(BaseNode):
    def __init__(
        self,
        name: str,
        dtype: np.dtype = np.dtype('int64'),
        shape: Optional[Sequence[Union[int, str]]] = None
    ) -> None:
        """
        A placeholder used to store the type and shape information.

        Args:
            name(str): The name of this placeHolder.
            dtype(np.dtype): The data type of this placeHolder.
            shape(List[int]): The shape of this placeHolder.
        """
        super().__init__(name, 'PlaceHolder')
        self._dtype: np.dtype = dtype
        self._shape: Sequence[Union[str, int]] = shape if shape is not None else []

    def __eq__(self, rhs: 'PlaceHolder') -> bool:
        if not isinstance(rhs, PlaceHolder):
            return False
        return self.dtype == rhs.dtype \
                and self.shape == rhs.shape \
                and super().__eq__(rhs)

    def __hash__(self) -> int:
        return super().__hash__()

    @classmethod
    def parse(cls, _) -> 'PlaceHolder':
        raise NotImplementedError()

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: np.dtype) -> None:
        self._dtype = dtype

    @property
    def shape(self) -> Sequence[Union[str, int]]:
        return self._shape

    @shape.setter
    def shape(self, shape: Sequence[Union[str, int]]) -> None:
        if -1 in shape:
            warnings.warn('To represent the dynamic dimension int -1 is converted to str "-1".')
        self._shape = ['-1' if dim == -1 else dim for dim in shape]

    def __str__(self) -> str:
        return f'{self.op_type}({self.name}): (shape={self.shape}, dtype={self.dtype})\n'

    def __repr__(self) -> str:
        return self.__str__()
