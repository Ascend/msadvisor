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

import numpy as np
from enum import Enum

from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph


class OnnxType(Enum):
    UNDEFINED  = 0
    FLOAT32    = 1
    UINT8      = 2
    INT8       = 3
    UINT16     = 4
    INT16      = 5
    INT32      = 6
    INT64      = 7
    STRING     = 8
    BOOLEAN    = 9
    FLOAT16    = 10
    FLOAT64    = 11
    UINT32     = 12
    UINT64     = 14
    COMPLEX128 = 15
    BFLOAT16   = 16


numpy_onnx_type_map = \
    { np.int32   : OnnxType.INT32
    , np.int64   : OnnxType.INT64
    , np.float32 : OnnxType.FLOAT32
    , np.float64 : OnnxType.FLOAT64
    }


def make_edge_type_dict(graph: BaseGraph):
    """ 生成图边类型信息
    :param graph: 整图
    :return     : 图边信息字典
    """
    edge_type_dict = {}
    for edge in graph.value_infos:
        edge_type_dict[edge.name] = edge.dtype
    for input_node in graph.inputs:
        edge_type_dict[input_node.name] = input_node.dtype
    for output_node in graph.outputs:
        edge_type_dict[output_node.name] = output_node.dtype
    for initializer in graph.initializers:
        edge_type_dict[initializer.name] = initializer.value.dtype
    return edge_type_dict