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

import os
import numpy as np

from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory
from auto_optimizer.pattern.pattern import Pattern, MATCH_PATTERN
from auto_optimizer.pattern.matcher import MatchResult
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph, Initializer
from auto_optimizer.graph_refactor.interface.base_node import BaseNode
from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase
from auto_optimizer.pattern.knowledges.utils import (insert_squeeze, insert_unsqueeze)
from auto_optimizer.common.utils import dump_op_outputs


@KnowledgeFactory.register()
class KnowledgeDynamicReshape(KnowledgeBase):
    """ calculate Reshape 'shape' value and replace """

    def __init__(self) -> None:
        super().__init__()

        pattern = Pattern() \
            .add_node('Reshape', ['Reshape']) \
            .set_input('Reshape') \
            .set_output('Reshape') \
            .set_node_loop('Reshape', MATCH_PATTERN.MATCH_ONCE)
        self._register_apply_funcs(pattern, [self._optimize_apply])

        # inference config
        self._dump_num = 3 # generate inferenca dump data for 3 time
        self._dump_path = 'dump'

    def _generate_inputs(self, dynamic_input, input_shape, input_dtype):
        '''
        generate random number for model input
        '''
        static_shape = [ dynamic_input.get(i) or i for i in input_shape ]
        if input_dtype in ['int32', 'int64']:
            data = np.random.randint(1, 10, static_shape, dtype = input_dtype)
        elif input_dtype in ['float16', 'float32', 'float64']:
            data = np.random.rand(*static_shape).astype(input_dtype)
        else:
            raise RuntimeError('data type: {} not supported.'.format(input_dtype))
        return data

    def _generate_dump_data(self, graph: BaseGraph, dynamic_axes):
        '''
        generate operator dump by inference base on skl2onnx module
        '''
        for j in range(self._dump_num):
            real_dump_path = f'{self._dump_path}{j}'
            if not os.path.exists(real_dump_path):
                os.makedirs(real_dump_path, 0o700)
            dynamic_input = {}
            # generate dynamic input shape
            for i, axis in enumerate(dynamic_axes):
                dynamic_input[axis] = (j + i + 1) * (j + i + 2)
            # generate operator dump
            input_data = []
            for x in graph.inputs:
                data = self._generate_inputs(dynamic_input, x.shape, x.dtype)
                input_data.append(data)
                np.save(os.path.join(real_dump_path, f'{x.name}.npy'), data)
            # inference
            dump_op_outputs(graph, input_data, real_dump_path)

    def _remove_dump_data(self):
        '''
        remove all dump data, clean disk space
        '''
        for j in range(self._dump_num):
            real_dump_path = f'{self._dump_path}{j}'
            if not os.path.exists(real_dump_path):
                continue
            for root, dirs, files in os.walk(real_dump_path, topdown = False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(real_dump_path)

    def _get_inout_shapes_from_dump_data(self, graph: BaseGraph, reshape: BaseNode):
        '''
        get reshape input and output shape by dump data
        '''
        # check reshape type
        prev_node = graph.get_prev_node(reshape.inputs[0])
        if prev_node is None:
            prev_node = graph[reshape.inputs[0]] # model input, prev node type is 'PlaceHolder'
        # get reshape input and output shapes from multi dump data
        in_shapes, out_shapes = [], []
        for i in range(self._dump_num):
            real_dump_path = f'{self._dump_path}{i}'
            if prev_node.op_type == 'PlaceHolder':
                dump_file = f'{prev_node.name}.npy'
            elif prev_node.op_type != 'Initializer':
                dump_file = f'{prev_node.name}_{prev_node.get_output_id(reshape.inputs[0])}.npy'
            else:
                raise RuntimeError('Reshape prev node is Constant type.')
            reshape_input = np.load(os.path.join(real_dump_path, dump_file))
            reshape_output = np.load(os.path.join(real_dump_path, f'{reshape.name}_0.npy'))
            in_shapes.append(reshape_input.shape)
            out_shapes.append(reshape_output.shape)
        return np.array(in_shapes), np.array(out_shapes)

    def _calculate_shape(self, in_shapes, out_shapes):
        '''
        calculate Reshape input 'shape' and optimization apply
        '''
        if len(in_shapes) == 0 or len(out_shapes) == 0:
            return None, None

        # init shape, dynamic dim need to calculate
        shape = [out_shapes[0][i] if is_constant else None
            for i, is_constant in enumerate(np.all(out_shapes == out_shapes[0, :], axis = 0))]

        insert = { 'squeeze': [], 'unsqueeze': [] }
        in_dim = 0
        for dim in range(len(shape)):
            if not shape[dim] is None:
                continue
            # the dim is dynamic
            tmp_dim = in_dim
            while in_dim < in_shapes.shape[1]:
                if np.all(out_shapes[:, dim] == in_shapes[:, in_dim]):
                    break
                in_dim += 1
            if in_dim == in_shapes.shape[1]:
                in_dim = tmp_dim
                continue
            if dim == in_dim:
                # the dim has no change
                #                (-1, 0, 32)
                # (bs, len, 256) -----------> (8*bs, len, 32)
                shape[dim] = 0
            elif dim < in_dim:
                #                  (-1, 1, 0, 32)                     Squeeze
                # (bs, 8, len, 32) --------------> (8*bs, 1, len, 32) -------> (8*bs, len, 32)
                shape[dim] = 0
                while dim < in_dim:
                    shape.insert(dim, 1)
                    insert.get('squeeze').append(dim)
                    dim += 1
            else:
                #                 Unsqueeze                     (-1, 8, 0, 32)
                # (8*bs, len, 32) ---------> (8*bs, 1, len, 32) --------------> (bs, 8, len, 32)
                shape[dim] = 0
                insert('unsqueeze').append(in_dim)
            # compute next dimension
            in_dim += 1
        # if exist two or more dynamic dimension, then will not be optimized.
        if shape.count(None) <= 1:
            return insert, [dim if not dim is None else -1 for dim in shape]
        else:
            return None, None

    def _optimize_reshape(self, graph: BaseGraph):
        '''
        visit all Reshape operators and optimize
        '''
        optimize_result = False
        for reshape in graph.get_nodes('Reshape'):
            if not graph.get_node(reshape.inputs[1], Initializer) is None:
                continue

            # get 'Reshape' input and output shape
            in_shapes, out_shapes = self._get_inout_shapes_from_dump_data(graph, reshape)

            # optimize Reshape operator
            insert, shape = self._calculate_shape(in_shapes, out_shapes)
            if insert is None or shape is None:
                continue
            # insert squeeze/unsqueeze
            if len(insert.get('unsqueeze')) != 0:
                attrs = {'axes': np.array(insert.get('unsqueeze'), dtype = np.int64)}
                insert_unsqueeze(graph, reshape, attrs, mode = 'before', refer_index = 0)
            if len(insert.get('squeeze')) != 0:
                attrs = {'axes': np.array(insert.get('squeeze'), dtype = np.int64)}
                insert_squeeze(graph, reshape, attrs, mode = 'after', refer_index = 0)

            # add constant shape for Reshape operator
            graph.add_initializer(f'Shape_for_{reshape.name}', np.array(shape))
            reshape.inputs[1] = f'Shape_for_{reshape.name}'

            optimize_result = True
        return optimize_result

    def _optimize_apply(self, graph: BaseGraph, match_result: MatchResult):
        '''
        optimize Reshape operator for dynamic model
        '''
        if match_result is None or match_result.is_empty():
            return False

        is_optimized = True
        for node_dict in match_result.node_dicts:
            if not 'Reshape' in node_dict:
                continue
            for node in node_dict.get('Reshape'):
                reshape = graph.get_node(node.name, Node)
                if reshape is None:
                    continue
                if graph.get_node(reshape.inputs[1], Initializer) is None:
                    is_optimized = False
        if is_optimized:
            return False

        # check model is dynamic and get dynamic input name
        for x in graph.inputs:
            dynamic_axes = set([shape for shape in x.shape if not type(shape) == int])
        if len(dynamic_axes) == 0:
            return False

        optimize_result = False
        try:
            # infer and generate operator dump, the purpose is to obtain the input and output shapes
            self._generate_dump_data(graph, dynamic_axes)

            # optimize Reshape operator
            optimize_result = self._optimize_reshape(graph)
        finally:
            # release temp resource
            self._remove_dump_data()

        return optimize_result

