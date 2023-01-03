# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
from auto_optimizer.graph_refactor.interface.base_graph import (BaseGraph, Initializer, Node)
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
            .set_node_loop('Reshape', MATCH_PATTERN.MATCH_ONCE)
        self._register_apply_funcs(pattern, [self._optimize_apply])

        # inference config
        self._dump_num = 3 # generate inference dump data for 3 time
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

    def _get_input_shape_range(self, dynamic_axes):
        '''
        parse input_shape_range from model.cfg, if none, then use default range [1, 64]
        '''
        res = {}
        for axes in dynamic_axes:
            res[axes] = [1, 64]
        input_shape_range = '' # 'n,h,w=1~100,-1,64*'
        cfg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../model.cfg')
        if not os.path.exists(cfg_path):
            return res
        with open(cfg_path) as f:
            contents = f.readlines()
        for line in contents:
            if line.startswith('input_shape_range'):
                pos = line.find('#')
                line = line[:pos]
                pos = line.find('=')
                if pos == -1:
                    break
                input_shape_range = line[pos + 1:].replace(' ', '')
                if input_shape_range.startswith('\''):
                    input_shape_range = input_shape_range[1:]
                if input_shape_range.endswith('\''):
                    input_shape_range = input_shape_range[:-1]
        shape_array = input_shape_range.split('=')
        if len(shape_array) != 2:
            return res
        inputs_name = shape_array[0].split(',')
        inputs_shape = shape_array[1].split(',')
        if len(inputs_name) != len(inputs_shape) or len(inputs_name) == 0:
            return res
        for i, name in enumerate(inputs_name):
            if res.get(name) is None:
                continue
            shape_range = inputs_shape[i]
            rng = shape_range.split('~')
            if len(rng) == 1:
                if rng[0].isdigit():
                    res[name] = [int(rng[0]), int(rng[0])]
                elif rng[0].endswith('*') and rng[0][:-1].isdigit():
                    val = []
                    for j in range(self._dump_num):
                        val.append((j + 1) * int(rng[0][:-1]))
                    res[name] = val
            elif len(rng) == 2:
                if rng[0].isdigit() and rng[1].isdigit():
                    res[name] = [int(rng[0]), int(rng[1])]
            else:
                continue
        return res

    def _generate_dump_data(self, graph: BaseGraph, dynamic_axes):
        '''
        generate operator dump by inference base on skl2onnx module
        '''
        inputs_shape_range = self._get_input_shape_range(dynamic_axes)
        for j in range(self._dump_num):
            real_dump_path = f'{self._dump_path}{j}'
            if not os.path.exists(real_dump_path):
                os.makedirs(real_dump_path, 0o700)
            dynamic_input = {}
            # generate dynamic input shape
            for axis in dynamic_axes:
                rng = inputs_shape_range.get(axis)
                if rng is None:
                    rng = [1, 64] # default range
                if len(rng) == self._dump_num:
                    dynamic_input[axis] = rng[j]
                elif rng[0] == rng[1]:
                    dynamic_input[axis] = rng[0]
                else:
                    dynamic_input[axis] = np.random.randint(rng[0], rng[1], 1, dtype = np.int32)[0]
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

        # get input and output shapes from multi-group dump data
        in_shapes, out_shapes = [], []
        for i in range(self._dump_num):
            real_dump_path = f'{self._dump_path}{i}'
            if prev_node.op_type == 'PlaceHolder':
                dump_file = f'{prev_node.name}.npy'
            elif prev_node.op_type != 'Initializer':
                dump_file = f'{prev_node.name}_{prev_node.get_output_id(reshape.inputs[0])}.npy'
            else:
                raise RuntimeError('Reshape prev node is Constant type.')
            data_in = np.load(os.path.join(real_dump_path, dump_file))
            data_out = np.load(os.path.join(real_dump_path, f'{reshape.name}_0.npy'))

            in_shapes.append(data_in.shape)
            out_shapes.append(data_out.shape)
        return np.array(in_shapes), np.array(out_shapes)

    def _calculate_shape(self, in_shapes, out_shapes):
        '''
        calculate Reshape input 'shape' and optimization apply
        '''
        if len(in_shapes) == 0 or len(out_shapes) == 0:
            raise RuntimeError('invalid input or output shapes.')

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
                insert.get('unsqueeze').append(in_dim)
            # compute next dimension
            in_dim += 1
        return insert, [dim if dim is not None else -1 for dim in shape]

    def _modify_dimension(self, graph, reshape, unsqueeze_dims, squeeze_dims):
        '''
        modify dimension by insert unsqueeze or squeeze
        '''
        if unsqueeze_dims is not None and len(unsqueeze_dims) != 0:
            attrs = {'axes': np.array(unsqueeze_dims, dtype = np.int64)}
            insert_unsqueeze(graph, reshape, attrs, mode = 'before', refer_index = 0)

        if squeeze_dims is not None and len(squeeze_dims) != 0:
            attrs = {'axes': np.array(squeeze_dims, dtype = np.int64)}
            insert_squeeze(graph, reshape, attrs, mode = 'after', refer_index = 0)

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
            if shape.count(-1) > 1:
                # if exist two or more dynamic dimension, cannot be optimized.
                continue

            # set shape value for Reshape operator
            graph.add_initializer(f'Shape_for_{reshape.name}', np.array(shape))
            reshape.inputs[1] = f'Shape_for_{reshape.name}'

            self._modify_dimension(graph, reshape, insert.get('unsqueeze'), insert.get('squeeze'))

            optimize_result = True
        return optimize_result

    def _optimize_apply(self, graph: BaseGraph, match_result: MatchResult):
        '''
        optimize Reshape operator for dynamic model
        '''
        if match_result is None or match_result.is_empty():
            return False

        # check reshape is or not optimized
        is_optimized = True
        for node_dict in match_result.node_dicts:
            if 'Reshape' not in node_dict:
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
        dynamic_axes = set()
        for x in graph.inputs:
            for shape in x.shape:
                if not type(shape) == int:
                    dynamic_axes.add(shape)
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

