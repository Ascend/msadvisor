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
import operator as op

from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory
from auto_optimizer.pattern.pattern import MatchBase, Pattern, MATCH_PATTERN
from auto_optimizer.pattern.matcher import MatchResult
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.interface.base_node import BaseNode, Node
from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase
from auto_optimizer.pattern.knowledges.utils import (insert_squeeze, insert_unsqueeze)


class ReshapeMatch(MatchBase):
    def __init__(self) -> None:
        super().__init__()

    def match(self, node: BaseNode, graph: BaseGraph) -> bool:
        if node is None:
            return False
        if not op.eq(node.op_type, 'Reshape'):
            return False
        return True

pattern = Pattern() \
    .add_node('Reshape', ['Reshape'], [ReshapeMatch()]) \
    .set_input('Reshape') \
    .set_output('Reshape') \
    .set_node_loop(MATCH_PATTERN.MATCH_ONCE)

@KnowledgeFactory.register()
class KnowledgeDynamicReshape(KnowledgeBase):
    """"""

    def __init__(self) -> None:
        super().__init__()
        self._register_apply_funcs(pattern, [self.optimize_apply])

    def get_model_dynamic_axes(self, graph: BaseGraph) -> bool:
        dynamic_axes = set()
        dynamic_axes.add([shape for shape in input.shape for input in graph.inputs])
        for input in graph.inputs:
            for shape in input.shape:
                if not type(shape) == int:
                    dynamic_axes.add(shape)
        return dynamic_axes

    def generate_inputs(self, dynamic_input, input_shape, input_dtype):
        static_shape = [ dynamic_input.get(i, None) or i for i in input_shape ]
        if input_dtype in ['int32', 'int64']:
            data = np.random.randint(1, 10, static_shape, dtype = input_dtype)
        elif input_dtype in ['float16', 'float32', 'float64']:
            data = np.random.rand(*input_shape).astype(input_dtype)
        else:
            raise RuntimeError('data type: {} not supported.'.format(input_dtype))
        return data

    def generate_dump_data(self, graph: BaseGraph, dump_num, dynamic_axes, dump_path = 'dump'):
        for j in range(dump_num):
            real_dump_path = f'{dump_path}{j}'
            if os.path.exists(real_dump_path):
                os.makedirs(real_dump_path, 0o700)
            dynamic_input = []
            for i, axis in enumerate(dynamic_axes):
                dynamic_input[axis] = (j + i + 1) * (j + i + 2)
            input_data = []
            for input in graph.inputs:
                data = self.generate_inputs(dynamic_input, input.shape, input.dtype)
                input_data.append(data)
                np.save(os.path.join(real_dump_path, f'{input.name}.npy'), data)
            # run model inference
            graph.dump(input_data, real_dump_path)

    def remove_dump_data(self, dump_num, dump_path = 'dump'):
        for j in range(dump_num):
            real_dump_path = f'{dump_path}{j}'
            if os.path.exists(real_dump_path):
                os.removedirs(real_dump_path)

    def get_node_inout_shapes_from_dump_data(self, graph: BaseGraph, reshape: Node, dump_num, dump_path = 'dump'):
        # check reshape type
        prev_node = graph.get_prev_node(reshape.inputs[0])
        # get reshape input and output shapes from all dump data
        in_shapes, out_shapes = [], []
        for i in range(dump_num):
            real_dump_path = f'{dump_path}{i}'
            if prev_node.op_type == 'PlaceHolder':
                dump_file = f'{prev_node.name}.npy'
            elif prev_node.op_type != 'Initializer':
                dump_file = f'{prev_node.name}_{prev_node.get_output_id(reshape.inputs[0])}.npy'
            else:
                raise RuntimeError('Reshape prev node is Constant type.')
            reshape_input = np.load(os.path.join(real_dump_path, dump_file))
            reshape_output = np.load(os.path.join(real_dump_path, f'{reshape.inputs[0]}_0.npy'))
            in_shapes.append(reshape_input.shape)
            out_shapes.append(reshape_output.shape)
        return np.array(in_shapes), np.array(out_shapes)

    def calculate_shape(self, shape, in_shapes, out_shapes):
        insert = { 'squeeze': [], 'unsqueeze': [] }
        in_dim = 0
        for out_dim in range(len(shape)):
            if not shape[out_dim] is None:
                continue
            # the dim is dynamic
            while in_dim < in_shapes.shape[1]:
                if np.all(out_shapes[:, out_dim] == in_shapes[:, in_dim]):
                    break
                in_dim += 1
            if in_dim == in_shapes.shape[1]:
                break
            if out_dim == in_dim:
                # the dim has no change
                #                Reshape(-1, 0, 32)
                # (bs, len, 256) ------------------> (8*bs, len, 32)
                shape[out_dim] = 0
            elif out_dim < in_dim:
                #                  Reshape(-1, 1, 0, 32)                     Squeeze
                # (bs, 8, len, 32) ---------------------> (8*bs, 1, len, 32) -------> (8*bs, len, 32)
                shape.insert(out_dim, 1)
                shape[out_dim] = 0
                insert['squeeze'].append(out_dim)
            else:
                #                 Unsqueeze                     Reshape(-1, 8, 0, 32)
                # (8*bs, len, 32) ---------> (8*bs, 1, len, 32) ---------------------> (bs, 8, len, 32)
                shape[out_dim] = 0
                insert['unsqueeze'].append(in_dim)
            # compute next dimension
            in_dim += 1
        # if exist two or more dynamic dimension, then will not be optimized.
        if shape.count(None) <= 1:
            return insert, [dim if dim != None else -1 for dim in shape]
        else:
            return None, None

    def optimize_reshape(self, graph: BaseGraph, dump_num, dump_path = 'dump'):
        optimize_result = False
        for reshape in graph.get_nodes('Reshape'):
            # get 'Reshape' input and output shape
            in_shapes, out_shapes = self.get_node_inout_shapes_from_dump_data(graph, reshape, dump_num, dump_path)

            # optimize Reshape operator
            shape = [out_shapes[0][i] if is_constant else None
                for i, is_constant in enumerate(np.all(out_shapes == out_shapes[0,:], axis = 0))]
            insert, shape = self.calculate_shape(shape, in_shapes, out_shapes)
            if insert is None or shape is None:
                continue
            if len(insert['unsqueeze']) != 0:
                attrs = {'axes': np.array(insert['unsqueeze'], dtype = np.int64)}
                insert_unsqueeze(graph, reshape, attrs, mode = 'before', refer_index = 0)
            if len(insert['squeeze']) != 0:
                attrs = {'axes': np.array(insert['squeeze'], dtype = np.int64)}
                insert_squeeze(graph, reshape, attrs, mode = 'after', refer_index = 0)

            # add constant shape for Reshape operator
            graph.add_initializer(f'Shape_for_{reshape.name}', np.array(shape))
            reshape.inputs[1] = f'Shape_for_{reshape.name}'

            optimize_result = True
        return optimize_result

    def optimize_apply(self, graph: BaseGraph, match_result: MatchResult):
        if match_result.is_empty():
            return False
        # check model is dynamic and get dynamic input name
        dynamic_axes = self.get_model_dynamic_axes(graph)
        if len(dynamic_axes) == 0:
            print('The model is static type, no need to optimize Reshape.')
            return False

        # run model inference and generate operator dump, the purpose is to obtain the input and output shapes
        dump_num = 5 # dump model for 5 time
        self.generate_dump_data(graph, dump_num, dynamic_axes)

        # optimize Reshape operator
        optimize_result = self.optimize_reshape(graph)

        # release temp resource
        self.remove_dump_data(dump_num)
        return optimize_result

if __name__ == '__main__':
    import sys
    onnx_file = sys.argv[1]
    from auto_optimizer import OnnxGraph
    graph = OnnxGraph.parse(onnx_file)
    knowledge = KnowledgeDynamicReshape()
    if knowledge.optimize_apply(graph, None):
        input_names = [i.name for i in graph.inputs]
        output_names = [o.name for o in graph.outputs]
        graph.extract(f'{onnx_file}_opt.onnx', input_names, output_names)

