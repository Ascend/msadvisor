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

import unittest
import numpy as np
from onnx import TensorProto

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges.knowledge_merge_casts import KnowledgeMergeCasts
from utils import optimize


class TestKnowledgeMergeCasts(unittest.TestCase):
    def test_merge_brother_casts(self):
        onnx_name = 'merge_casts_test_merge_brother_casts'
        x = np.random.randn(10, 10).astype(np.float32)
        graph = OnnxGraph(name=onnx_name)
        graph.add_input('X', x.dtype, x.shape)
        graph.add_initializer('Add_value', np.random.randn(*x.shape).astype(x.dtype))
        graph.add_node('Add0', 'Add', ['X', 'Add_value'], ['Add_O'])
        graph.add_node('Cast1', 'Cast', ['Add_O'], ['Cast_O1'], attrs={'to': TensorProto.INT32})
        graph.add_node('Cast2', 'Cast', ['Add_O'], ['Cast_O2'], attrs={'to': TensorProto.INT32})
        graph.add_node('Cast3', 'Cast', ['Add_O'], ['Cast_O3'], attrs={'to': TensorProto.INT64})
        graph.update_map()
        graph.infershape()

        knowledge = KnowledgeMergeCasts()
        result = optimize(graph, knowledge)
        self.assertTrue(result)
        next_nodes = graph.get_next_nodes('Add_O')
        self.assertTrue(len(next_nodes) == 2)

    def test_transfer_cast_to_root(self):
        onnx_name = 'merge_casts_test_transfer_cast_to_root'
        x = np.random.randn(10, 10).astype(np.float32)
        graph = OnnxGraph(name=onnx_name)
        graph.add_input('X', x.dtype, x.shape)
        graph.add_initializer('Add_value', np.random.randn(*x.shape).astype(x.dtype))
        graph.add_node('Add0', 'Add', ['X', 'Add_value'], ['Add_O1'])
        graph.add_node('Cast1', 'Cast', ['Add_O1'], ['Cast_O1'], attrs={'to': TensorProto.INT64})
        graph.add_node('Cast2', 'Cast', ['Add_O1'], ['Cast_O2'], attrs={'to': TensorProto.INT32})
        graph.add_node('Cast3', 'Cast', ['Cast_O1'], ['Cast_O3'], attrs={'to': TensorProto.INT32})
        graph.add_initializer('Add_value2', x.astype(np.int64))
        graph.add_node('Add1', 'Add', ['Cast_O1', 'Add_value2'], ['Add_O2'])
        graph.update_map()
        graph.infershape()

        knowledge = KnowledgeMergeCasts()
        result = optimize(graph, knowledge)
        self.assertTrue(result)
        next_nodes = graph.get_next_nodes('Add_O1')
        self.assertTrue(len(next_nodes) == 2)
        self.assertTrue(all(map(lambda n: n.op_type == 'Cast', next_nodes)))
        next_nodes = graph.get_next_nodes('Cast_O1')
        self.assertTrue(len(next_nodes) == 1)
        self.assertTrue(next_nodes[0].op_type == 'Add')

    def test_remove_parent_cast(self):
        onnx_name = 'merge_casts_test_remove_parent_cast'
        x = np.random.randn(10, 10).astype(np.float32)
        graph = OnnxGraph(name=onnx_name)
        graph.add_input('X', x.dtype, x.shape)
        graph.add_initializer('Add_value', np.random.randn(*x.shape).astype(x.dtype))
        graph.add_node('Add0', 'Add', ['X', 'Add_value'], ['Add_O1'])
        graph.add_node('Cast1', 'Cast', ['Add_O1'], ['Cast_O1'], attrs={'to': TensorProto.INT64})
        graph.add_node('Cast2', 'Cast', ['Cast_O1'], ['Cast_O2'], attrs={'to': TensorProto.INT32})
        graph.update_map()
        graph.infershape()

        knowledge = KnowledgeMergeCasts()
        result = optimize(graph, knowledge)
        self.assertTrue(result)
        next_nodes = graph.get_next_nodes('Add_O1')
        self.assertTrue(len(next_nodes) == 1)
        self.assertTrue(next_nodes[0].name == 'Cast2')

    def test_remove_cast_after_root(self):
        onnx_name = 'merge_casts_test_remove_cast_after_root'
        x = np.random.randn(10, 10).astype(np.int32)
        graph = OnnxGraph(name=onnx_name)
        graph.add_input('X', x.dtype, x.shape)
        graph.add_initializer('Add_value', np.random.randn(*x.shape).astype(x.dtype))
        graph.add_node('Add0', 'Add', ['X', 'Add_value'], ['Add_O1'])
        graph.add_node('Cast1', 'Cast', ['Add_O1'], ['Cast_O1'], attrs={'to': TensorProto.INT64})
        graph.add_node('Cast2', 'Cast', ['Add_O1'], ['Cast_O2'], attrs={'to': TensorProto.INT32})
        graph.update_map()
        graph.infershape()

        knowledge = KnowledgeMergeCasts()
        result = optimize(graph, knowledge)
        self.assertTrue(result)
        next_nodes = graph.get_next_nodes('Add_O1')
        self.assertTrue(len(next_nodes) == 1)
        self.assertTrue(next_nodes[0].name == 'Cast1')


if __name__ == '__main__':
    unittest.main()
    TestKnowledgeMergeCasts().test_merge_brother_casts()
    print('aaa')