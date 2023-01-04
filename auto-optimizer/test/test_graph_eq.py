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

import copy
import unittest

import numpy as np

from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.graph_refactor.onnx.node import (
    OnnxPlaceHolder, OnnxInitializer, OnnxNode
)
from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from helper import KnowledgeTestHelper


def create_test_graph(name: str = 'test_graph') -> BaseGraph:
    in0 = OnnxPlaceHolder(name='in0', dtype=np.dtype('float32'), shape=[2, 3])
    in1 = OnnxPlaceHolder(name='in1', dtype=np.dtype('float32'), shape=[2, 3])
    out0 = OnnxPlaceHolder(name='out0', dtype=np.dtype('float32'), shape=[2, 3])
    out1 = OnnxPlaceHolder(name='out1', dtype=np.dtype('float32'), shape=[2, 3])
    ini0 = OnnxInitializer(name='ini0', value=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    ini1 = OnnxInitializer(name='ini1', value=np.array([[6, 5, 4], [3, 2, 1]], dtype=np.float32))
    op0 = OnnxNode(name='add', op_type='Add', inputs=['in0', 'ini0'], outputs=['o0'], attrs={})
    op1 = OnnxNode(name='mul', op_type='Mul', inputs=['in1', 'o0'], outputs=['o1'], attrs={})
    op2 = OnnxNode(name='elu', op_type='Elu', inputs=['o1'], outputs=['out0'], attrs={'alpha': 1.0})
    op3 = OnnxNode(name='sum', op_type='Sum', inputs=['ini1', 'o0', 'o1'], outputs=['out1'], attrs={})
    return OnnxGraph(
        name=name,
        nodes=[op0, op1, op2, op3],
        inputs=[in0, in1],
        outputs=[out0, out1],
        initializers=[ini0, ini1]
    )


class TestGraphEq(unittest.TestCase):

    def test_place_holder_eq(self):
        ph0 = OnnxPlaceHolder(name='ph', dtype=np.dtype('float32'), shape=[1])
        ph1 = OnnxPlaceHolder(name='ph', dtype=np.dtype('float32'), shape=[1])
        self.assertEqual(ph0, ph1)

        ph2 = OnnxPlaceHolder(name='xxx', dtype=np.dtype('float32'), shape=[1])
        self.assertNotEqual(ph0, ph2)
        ph3 = OnnxPlaceHolder(name='ph', dtype=np.dtype('float64'), shape=[1])
        self.assertNotEqual(ph0, ph3)
        ph4 = OnnxPlaceHolder(name='ph', dtype=np.dtype('float32'), shape=[1, 2])
        self.assertNotEqual(ph0, ph4)
        ph5 = OnnxPlaceHolder(name='ph', dtype=np.dtype('float32'), shape=['1'])
        self.assertNotEqual(ph0, ph5)

    def test_initializer_eq(self):
        init0 = OnnxInitializer(name='init', value=np.array([1.3, 2, 3], dtype=np.float32))
        init1 = OnnxInitializer(name='init', value=np.array([1.3, 2, 3], dtype=np.float32))
        self.assertEqual(init0, init1)

        init2 = OnnxInitializer(name='xx', value=np.array([1.3, 2, 3], dtype=np.float32))
        self.assertNotEqual(init0, init2)
        init3 = OnnxInitializer(name='init', value=np.array([1.3, 2], dtype=np.float32))
        self.assertNotEqual(init0, init3)
        init4 = OnnxInitializer(name='init', value=np.array([1.3, 2, 3], dtype=np.float64))
        self.assertNotEqual(init0, init4)

    def test_node_eq(self):
        node0 = OnnxNode(name='n', op_type='Add', inputs=['i0', 'p0'], outputs=['o'], attrs={}, domain='ai.onnx')
        node1 = OnnxNode(name='n', op_type='Add', inputs=['i0', 'p0'], outputs=['o'], attrs={}, domain='ai.onnx')
        self.assertEqual(node0, node1)

        node2 = OnnxNode(name='x', op_type='Add', inputs=['i0', 'p0'], outputs=['o'], attrs={}, domain='ai.onnx')
        self.assertNotEqual(node0, node2)
        node3 = OnnxNode(name='n', op_type='Mul', inputs=['i0', 'p0'], outputs=['o'], attrs={}, domain='ai.onnx')
        self.assertNotEqual(node0, node3)
        node4 = OnnxNode(name='n', op_type='Add', inputs=['p0', 'i0'], outputs=['o'], attrs={}, domain='ai.onnx')
        self.assertNotEqual(node0, node4)
        node5 = OnnxNode(name='n', op_type='Add', inputs=['i0', 'p0'], outputs=['x'], attrs={}, domain='ai.onnx')
        self.assertNotEqual(node0, node5)
        node6 = OnnxNode(name='n', op_type='Add', inputs=['i0', 'p0'], outputs=['o'], attrs={'x': 0}, domain='ai.onnx')
        self.assertNotEqual(node0, node6)
        node7 = OnnxNode(name='n', op_type='Add', inputs=['i0', 'p0'], outputs=['o'], attrs={}, domain='')
        self.assertNotEqual(node0, node7)

        node8 = OnnxNode(name='n', op_type='Split', inputs=['i0'], outputs=['o0', 'o1'], attrs={'axis': 0, 'split': [1, 1]})
        node9 = OnnxNode(name='n', op_type='Split', inputs=['i0'], outputs=['o1', 'o0'], attrs={'axis': 0, 'split': [1, 1]})
        self.assertNotEqual(node8, node9)

    def test_graph_eq(self):
        graph0 = create_test_graph()
        graph1 = copy.deepcopy(graph0)
        self.assertTrue(KnowledgeTestHelper.graph_equal(graph0, graph1))

        # reverse save order shouldn't matter
        graph1.inputs.reverse()
        self.assertTrue(KnowledgeTestHelper.graph_equal(graph0, graph1))
        graph1.outputs.reverse()
        self.assertTrue(KnowledgeTestHelper.graph_equal(graph0, graph1))
        graph1.initializers.reverse()
        self.assertTrue(KnowledgeTestHelper.graph_equal(graph0, graph1))
        graph1.nodes.reverse()
        self.assertTrue(KnowledgeTestHelper.graph_equal(graph0, graph1))
        graph1.infershape()
        self.assertTrue(KnowledgeTestHelper.graph_equal(graph0, graph1))

        # reverse operator input order should count as difference
        graph2 = copy.deepcopy(graph0)
        graph2['add'].inputs.reverse()
        self.assertFalse(KnowledgeTestHelper.graph_equal(graph0, graph2))

        graph3 = copy.deepcopy(graph0)
        graph3['ini0'].value = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        self.assertFalse(KnowledgeTestHelper.graph_equal(graph0, graph3))

        graph4 = copy.deepcopy(graph0)
        graph4['elu'].attrs['alpha'] = 0.5
        self.assertFalse(KnowledgeTestHelper.graph_equal(graph0, graph4))

        graph5 = copy.deepcopy(graph0)
        graph5.add_input(name='in2', dtype=np.dtype('float32'), shape=[2, 3])
        graph5['add'].inputs[1] = 'in2'
        self.assertFalse(KnowledgeTestHelper.graph_equal(graph0, graph5))


if __name__ == '__main__':
    unittest.main()
