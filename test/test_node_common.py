import unittest

import numpy as np
from onnx import helper
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from auto_optimizer.graph_refactor.onnx.node import PlaceHolder, Initializer, Node

def is_node_equal(node1, node2, msg=None):
    return node1.name == node2.name and \
        node1.op_type == node2.op_type and \
        node1.inputs == node2.inputs and \
        node1.outputs == node2.outputs and \
        node1.attrs == node2.attrs and \
        node1.domain == node2.domain

def is_ph_equal(ph1, ph2, msg=None):
    return ph1.name == ph2.name and \
        ph1.dtype == ph2.dtype and \
        ph1.shape == ph2.shape

def is_ini_equal(ini1, ini2, msg=None):
    return ini1.name == ini2.name and \
        np.array_equal(ini1.value, ini2.value, equal_nan=True) and \
        ini1.value.dtype == ini2.value.dtype

def create_node(node_type):
    if node_type == 'Node':
        return Node('test_node', 'Conv', ['0', '1'], ['2'], attrs={'kernel_shape': 3}, domain='')
    if node_type == 'PlaceHolder':
        return PlaceHolder('test_ph', np.dtype('float32'), [1,3,224,224])
    if node_type == 'Initializer':
        return Initializer('test_ini', np.array([[1,2,3,4,5]], dtype='int32'))

class TestNodeCommon(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(Node, is_node_equal)
        self.addTypeEqualityFunc(PlaceHolder, is_ph_equal)
        self.addTypeEqualityFunc(Initializer, is_ini_equal)

    def test_create_node(self):
        node = Node('test_node', 'Conv', ['0', '1'], ['2'], attrs={'kernel_shape': 3}, domain='')
        self.assertIsInstance(node, Node)

    def test_create_placeholder(self):
        ph = PlaceHolder('test_ph', np.dtype('float32'), [1,3,224,224])
        self.assertIsInstance(ph, PlaceHolder)
    
    def test_creaete_initializer(self):
        ini = Initializer('test_ini', np.array([[1,2,3,4,5]], dtype='int32'))
        self.assertIsInstance(ini, Initializer)

    def test_node_parse(self):
        node_proto = helper.make_node('Conv', ['0', '1'], ['2'], 'test_node', domain='', kernel_shape=3)
        test_node = Node.parse(node_proto)
        self.assertEqual(test_node, create_node('Node'))

    def test_node_to_proto(self):
        node = create_node('Node')
        test_proto = Node.to_proto(node)
        self.assertEqual(test_proto, helper.make_node('Conv', ['0', '1'], ['2'], 'test_node', domain='', kernel_shape=3))

    def test_initializer_parse(self):
        test_proto = helper.make_tensor('test_ini', NP_TYPE_TO_TENSOR_TYPE[np.dtype('int32')], (1,5), np.array([1,2,3,4,5]))
        test_init = Initializer.parse(test_proto)
        self.assertEqual(test_init, create_node('Initializer'))

    def test_initializer_to_proto(self):
        ini = create_node('Initializer')
        test_proto = Initializer.to_proto(ini)
        self.assertEqual(test_proto, helper.make_tensor('test_ini', NP_TYPE_TO_TENSOR_TYPE[np.dtype('int32')], (1,5), np.array([1,2,3,4,5])))

    def test_placeholder_parse(self):
        test_proto = helper.make_tensor_value_info('test_ph', NP_TYPE_TO_TENSOR_TYPE[np.dtype('float32')], (1,3,224,224))
        test_ph = PlaceHolder.parse(test_proto)
        self.assertEqual(test_ph, create_node('PlaceHolder'))

    def test_placeholder_to_proto(self):
        ph = create_node('PlaceHolder')
        test_proto = PlaceHolder.to_proto(ph)
        self.assertEqual(test_proto, helper.make_tensor_value_info('test_ph', NP_TYPE_TO_TENSOR_TYPE[np.dtype('float32')], (1,3,224,224)))

if __name__ == "__main__":
    unittest.main()