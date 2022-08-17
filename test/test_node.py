import unittest

from onnx import helper

from auto_optimizer.graph_refactor.onnx.node import Node
from test_node_common import create_node

class TestNode(unittest.TestCase):

    def test_node_op_type(self):
        node = create_node('Node')
        self.assertEqual(node.op_type, 'Conv')

    def test_node_get_inputs(self):
        node = create_node('Node')
        self.assertEqual(node.inputs, ['0', '1'])

    def test_node_set_inputs(self):
        node = create_node('Node')
        node.inputs = ['0', '3']
        self.assertEqual(node.inputs, ['0', '3'])

    def test_node_get_input_id(self):
        node = create_node('Node')
        self.assertEqual(node.get_input_id('0'), 0)
        self.assertEqual(node.get_input_id('1'), 1)

    def test_node_get_outputs(self):
        node = create_node('Node')
        self.assertEqual(node.outputs, ['2'])

    def test_node_set_outputs(self):
        node = create_node('Node')
        node.outputs = ['5']
        self.assertEqual(node.outputs, ['5'])

    def test_node_get_input_id(self):
        node = create_node('Node')
        self.assertEqual(node.get_output_id('2'), 0)

    def test_node_get_attrs(self):
        node = create_node('Node')
        self.assertEqual(node.attrs, {'kernel_shape': 3})
    
    def test_node_get_attr(self):
        node = create_node('Node')
        self.assertEqual(node['kernel_shape'], 3)

    def test_node_set_attr(self):
        node = create_node('Node')
        node['kernel_shape'] = 5
        self.assertEqual(node['kernel_shape'], 5)

    def test_node_set_attr(self):
        node = create_node('Node')
        node['kernel_shape'] = 5
        self.assertEqual(node['kernel_shape'], 5)

    def test_node_domain(self):
        node = create_node('Node')
        self.assertEqual(node.domain, '')

if __name__ == "__main__":
    unittest.main()