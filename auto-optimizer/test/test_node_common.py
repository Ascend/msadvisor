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
from typing import cast

import numpy as np
from onnx import helper, numpy_helper

from auto_optimizer.graph_refactor.onnx.node import OnnxPlaceHolder, OnnxInitializer, OnnxNode

try:
    np_dtype_to_tensor_dtype = helper.np_dtype_to_tensor_dtype
    
except AttributeError as e:
    from onnx import mapping

    def np_dtype_to_tensor_dtype(np_dtype: np.dtype) -> int:
        return cast(int, mapping.NP_TYPE_TO_TENSOR_TYPE[np_dtype])


def is_node_equal(node1, node2, msg=None):
    ret = node1.name == node2.name and \
        node1.op_type == node2.op_type and \
        node1.inputs == node2.inputs and \
        node1.outputs == node2.outputs and \
        node1.attrs == node2.attrs and \
        node1.domain == node2.domain
    if not ret:
        msg = 'two nodes are not equal!'
        raise unittest.TestCase.failureException(msg)
    return ret


def is_ph_equal(ph1, ph2, msg=None):
    ret = ph1.name == ph2.name and \
        ph1.dtype == ph2.dtype and \
        ph1.shape == ph2.shape
    if not ret:
        msg = 'two nodes are not equal!'
        raise unittest.TestCase.failureException(msg)
    return ret


def is_ini_equal(ini1, ini2, msg=None):
    ret = ini1.name == ini2.name and \
        np.array_equal(ini1.value, ini2.value, equal_nan=True) and \
        ini1.value.dtype == ini2.value.dtype
    if not ret:
        msg = 'two nodes are not equal!'
        raise unittest.TestCase.failureException(msg)
    return ret


def create_node(node_type):
    if node_type == 'OnnxNode':
        return OnnxNode('test_node', 'Conv', ['0', '1'], ['2'], attrs={'kernel_shape': 3}, domain='')
    if node_type == 'OnnxPlaceHolder':
        return OnnxPlaceHolder('test_ph', np.dtype('float32'), [1, 3, 224, 224])
    if node_type == 'OnnxInitializer':
        return OnnxInitializer('test_ini', np.array([[1, 2, 3, 4, 5]], dtype='int32'))


class TestNodeCommon(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(OnnxNode, is_node_equal)
        self.addTypeEqualityFunc(OnnxPlaceHolder, is_ph_equal)
        self.addTypeEqualityFunc(OnnxInitializer, is_ini_equal)

    def test_create_node(self):
        node = OnnxNode('test_node', 'Conv', ['0', '1'], ['2'], attrs={'kernel_shape': 3}, domain='')
        self.assertIsInstance(node, OnnxNode)

    def test_create_placeholder(self):
        ph = OnnxPlaceHolder('test_ph', np.dtype('float32'), [1, 3, 224, 224])
        self.assertIsInstance(ph, OnnxPlaceHolder)

    def test_creaete_initializer(self):
        ini = OnnxInitializer('test_ini', np.array([[1, 2, 3, 4, 5]], dtype='int32'))
        self.assertIsInstance(ini, OnnxInitializer)

    def test_node_parse(self):
        node_proto = helper.make_node('Conv', ['0', '1'], ['2'], 'test_node', domain='', kernel_shape=3)
        test_node = OnnxNode.parse(node_proto)
        self.assertEqual(test_node, create_node('OnnxNode'))

    def test_node_to_proto(self):
        node = create_node('OnnxNode')
        test_proto = OnnxNode.proto(node)
        self.assertEqual(test_proto, helper.make_node('Conv', ['0', '1'], [
                         '2'], 'test_node', domain='', kernel_shape=3))

    def test_initializer_parse(self):
        test_proto = helper.make_tensor(
            'test_ini', np_dtype_to_tensor_dtype(np.dtype('int32')), (1, 5), np.array([1, 2, 3, 4, 5]))
        test_init = OnnxInitializer.parse(test_proto)
        self.assertEqual(test_init, create_node('OnnxInitializer'))

    def test_initializer_to_proto(self):
        ini = create_node('OnnxInitializer')
        test_proto = OnnxInitializer.proto(ini)
        tensor_proto = numpy_helper.from_array(ini.value)
        tensor_proto.name = ini.name
        self.assertEqual(test_proto, tensor_proto)

    def test_placeholder_parse(self):
        test_proto = helper.make_tensor_value_info(
            'test_ph', np_dtype_to_tensor_dtype(np.dtype('float32')), (1, 3, 224, 224))
        test_ph = OnnxPlaceHolder.parse(test_proto)
        self.assertEqual(test_ph, create_node('OnnxPlaceHolder'))

    def test_placeholder_to_proto(self):
        ph = create_node('OnnxPlaceHolder')
        test_proto = OnnxPlaceHolder.proto(ph)
        self.assertEqual(test_proto, helper.make_tensor_value_info(
            'test_ph', np_dtype_to_tensor_dtype(np.dtype('float32')), (1, 3, 224, 224)))


if __name__ == "__main__":
    unittest.main()
