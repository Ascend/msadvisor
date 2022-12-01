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

import os
import random

import numpy as np
import onnx
from onnx import (
    helper,
    TensorProto,
)

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges.knowledge_split_qkv_matmul import KnowledgeSplitQKVMatmul
from utils import inference, optimize


def make_basic_qkv_matmul_model(onnx_name, perm, gathers=3, axis=0, ops1=1, ops2=1,
                                valid_gather=True, valid_reshape=True) -> bool:
    if gathers not in [2, 3, 4, 5, 6, 7, 8]:
        return False
    input_ = helper.make_tensor_value_info("input", TensorProto.FLOAT, (1, 10, 112))
    output_ = helper.make_tensor_value_info("output", TensorProto.FLOAT, (10, 10))

    inits, vinfo, nodes = [], [], []
    supported_ops = ["Add", "Sub", "Mul", "Div"]
    idx = 0
    last_output = "input"

    for _ in range(ops1):
        _op = random.choice(supported_ops)
        _op_idx = idx
        idx += 1
        _weight = f"{_op}{_op_idx}_w"
        _output = f"{_op}{_op_idx}_o"
        _name = f"{_op}_{_op_idx}"
        _input = [last_output, _weight]
        random.shuffle(_input)
        inits.append(
            helper.make_tensor(_weight, TensorProto.FLOAT, [112], np.random.rand(112).astype(np.float32) + 0.5)
        )
        nodes.append(helper.make_node(_op, _input, [_output], _name))
        last_output = _output

    _op = "MatMul"
    _op_idx = idx
    idx += 1
    _weight = f"{_op}{_op_idx}_w"
    _output = f"{_op}{_op_idx}_o"
    _name = f"{_op}_{_op_idx}"
    inits.append(
        helper.make_tensor(_weight, TensorProto.FLOAT, [112, 8400], np.random.rand(112, 8400).astype(np.float32) + 0.5)
    )
    nodes.append(helper.make_node(_op, [last_output, _weight], [_output], _name))
    last_output = _output

    for _ in range(ops2):
        _op = random.choice(supported_ops)
        _op_idx = idx
        idx += 1
        _weight = f"{_op}{_op_idx}_w"
        _output = f"{_op}{_op_idx}_o"
        _name = f"{_op}_{_op_idx}"
        _input = [last_output, _weight]
        random.shuffle(_input)
        inits.append(
            helper.make_tensor(_weight, TensorProto.FLOAT, [8400], np.random.rand(8400).astype(np.float32) + 0.5)
        )
        nodes.append(helper.make_node(_op, _input, [_output], _name))
        last_output = _output

    _op = "Reshape"
    _op_idx = idx
    idx += 1
    _weight = f"{_op}{_op_idx}_w"
    _output = f"{_op}{_op_idx}_o"
    _name = f"{_op}_{_op_idx}"
    _shape = [1, 10, gathers, 840 // gathers, 10] if valid_reshape else [1, 20, gathers, 420 // gathers, 10]

    inits.append(helper.make_tensor(_weight, TensorProto.INT64, [5], np.array(_shape, dtype=np.int64)))
    nodes.append(helper.make_node(_op, [last_output, _weight], [_output], _name))
    last_output = _output

    _op = "Transpose"
    _op_idx = idx
    idx += 1
    _output = f"{_op}{_op_idx}_o"
    _name = f"{_op}_{_op_idx}"
    nodes.append(helper.make_node(_op, [last_output], [_output], _name, perm=perm))
    last_output = _output

    indices = [i if valid_gather else i * (i - 1) for i in range(gathers)]
    random.shuffle(indices)
    gather_outputs = []
    for i in range(gathers):
        _op = "Gather"
        _weight = f"{_op}{i}_w"
        _output = f"{_op}{i}_o"
        _name = f"{_op}_{i}"
        inits.append(helper.make_tensor(_weight, TensorProto.INT64, [], np.array([indices[i]], dtype=np.int64)))
        nodes.append(helper.make_node(_op, [last_output, _weight], [_output], _name, axis=axis))
        gather_outputs.append(_output)

    last = gather_outputs[0]
    for out in gather_outputs[1:-1]:
        _op = random.choice(supported_ops)
        _op_idx = idx
        idx += 1
        _output = f"{_op}{_op_idx}_o"
        _name = f"{_op}_{_op_idx}"
        nodes.append(helper.make_node(_op, [last, out], [_output], _name))
        last = _output

    _op = "Transpose"
    _op_idx = idx
    idx += 1
    _output = f"{_op}{_op_idx}_o"
    _name = f"{_op}_{_op_idx}"
    _perm = [0, 1, 2, 3]
    random.shuffle(_perm)
    nodes.append(helper.make_node(_op, [gather_outputs[-1]], [_output], _name, perm=_perm))
    last_output = _output

    out0, out1 = last, last_output

    _op = "Reshape"
    _op_idx = idx
    idx += 1
    _weight = f"{_op}{_op_idx}_w"
    _output = f"{_op}{_op_idx}_o"
    _name = f"{_op}_{_op_idx}"
    _shape = [10, 8400 // gathers]

    inits.append(helper.make_tensor(_weight, TensorProto.INT64, [2], np.array(_shape, dtype=np.int64)))
    nodes.append(helper.make_node(_op, [out0, _weight], [_output], _name))
    out0 = _output

    _op = "Reshape"
    _op_idx = idx
    idx += 1
    _weight = f"{_op}{_op_idx}_w"
    _output = f"{_op}{_op_idx}_o"
    _name = f"{_op}_{_op_idx}"
    _shape = [8400 // gathers, 10]

    inits.append(helper.make_tensor(_weight, TensorProto.INT64, [2], np.array(_shape, dtype=np.int64)))
    nodes.append(helper.make_node(_op, [out1, _weight], [_output], _name))
    out1 = _output

    _op = "MatMul"
    _op_idx = idx
    idx += 1
    _name = f"{_op}_{_op_idx}"
    nodes.append(helper.make_node(_op, [out0, out1], ["output"], _name))

    graph = helper.make_graph(nodes, "qkv_slice_test", [input_], [output_], inits, value_info=vinfo)
    model = helper.make_model(graph)

    del model.opset_import[:]
    opset = model.opset_import.add()
    opset.domain = ''
    opset.version = 14
    onnx.save(model, onnx_name)
    return True


class TestKnowledgeSplitQKVMatmul(unittest.TestCase):

    def test_basic_qkv_slice(self):
        params = [
            # RES,  PERM,            G, A, S1  S2  valid_gathers valid_reshape
            (True,  [2, 0, 1, 3, 4], 2, 0, 1,  1,  True,         True),
            (True,  [2, 0, 3, 1, 4], 2, 0, 1,  1,  True,         True),
            (True,  [2, 0, 3, 1, 4], 2, 0, 1,  0,  True,         True),
            (True,  [2, 0, 3, 1, 4], 2, 0, 0,  1,  True,         True),
            (True,  [2, 0, 3, 1, 4], 2, 0, 0,  0,  True,         True),
            # invalid gather nodes
            (False, [2, 0, 3, 1, 4], 2, 0, 1,  1,  False,        True),
            # invalid reshape node
            (False, [2, 0, 3, 1, 4], 2, 0, 1,  1,  True,         False),
            (True,  [2, 0, 3, 1, 4], 3, 0, 1,  1,  True,         True),
            (True,  [2, 0, 3, 1, 4], 4, 0, 4,  1,  True,         True),
            (True,  [2, 0, 3, 1, 4], 5, 0, 2,  5,  True,         True),
            (True,  [2, 0, 3, 1, 4], 6, 0, 10, 12, True,         True),
            (True,  [2, 0, 3, 1, 4], 7, 0, 2,  3,  True,         True),
            (True,  [2, 0, 3, 1, 4], 8, 0, 3,  4,  True,         True),
            # gather operator not pick from the first axis
            (False, [2, 0, 3, 1, 4], 3, 2, 1,  1,  True,         True),
            # invalid permutation
            (False, [1, 2, 3, 0, 4], 3, 0, 1,  1,  True,         True),
        ]
        if not os.path.exists("./onnx"):
            os.mkdir("./onnx")
        for expect, perm, gathers, axis, s1, s2, vg, vr in params:
            pstr = ''.join(str(k) for k in perm)
            name = f"qkv_slice_p{pstr}_g{gathers}_a{axis}_s_{s1}_{s2}_g{int(vg)}_r{int(vr)}"
            with self.subTest(name):
                onnx_path = f"./onnx/{name}.onnx"
                optimize_onnx_path = f"./onnx/{name}_optimize.onnx"

                ok = make_basic_qkv_matmul_model(
                    onnx_path,
                    perm=perm,
                    gathers=gathers,
                    axis=axis,
                    ops1=s1,
                    ops2=s2,
                    valid_gather=vg,
                    valid_reshape=vr
                )
                self.assertTrue(ok)
                if not ok:
                    continue

                graph = OnnxGraph.parse(onnx_path)

                knowledge = KnowledgeSplitQKVMatmul()
                result = optimize(graph, knowledge)
                self.assertEqual(result, expect)
                if not result:
                    continue
                graph.save(optimize_onnx_path)

                input_ = np.random.rand(1, 10, 112).astype(np.float32) + 0.5
                matrix_before_apply = inference(onnx_path, [input_])
                matrix_after_apply = inference(optimize_onnx_path, [input_])
                self.assertTrue(len(matrix_before_apply) == len(matrix_after_apply))
                for lmatrix, rmatrix in zip(matrix_before_apply, matrix_after_apply):
                    self.assertTrue(np.allclose(lmatrix, rmatrix, atol=1e-4, rtol=1e-2))

                result = optimize(graph, knowledge)
                self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
