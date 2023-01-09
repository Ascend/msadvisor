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
from inspect import signature
from functools import wraps

import numpy as np
from numpy.typing import NDArray
from numpy.linalg import norm
import onnxruntime as rt

def typeassert(*ty_args, **ty_kwargs):
    def decorate(func):
        # Map function argument names to supplied types
        sig = signature(func)
        bound_types = sig.bind_partial(*ty_args, **ty_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            # Enforce type assertions across supplied arguments
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError(
                            f'Argument {name} must be {bound_types[name]}'
                        )
            return func(*args, **kwargs)
        return wrapper
    return decorate


@typeassert(path=str)
def format_to_module(path):
    """
    路径转换，把文件相对路径转换成python的import路径
    """
    format_path = ""
    if path.endswith(".py"):
        # [:-3]： 删除.py
        format_path = path.replace(os.sep, ".")[:-3]

    if format_path.startswith("."):
        format_path = format_path.replace(".", "", 1)

    return format_path


def check_file_exist(file, msg='file "{}" does not exist'):
    if not os.path.isfile(file):
        raise FileNotFoundError(msg.format(file))


def dump_op_outputs(graph, input_data, dump_path, outputs=[]):
    def _run(model, input_data):
        sess = rt.InferenceSession(model)
        inputs = [ipt.name for ipt in sess.get_inputs()]
        outputs = [out.name for out in sess.get_outputs()]
        ret = sess.run(outputs, {name: data for name, data in zip(inputs, input_data)})
        return ret

    from skl2onnx.helpers.onnx_helper import (select_model_inputs_outputs,
                                                enumerate_model_node_outputs)

    ori_model = graph.model()
    if len(outputs) == 0:
        outputs = [
            name for name in enumerate_model_node_outputs(ori_model)]
    new_model = select_model_inputs_outputs(ori_model, outputs)
    new_model_byte = new_model.SerializeToString()
    arrs = _run(new_model_byte, input_data)
    idx = 0
    if not os.path.exists(dump_path):
        os.makedirs(dump_path, mode=0o700)
    for node in ori_model.graph.node:
        for i, output in enumerate(node.output):
            fname = f'{node.name}_{i}.npy'
            np.save(os.path.join(dump_path, fname), arrs[idx])
            idx += 1


def cosine_similarity(mat0: NDArray, mat1: NDArray) -> float:
    m0 = np.ndarray.flatten(mat0) / norm(mat0)
    m1 = np.ndarray.flatten(mat1) / norm(mat1)
    return np.dot(m0, m1)


def meet_precision(lmat: NDArray, rmat: NDArray, cos_th: float, atol: float, rtol: float) -> bool:
    if (np.any(np.isinf(lmat)) or np.any(np.isinf(rmat))) \
         or (np.isclose(norm(lmat), 0) and np.isclose(norm(rmat), 0)):
        # if overflow happens or norm is close to 0, we fallback to allclose
        return np.allclose(rmat, lmat, atol=atol, rtol=rtol, equal_nan=True)
    # avoid norm overflow, this affects cosine_similarity
    while norm(lmat) > 1e10 or norm(rmat) > 1e10:
        lmat /= 2
        rmat /= 2
    lnorm, rnorm = norm(lmat), norm(rmat)
    # normal cases we check cosine distance and norm closeness
    return 1 - cosine_similarity(lmat, rmat) <= cos_th \
        and bool(np.isclose(rnorm, lnorm, atol=atol, rtol=rtol))
