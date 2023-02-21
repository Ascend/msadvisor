#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

add
"""
from __future__ import absolute_import

import tbe.dsl as tbe
from functools import reduce
from tbe import tvm
from tbe.common.register import register_op_compute
from tbe.common.utils import para_check
from tbe.common.utils import shape_util

# General limitation of the reduce size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@register_op_compute("ExpNeg", op_mode="dynamic", support_fusion=True)
def exp_neg_compute(input_x, output_z, kernel_name="exp_neg"):
    shape_x = shape_util.shape_to_list(input_x.shape)
    shape_x, shape_max = shape_util.broadcast_shapes(shape_x, param_name_input1="input_x")
    shape_size = reduce(lambda x, y: x * y, shape_max[:])
    if shape_size > SHAPE_SIZE_LIMIT:
        raise RuntimeError("the shape is too large to calculate")

    input_x = tbe.broadcast(input_x, shape_max)
    res = tbe.vdiv(input_x, tbe.vexp(input_x))

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def exp_neg_dsl(input_x, output_z, kernel_name="exp_neg"):
    shape_x = input_x.get("shape")

    check_tuple = ("float16", "int32")
    input_data_type = input_x.get("dtype").lower()
    para_check.check_dtype(input_data_type, check_tuple, param_name="input_x")

    shape_x, shape_max = shape_util.broadcast_shapes(shape_x, param_name_input1="input_x")

    data_x = tvm.placeholder(shape_x, name="data_1", dtype=input_data_type)

    res = exp_neg_compute(data_x, output_z, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": (data_x, res)}
    tbe.build(schedule, config)