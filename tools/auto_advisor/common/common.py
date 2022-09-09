import os
import sys
import onnx
import operator as op

def get_model_shape(model_path: str) -> str:
    """
    获取模型的输入shape，支持onnx模型
    """
    res = ''
    if not model_path.endswith('.onnx'):
        print('Get model shape failed, only support onnx.')
        return res
    model = onnx.load(model_path)
    inputs = model.graph.input
    if len(inputs) == 0:
        return res
    dims = inputs[0].type.tensor_type.shape.dim
    for dim in dims:
        if len(dim.dim_param) != 0:
            res += dim.dim_param + ','
            continue
        res += str(dim.dim_value) + ','
    return res[0:-1]


def is_dynamic(model_path: str) -> bool:
    """
    判断模型是否是动态shape，支持onnx模型
    """
    if not model_path.endswith('.onnx'):
        print('Check model dynamic failed, only support onnx.')
        return False
    model = onnx.load(model_path)
    inputs = model.graph.input
    if len(inputs) == 0:
        return False
    for inputv in inputs:
        dims = inputv.type.tensor_type.shape.dim
        for dim in dims:
            if len(dim.dim_param) != 0:
                return True
            if dim.dim_value < 0:
                return True
    return False

def get_advisor_conf(conf_path: str, title: str) -> str:
    """
    获取配置
    """
    res = ''
    if not os.path.isfile(conf_path):
        return res
    hit_flag = False
    start = '[%s]' % title
    end = '[%s-end]' % title
    f = open(conf_path, 'r')
    for line in f.readlines():
        line = line[:-1]
        if op.eq(line, start):
            hit_flag = True
            continue
        if op.eq(line, end):
            hit_flag = False
            break
        if hit_flag and len(line) != 0:
            res += line + ';'
    if len(res) != 0:
        res = res[:-1]
    return res

