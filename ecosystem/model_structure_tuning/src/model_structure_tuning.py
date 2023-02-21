#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""
import os
import json
import onnx
from function import add_pad

LOG_INFO = 1
LOG_WARN = 2
LOG_ERR = 3
class ExtendResult:
    def __init__(self):
        self.type = '0'
        self.extend_title = ""
        self.data_type = []     # table type is an array with multiple elements, list type with only one element
        self.key = []           # this field is only used for table type result
        self.value = []         # table type is a two-dimensional array, list type is a one-dimensional array

class Result:
    def __init__(self):
        self.class_type = '1'
        self.error_code = '0'
        self.summary = ""
        self.extend_result = []

    def generate(self):
        extend_data = []
        for item in self.extend_result:
            data = {"type": item.type, "extendTitle": item.extend_title,
                    "dataType": item.data_type, "key": item.key, "value": item.value}
            extend_data.append(data)
        res = {"classType": self.class_type, "errorCode": self.error_code,
               "summary": self.summary, "extendResult": extend_data}
        outputstr = json.dumps(res)
        return outputstr

class_type = {'op': '0', 'model': '1'}
error_code = {'success': '0', 'optimized': '1'}

def find_model_file(path, suffix = '.onnx'):
    if not os.path.isdir(path):
        return []
    return list(filter(lambda f: f.endswith(suffix), os.listdir(path)))

def evaluate(data_path, parameter = '{}'):
    """
    interface function called by msadvisor
    Args:
        data_path: string data_path
        parameter: string parameter
    Returns:
        json string of result info 
        result must by ad_result
    """
    result = Result()
    result.class_type = class_type['model']
    result.error_code = error_code['optimized']
    result.summary = "All models are well optimized."

    data_path = os.path.realpath(data_path)
    parameter = json.loads(parameter)
    if isinstance(parameter, dict) and 'model_file' in parameter:
        model_files = [parameter['model_file']]
    else:
        model_files = find_model_file(data_path)

    extend_result = ExtendResult()

    for model_file in model_files:
        model_path = os.path.join(data_path, model_file)
        onnx_model = add_pad(onnx.load(model_path))
        if onnx_model:
            model_name, ext = os.path.splitext(model_file)
            output_path = os.path.join(data_path, f'{model_name}_optimized{ext}')
            onnx.save(onnx_model, output_path)
            extend_result.data_type.append('0')
            extend_result.value.append(f'{model_path} need to be optimized')
    
    if extend_result.value:
        extend_result.extend_title = '{} model(s) can be optimized by LayerNorm fusion' \
                                     .format(len(extend_result.value))
        result.error_code = error_code['success']
        result.summary = extend_result.extend_title
        result.extend_result.append(extend_result)
    return result.generate()

if __name__ == "__main__":
    data_path = "../data/onnx" 
    ret = evaluate(data_path)
    print("sample in:{} out:{}".format(data_path, ret))
