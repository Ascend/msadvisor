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
    file_list = []
    files = os.listdir(path)
    for filename in files:
        if filename.endswith(suffix):
            file_list.append(filename)
    return file_list

def evaluate(data_path, parameter = {'model_file': ''}):
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
    result.error_code = error_code['success']
    result.summary = "The current model is well optimized."

    datapath = os.path.realpath(data_path)
    onnx_path = None
    sub_path = 'onnx'
    model_file = parameter['model_file']#params.get("model_file")
    
    if model_file is None or model_file == '':
        # find onnx model in datapath
        model_file_list = find_model_file(os.path.join(datapath, sub_path)) 
        if not model_file_list:
            raise RuntimeError('model file not exist in datapath')   
        else:
            for model_file in model_file_list:
                onnx_path = os.path.join(os.path.join(datapath, sub_path), model_file)
                onnx_model = add_pad(onnx_path)
                onnx.save(onnx_model,os.path.join(os.path.join(datapath,"results"),model_file))
    else:
        # check onnx model do or not exist
        onnx_path = os.path.join(os.path.join(datapath, sub_path), model_file)
        if not os.path.isfile(onnx_path):
            onnx_path = os.path.realpath(os.path.join(datapath, model_file))
            if not os.path.isfile(onnx_path):
                raise RuntimeError('model file not exist, filename={}'.format(model_file))
            sub_path = ''
        onnx_model = add_pad(onnx_path)
        onnx.save(onnx_model,os.path.join(os.path.join(datapath,"results"),model_file))
    if result.error_code == error_code['optimized']:
        ide_out_path = os.path.join(os.path.join(onnx_path,"results"),model_file)
        result.summary = "The current model has already been optimized, \
            the optimized model path is:%s" % ide_out_path
    return result.generate()

if __name__ == "__main__":
    data_path = "../data" 
    ret = evaluate(data_path)
    print("sample in:{} out:{}".format(data_path, ret))
