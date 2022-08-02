#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

import json

class ExtendResult:
    def __init__(self):
        self.type = '0'
        self.extend_title = ""
        self.data_type = []      # table type is an array with multiple elements, list type with only one element
        self.key = []           # this field is only used for table type result
        self.value = []         # table type is a two-dimensional array, list type is a one-dimensional array


class Result:
    def __init__(self):
        self.class_type = '0'
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
extend_type = {'list': '0', 'table': '1', 'sourcedata': '2'}
extend_data_type = {'str': '0', 'int': '1', 'double': '2'}


def evaluate(data_path, parameter):
    """
    interface function called by msadvisor
    Args:
        data_path: string data_path
        parameter: string parameter
    Returns:
        json string of result info 
        result must by ad_result
    """

    # do evaluate work by file data

    # fill result
    result = Result()
    result.class_type = class_type['model']
    result.error_code = error_code['success']
    result.summary = "Op operations need to be optimized"
    extend_result = ExtendResult()
    extend_result.type = extend_type['list']
    # extendResult.type = extend_type['table']

    # list type result
    if extend_result.type == '0':
        extend_result.extend_title = "Recommendations of Ops_Not_Support_Heavy_Format"
        extend_result.data_type.append(extend_data_type['str'])
        extend_result.value.append("Modify the operation to support light format")
        extend_result.value.append("Modify the operation to support heavy format")
        result.extend_result.append(extend_result)
            
    # table type result
    elif extend_result.type == '1':
        extend_result.extend_title = "Op list"
        extend_result.key.append("Op Name")
        extend_result.key.append("Aicore Time(us)")
        extend_result.key.append("Bottleneck Pathway")

        extend_result.data_type.append(extend_data_type['str'])
        extend_result.data_type.append(extend_data_type['double'])
        extend_result.data_type.append(extend_data_type['str'])

        value = []
        value.append("format trans")
        value.append("100.52")
        value.append("mte2->ddr")
        extend_result.value.append(value)

        value1 = []
        value1.append("add")
        value1.append("152.55")
        value1.append("mte3->ub")
        extend_result.value.append(value1)

        result.extend_result.append(extend_result)
    return result.generate()


if __name__ == "__main__":
    data_path = "./"
    ret = evaluate(data_path, "")
    print("sample in:{} out:{}".format(data_path, ret))
