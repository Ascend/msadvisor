

import os
import time
import sys
import json
import data_process
from util import log
# define datatype

class_type = {'op': '0', 'model': '1'}
error_code = {'success': '0', 'optimized': '1'}
extend_type = {'list': '0', 'table': '1', 'sourcedata': '2'}
extend_data_type = {'str': '0', 'int': '1', 'double': '2'}
invalid_result = ""


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
        outputstr = json.dumps(res, indent='\t')
        return outputstr


# define extend_result
def init_extent_result():
    extend_result = ExtendResult()
    extend_result.type = extend_type['table']
    extend_result.data_type.append(extend_data_type['str'])
    extend_result.data_type.append(extend_data_type['str'])
    extend_result.data_type.append(extend_data_type['str'])
    extend_result.key.append("API Name")
    extend_result.key.append("Optimization Suggestion")
    extend_result.key.append("API Location")
    return extend_result


def evaluate(dataPath, parameter):
    extend_result = init_extent_result()
    if os.path.isdir(dataPath):
        extend_result = data_process.data_process(dataPath, extend_result)
    else:
        log.ad_log(log.AD_ERROR, "The input dataPath is incorrect. Please check -d path.")

    result = Result()
    return result_parse(result, extend_result)


def result_parse(result, extend_result):
    if not extend_result.value:
        result.class_type = class_type['op']
        result.error_code = error_code['optimized']
        result.summary = "310B API operations are well optimized"
        return result.generate()
    result.class_type = class_type['op']
    result.error_code = error_code['success']
    result.summary = "310B API operations need to be optimized"
    result.extend_result.append(extend_result)
    return result.generate()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('path argument required!')
    print(evaluate(sys.argv[1], 1))