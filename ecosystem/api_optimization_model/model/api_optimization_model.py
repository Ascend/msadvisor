# Copyright 2023 Huawei Technologies Co., Ltd
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
import time
import sys
import json
from src import data_process
from src import config_parser
from util import log
from util import utils

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
    """
    interface function called by msadvisor
    Args:
        dataPath: string data_path
        parameter: input parameter
    Returns:
        json string of result info
        result must be ad_result
    """

    extend_result = init_extent_result()
    if os.path.isdir(dataPath):
        environment_data = utils.get_data('api_optimization_model.json')  # 获取系统配置文件的数据api_optimization_model.json
        environment = environment_data['model_list'][0]['session_list'][0]['parameter']['env']
        mode_data = utils.get_data('api_optimization_model.json')  # 获取系统配置文件的数据api_optimization_model.json
        mode = mode_data['model_list'][0]['session_list'][0]['parameter']['mode']
        environment, mode = config_parser.para_parser(parameter, environment, mode)
        if environment == '310P':
            log.ad_log(log.ad_info, "The knowledge is 310P.")
            extend_result = data_process.data_process_310P(dataPath, extend_result)
            if mode == 'RC':
                extend_result = data_process.data_process_310B_mode(dataPath, extend_result)
                log.ad_log(log.ad_info, "The knowledge is zero memory copy.")
        else:
            log.ad_log(log.ad_info, "The knowledge is 310B.")
            extend_result = data_process.data_process_310B(dataPath, extend_result)
            if mode == 'RC':
                extend_result = data_process.data_process_310B_mode(dataPath, extend_result)
                log.ad_log(log.ad_info, "The knowledge is zero memory copy.")
    else:
        log.ad_log(log.ad_error, "The input dataPath is incorrect. Please check -d path.")
    result = Result()
    return result_parse(result, extend_result, environment)


def result_parse(result, extend_result, environment):
    if not extend_result.value:
        result.class_type = class_type['op']
        result.error_code = error_code['optimized']
        if environment == '310P':
            result.summary = "310P API operations are well optimized"
        else:
            result.summary = "310B API operations are well optimized"
        return result.generate()
    result.class_type = class_type['op']
    result.error_code = error_code['success']
    if environment == '310P':
        result.summary = "310P API operations need to be optimized"
    else:
        result.summary = "310B API operations need to be optimized"
    result.extend_result.append(extend_result)
    return result.generate()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('path argument required!')
    print(evaluate(sys.argv[1], 1))