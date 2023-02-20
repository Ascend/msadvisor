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
import sys
import knowledges
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
from util import log
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
from util import utils

def data_process_310B(file_pathname, extend_result):
    if not os.path.isdir(file_pathname):
        log.ad_log(log.AD_ERROR, "The file_pathname is incorrect. Please check file_pathname path.")
        return extend_result
    log.ad_log(log.ad_info, "Start scanning file.")
    # 遍历该目录下的所有code文件
    for root, dirs, files in os.walk(file_pathname):
        for file in files:
            if file.endswith('.cpp') or file.endswith('.py') or file.endswith('.h'):
                line_num = 0
                path = os.path.join(root, file)
                with open(path, encoding='UTF-8') as file:
                    for line in file.readlines():
                        line_num += 1
                        for api, knowledge in knowledges.knowledges_api_change.items(): # 遍历API变更迁移分析知识库
                            if api in line:
                                value = []
                                value.append(api)
                                value.append(knowledge)
                                value.append(str(file.name) + ' Line:' + str(line_num))
                                extend_result.value.append(value)
                        if  utils.get_data('api_optimization_model.json').get('mode') == 'RC':
                            for api, knowledge in knowledges.knowledges_zero_memory_copy.items():  # 遍历内存操作模式迁移分析知识库
                                if api in line:
                                    value = []
                                    value.append(api)
                                    value.append(knowledge)
                                    value.append(str(file.name) + ' Line:' + str(line_num))
                                    extend_result.value.append(value)
    log.ad_log(log.ad_info, "Finish scanning file.")
    return extend_result