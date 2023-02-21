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
import json
import log

def get_data(filename):
    file_path = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(file_path, filename)
    real_file_path = os.path.realpath(file_path)
    if not os.path.isfile(real_file_path):
        log.ad_log(log.ad_error, "The json file path error.")
        return
    with open(real_file_path, 'r', encoding='UTF-8') as task_json_file:
        task_data = json.load(task_json_file)
    return task_data
