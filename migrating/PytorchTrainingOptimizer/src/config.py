# coding=utf-8
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
from form_class import TrainingForm, LogForm, LayerForm, ProForm

# msadvisor识别的结果类型
CLASS_TYPE = {'op': '0', 'model': '1'}
ERROR_CODE = {'success': '0', 'optimized': '1'}
EXTEND_TYPE = {'list': '0', 'table': '1', 'sourcedata': '2'}
EXTEND_DATA_TYPE = {'str': '0', 'int': '1', 'double': '2'}

MSADVISOR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(MSADVISOR_PATH, "data", "OptimizerConfig")

# key
CONFIG_PATH_KEY = "config_path"
# parameter keys
CANN_PRO = "profiling"
MODEL_MAP = "modelmap"
TRAIN = "train"
LOG = "log"
NAME = "name"
MODULE = "module"
LAYER = "layer"

# name of config file
TRAIN_CONFIG = "trainConfiguration.json"
LOG_CONFIG = "logConfiguration.json"
LAYER_CONFIG = "layerConfiguration.json"
MODEL_MAP_CONFIG = "ModelMap.json"

CONFIG_DICT = {
    TRAIN: TRAIN_CONFIG,
    LOG: LOG_CONFIG,
    LAYER: LAYER_CONFIG,
    MODEL_MAP: MODEL_MAP_CONFIG,
}

# 文件模块和类的对应
GET_CLASSES = {
    TRAIN: TrainingForm,
    LOG: LogForm,
}
