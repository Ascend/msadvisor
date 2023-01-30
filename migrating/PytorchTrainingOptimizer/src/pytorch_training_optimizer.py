#!/usr/bin/env python
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
import json
import glob

import config
from log import AD_ERROR, ad_print_and_log
from util import load_json

INVLID_RESULT = ""


def evaluate(datapath, parameter):
    parameter = json.loads(parameter)
    if not os.path.exists(datapath):
        ad_print_and_log(AD_ERROR, f"datapath {datapath} is not exist!")
        return INVLID_RESULT

    for config_name in config.CONFIG_DICT.values():
        path = os.path.join(config.CONFIG_PATH, config_name)
        if not os.path.exists(path):
            # The configuration file must exist because the developer maintains it
            ad_print_and_log(AD_ERROR, "no exit file or dir {}".format(path))
            return INVLID_RESULT
        data = load_json(path)
        if config_name == config.MODEL_MAP_CONFIG:
            for model_dict in data[config.MODULE]:
                assert set(list(model_dict.keys())) == {config.NAME, config.MODULE, config.TRAIN, config.LOG}
        elif config_name in [config.LOG_CONFIG, config.TRAIN_CONFIG]:
            assert set(data.keys()) == {'aligned', 'misaligned', 'mutialigned'}
        elif config_name == config.LAYER_CONFIG:
            assert type(data).__name__ == 'dict'

    get_configuration = {}
    # Full scan datapath to find all .py files
    data_dir_iter = os.walk(datapath)
    python_files = []
    for cur_dir, _, _ in data_dir_iter:
        python_files.extend(glob.glob(os.path.join(cur_dir, "*.py")))
    get_configuration[config.TRAIN] = python_files
    file_count = 0
    for path in get_configuration[config.TRAIN]:
        if os.path.exists(path):
            file_count += 1

    # Record the location of the log file, the profilling directory, config_path
    for file_type in [config.LOG, config.CANN_PRO]:
        if not parameter.get(file_type):
            continue

        file_path = os.path.join(datapath, parameter.get(file_type))
        if os.path.exists(file_path):
            get_configuration[file_type] = file_path
            file_count += 1

    get_configuration[config.CONFIG_PATH_KEY] = config.CONFIG_PATH

    if file_count == 0:
        ad_print_and_log(AD_ERROR, f"There is not python file in {datapath}")
        return INVLID_RESULT
    result = process_data(get_configuration)
    return result


def process_data(get_configuration):
    # layer_file -> tell LayerForm which files don't exist or exist
    layer_file = {"train": False, "module": False, "log": False}
    result = []
    muti_result_list = dict()
    for file_type in get_configuration:
        if file_type in [config.CANN_PRO, config.CONFIG_PATH_KEY]:
            continue
        loadclasses = config.GET_CLASSES[file_type](get_configuration)
        # get all existing files
        pathlist = []
        if file_type == config.TRAIN:
            for path in get_configuration[file_type]:
                layer_file[file_type] = True
                pathlist.append(path)
            curesult, curmutilist = loadclasses.run(pathlist)
        elif file_type == config.LOG:
            path = get_configuration[file_type]
            layer_file[config.LOG] = True
            curesult, curmutilist = loadclasses.run(path)
        else:
            continue
        result += curesult
        muti_result_list[file_type] = curmutilist
    layer_result = config.LayerForm(get_configuration).run(muti_result_list, layer_file)
    result = result + layer_result
    if get_configuration.__contains__("profilling"):
        result = result + config.ProForm(get_configuration).run()
    return result_parse(result)


def result_parse(record):
    result = Result()
    list_record = [i[1] for i in record]
    list_record = list(set(list_record))

    final_record = []
    for i in list_record:
        tmp = {}
        for j in record:
            if j[1] == i:
                file_path = j[2] if j[2] else '-'
                if file_path not in tmp:
                    tmp[file_path] = [str(j[0])]
                else:
                    tmp[file_path].append(str(j[0]))
        file_paths, lines = [], []
        for file_path, line in tmp.items():
            file_paths.append(file_path)
            lines.append(','.join(line))
        file_paths = ';'.join(file_paths)
        line_str = ';'.join(lines)
        final_record.append([i, file_paths, line_str])

    if len(final_record) == 0:
        result.class_type = config.CLASS_TYPE['model']
        result.error_code = config.ERROR_CODE['optimized']
        result.summary = "The model is well optimized"
        return result.generate()
    result.class_type = config.CLASS_TYPE['model']
    result.error_code = config.ERROR_CODE['success']
    result.summary = "The model need to be optimized"

    # 创建ExtendResult
    model_identi_extend = ExtendResult()
    model_identi_extend.type = config.EXTEND_TYPE['table']
    model_identi_extend.extend_title = "Recommendations"

    model_identi_extend.data_type.append(config.EXTEND_DATA_TYPE["str"])
    model_identi_extend.data_type.append(config.EXTEND_DATA_TYPE["str"])
    model_identi_extend.data_type.append(config.EXTEND_DATA_TYPE["str"])
    model_identi_extend.key.append("advice")
    model_identi_extend.key.append("file")
    model_identi_extend.key.append("line")

    for value in final_record:
        model_identi_extend.value.append(value)

    result.extends.append(model_identi_extend.generate())
    return result.generate()


class ExtendResult:
    def __init__(self):
        self.type = '0'
        # table type is an array with multiple elements, list type with only
        # one element
        self.data_type = []
        self.extend_title = ""
        self.identifier = ""
        self.key = []  # this field is only used for table type result
        self.value = []  # table type is a two-dimensional array, list type is a one-dimensional array

    def generate(self):
        res = {
            "type": self.type,
            "dataType": self.data_type,
            "extendTitle": self.extend_title,
            "identifier": self.identifier,
            "key": self.key,
            "value": self.value}
        return res


class Result:
    def __init__(self):
        self.class_type = '0'
        self.error_code = '0'
        self.title = ""
        self.summary = ""
        self.extends = []

    def generate(self):
        res = {
            "classType": self.class_type,
            "errorCode": self.error_code,
            "title": self.title,
            "summary": self.summary,
            "extendResult": self.extends}
        return json.dumps(res, indent="\t")