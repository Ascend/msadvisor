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

import re
import os
import csv
import glob

import config
import layer_sub_class
from log import AD_WARN, ad_print_and_log
from util import load_json
from tools.parser import python


def text2dict(content):
    contentlines = content.split("\n")
    result = {}
    for i, content in enumerate(contentlines):
        result[i+1] = content
    return result


def get_linenum(ori_text, remove_text):
    ori_len = len(ori_text)
    remove_len = len(remove_text)
    ori_point = 1
    remove_point = 1
    while ori_point < ori_len and remove_point < remove_len:
        if ori_text[ori_point].strip() == '':
            ori_text.pop(ori_point)
            ori_point = ori_point + 1
        elif remove_text[remove_point].strip() == '':
            remove_point = remove_point + 1
        elif ori_text[ori_point].startswith(remove_text[remove_point]):
            ori_point = ori_point + 1
            remove_point = remove_point + 1
        else:
            ori_text.pop(ori_point)
            ori_point = ori_point + 1
    return ori_text


class Scanner:
    def __init__(self):
        pass

    def crawls(self, read_pair, dic_determine, file_path):
        line_content_dict, all_content = read_pair
        aligned_dict = dic_determine["aligned"]
        misaligned_dict = dic_determine["misaligned"]
        mutialigned_list = dic_determine["mutialigned"]
        result = []
        muti_result = []
        file_path = os.path.basename(file_path)
        for line, conent in line_content_dict.items():
            for alig, comment in aligned_dict.items():
                if re.search(alig, conent):
                    result.append([line, comment, file_path])
            for mutialig in mutialigned_list:
                if re.search(mutialig, conent):
                    muti_result.append([line, mutialig, file_path])
        for misalig in misaligned_dict:
            if not re.search(misalig, all_content):
                result.append(['-', misaligned_dict[misalig], file_path])
        return result, muti_result


class TrainingForm:
    def __init__(self, get_configuration):
        # train_config_path -> trainConfiguration.json
        train_config_path = os.path.join(
           get_configuration[config.CONFIG_PATH_KEY], config.CONFIG_DICT[config.TRAIN])
        self.train_json = load_json(train_config_path)

    def pretreat(self, filepath):
        with open(filepath, "r", encoding="UTF-8") as f:
            content = f.read()
        # ori_text --> line num : line content
        ori_text = text2dict(content)
        contents = python.remove_comments(content)
        remove_text = text2dict(contents)
        ori_text = get_linenum(ori_text, remove_text)
        return ori_text, contents

    def run(self, filepath_list):
        result, mutialigned_list = [], []
        for filepath in filepath_list:
            read_lines, read = self.pretreat(filepath)
            if not read_lines:
                ad_print_and_log(AD_WARN, "{} has no analyzable content".format(filepath))
                continue
            sub_result, sub_mutialigned_list = Scanner().crawls(
                [read_lines, read], self.train_json, filepath)
            result += sub_result
            mutialigned_list += sub_mutialigned_list
        return result, mutialigned_list


# class ModuleForm:
#     def __init__(self, get_configuration):
#         self.scr = None
#         self.train_json = load_json(os.path.join(
#             get_configuration[config.CONFIG_PATH_KEY], config.CONFIG_DICT[config.MODULE]))
#
#     def pretreat(self, filepath):
#         with open(filepath, "r", encoding="UTF-8") as f:
#             content = f.read()
#         ori_text = text2dict(content)
#         contents = python.remove_comments(content)
#         remove_text = text2dict(contents)
#         ori_text = get_linenum(ori_text, remove_text)
#         return ori_text, contents
#
#     def run(self, filepath_list):
#         result, mutialigned_list = [], []
#         for filepath in filepath_list:
#             read_lines, read = self.pretreat(filepath)
#             if not read_lines:
#                 ad_print_and_log(AD_WARN, "{} has no analyzable content".format(filepath))
#                 continue
#             sub_result, sub_mutialigned_list = Scanner().crawls(
#                 [read_lines, read], self.train_json, filepath)
#             result += sub_result
#             mutialigned_list += sub_mutialigned_list
#         return result, mutialigned_list


class LogForm:
    def __init__(self, get_configuration):
        self.scr = None
        self.log_json = load_json(os.path.join(
            get_configuration[config.CONFIG_PATH_KEY], config.CONFIG_DICT[config.LOG]))

    def pretreat(self, filepath):
        with open(filepath, "r", encoding="UTF-8") as f:
            content = f.read()
        ori_text = text2dict(content)
        return ori_text, content

    def run(self, filepath):
        read_lines, read = self.pretreat(filepath)
        result, mutialigned_list = Scanner().crawls(
            [read_lines, read], self.log_json, filepath)
        return result, mutialigned_list


class ProForm:
    def __init__(self, get_configuration):
        self.op_summary_file_list = glob.glob(os.path.join(
            get_configuration[config.CANN_PRO], "*", "device_*", "summary", "op_summary*.csv"))

        self.op_statistic_file_list = glob.glob(os.path.join(
            get_configuration[config.CANN_PRO], "*", "device_*", "summary", "op_statistic*.csv"))
        self.rule_list = []
        self.rule_list.append(self.rule_1())
        self.rule_list.append(self.rule_2())
        self.result = []

    def pretreat(self):
        pass

    def rule_1(self):
        flag_one = False
        flag_two = False
        if len(self.op_statistic_file_list) != 0:
            with open(self.op_statistic_file_list[0], "r", encoding="UTF-8") as f:
                static_reader = csv.DictReader(f)
                for i, row in enumerate(static_reader):
                    if row["OP Type"] == "TransData":
                        time = float(row["Total Time(us)"])
                        if time > 10 * 1000:
                            flag_one = True
                    if row["OP Type"] == "MatMul" and i < 10:
                        flag_two = True
            if flag_one and flag_two:
                return ['-', "整网MatMul算子引入转置算子,建议用户使用export BMMV2_ENABLE=1进行优化", ""]
            else:
                return None

    def rule_2(self):
        if len(self.op_summary_file_list) != 0:
            with open(self.op_summary_file_list[0], "r", encoding="UTF-8") as f:
                summary_reader = csv.DictReader(f)
                for i, row in enumerate(summary_reader):
                    if (row['OP Type'] == "ReduceSumD") and ("FLOAT16" in row['Input Data Types']) and (
                            int(row["Block Dim"]) < 32) and (float(row["aicore_time(us)"]) > 1000):
                        return ['-', "ReduceSumD算子Fp16性能较差，建议使用黑名单走FP32", ""]

    def run(self):
        for rule in self.rule_list:
            if rule is not None:
                self.result.append(rule)
        return self.result


class LayerForm:
    def __init__(self, parameter):
        self.train_json = load_json(os.path.join(
            config.CONFIG_PATH, config.CONFIG_DICT[config.LAYER]))

    def run(self, multialigned, layer_file):
        result_list = []
        expert_list = layer_sub_class.analysis_cls(
            self.train_json, multialigned, layer_file)
        for exp in expert_list:
            com = exp.run()
            if com:
                result_list.append(com)
        return result_list
