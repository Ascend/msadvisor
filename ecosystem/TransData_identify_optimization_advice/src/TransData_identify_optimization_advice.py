#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

import json
import os
import onnx
import csv


class ExtendResult:
    def __init__(self):
        self.type = '0'
        self.extend_title = ""
        self.data_type = []  # table type is an array with multiple elements, list type with only one element
        self.key = []  # this field is only used for table type result
        self.value = []  # table type is a two-dimensional array, list type is a one-dimensional array


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
        outputstr = json.dumps(res, indent="\t", ensure_ascii=False)
        return outputstr


class_type = {'op': '0', 'model': '1'}
error_code = {'success': '0', 'optimized': '1'}
extend_type = {'list': '0', 'table': '1', 'sourcedata': '2'}
extend_data_type = {'str': '0', 'int': '1', 'double': '2'}


def evaluate(data_path, parameter=None):
    if parameter is None:
        parameter = {}
    if 'onnx_name' not in parameter.keys():
        print("ERROR: Please enter the name of the onnx diagram in the parameter.")
        return
    onnx_node, op_summary, flag = get_data(data_path, parameter)
    if flag:
        advice = get_info(onnx_node, op_summary)
        language = 'English'
        if 'language' in parameter.keys() and parameter['language'] == 'Chinese':
            language = 'Chinese'
        result = result_parse(advice[language], language)
        return result


def get_data(data_path, parameter):
    op_summary = []
    op_summary_id = 0
    onnx_node = []
    flag1, flag2 = False, False
    try:
        if not os.path.exists(data_path + '/profiling'):
            raise Exception(f"the path of profiling does not exist, and the error path is {data_path + '/profiling'},"
                            f" Please check and modify data_path.")
        files_profiling = os.listdir(data_path + '/profiling')
        PROF_list = []
        for file_prof in files_profiling:
            if 'PROF' in file_prof:
                PROF_list.append(file_prof)
        if not len(PROF_list):
            raise Exception("There is no PROF file in the profiling folder, please add PROF file.")
        for PROF_i in range(len(PROF_list)):
            device_list = []
            files_device = os.listdir(data_path + '/profiling/' + PROF_list[PROF_i])
            for file_device in files_device:
                if 'device' in file_device:
                    device_list.append(file_device)
            if not len(device_list):
                continue
            path_summary = os.path.join(data_path, 'profiling/' + PROF_list[PROF_i] + '/' + device_list[
                0] + '/summary')
            if not os.path.exists(path_summary):
                continue
            flies_summary = os.listdir(path_summary)
            op_summary_name = ''
            for file_name in flies_summary:
                if 'op_summary' in file_name and '.csv' in file_name:
                    op_summary_name = file_name
                    break
            if not op_summary_name:
                continue
            path_csv = os.path.join(path_summary, op_summary_name)
            if not os.path.exists(path_csv):
                continue
            flag1 = True
            op_summary_elem = []
            with open(path_csv, 'r') as f:
                reader = csv.reader(f)
                op_csv = list(reader)
                for i in range(1, len(op_csv)):
                    op_summary_elem.append(dict(zip(op_csv[0], op_csv[i])))
            op_summary.append(op_summary_elem)
            if len(op_summary_elem) > 0 and (
                    parameter['onnx_name'].replace('.onnx', '') in op_summary_elem[0]['Model Name']
                    or op_summary_elem[0]['Model Name'] in parameter['onnx_name'].replace('.onnx', '')):
                op_summary_id = -1
                break
        if not flag1:
            raise Exception(
                "There is no op_summary.csv file in the PROF folder, please add op_summary.csv file.")
        else:
            op_summary = op_summary[op_summary_id]
        path_onnx = os.path.join(data_path, 'project/' + parameter['onnx_name'])
        if os.path.exists(path_onnx):
            flag2 = True
            onnx_model = onnx.load(path_onnx)
            graph = onnx_model.graph
            onnx_node = graph.node
        else:
            raise Exception(
                f"the path of onnx does not exist, and the error path is {path_onnx}, Please check and modify the "
                f"file path.")
    except Exception as e:
        print("ERROR:", e)
    finally:
        return onnx_node, op_summary, flag1 and flag2


def get_info(onnx_node, op_summary):
    trans_info = []
    for i in range(len(op_summary)):
        if op_summary[i]['OP Type'] == 'TransData':
            trans_info.append(op_summary[i])
    table_onnx_csv = dict_onnx_csv(onnx_node, op_summary)

    format_4D = ['NHWC', 'NCHW', 'HWCN']
    format_5HD = ['NC1HWC0', 'NHWC1C0']
    format_NZ = ['FRACTAL_NZ', 'FRACTAL_Z']
    format_ND = ['FORMAT_ND']
    Reshape_type = ['Reshape']

    advice_Chinese = []
    advice_English = []

    for i in range(len(trans_info)):
        # 场景1：存在冗余transdata
        if i < len(trans_info) - 3 and op_summary.index(trans_info[i + 1]) + 1 != op_summary.index(trans_info[i + 2]):
            if (trans_info[i]['Input Formats'] in format_4D and trans_info[i]['Output Formats'] in format_5HD and
                trans_info[i + 1]['Input Formats'] in format_5HD and trans_info[i + 1][
                    'Output Formats'] in format_4D and
                trans_info[i + 2]['Input Formats'] in format_4D and trans_info[i + 2][
                    'Output Formats'] in format_5HD and
                trans_info[i + 3]['Input Formats'] in format_5HD and trans_info[i + 3][
                    'Output Formats'] in format_4D) or \
                    (trans_info[i]['Input Formats'] in format_ND and trans_info[i]['Output Formats'] in format_NZ and
                     trans_info[i + 1]['Input Formats'] in format_NZ and trans_info[i + 1][
                         'Output Formats'] in format_ND and
                     trans_info[i + 2]['Input Formats'] in format_ND and trans_info[i + 2][
                         'Output Formats'] in format_NZ and
                     trans_info[i + 3]['Input Formats'] in format_NZ and trans_info[i + 3][
                         'Output Formats'] in format_ND):
                down_name, down_type = relevant_node(trans_info[i + 1], op_summary, table_onnx_csv, 'down')
                advice_Chinese.append(f"冗余的transdata产生由于onnx图的'{down_name}'节点。"
                                      f"在不影响精度的情况下尽量避免非连续的操作，请消除打断5D或NZ连续计算的算子。")
                advice_English.append(
                    f"Redundant transdata is generated due to the '{down_name}' node of the onnx graph."
                    f" Discontinuous operations should be avoided as far as possible without affecting"
                    f" the accuracy. Please eliminate operators that interrupt 5D or NZ continuous calculations.")
        # 场景2：存在Reshape类算子
        if i < len(trans_info) - 1 and op_summary.index(trans_info[i]) + 1 == op_summary.index(trans_info[i + 1]):
            up_name, up_type = relevant_node(trans_info[i], op_summary, table_onnx_csv, 'up')
            down_name, up_type = relevant_node(trans_info[i + 1], op_summary, table_onnx_csv, 'down')
            reshape_node = find_op_Reshape(up_name, down_name, onnx_node, Reshape_type)
            if reshape_node != 'none':
                advice_Chinese.append(f"transdata产生由于onnx图中存在Reshape类算子,请检查onnx图的{reshape_node}节点附近并尽量减少Reshape类操作。")
                advice_English.append(
                    f"Transdata is generated because there are Reshape class operators in the onnx graph. "
                    f"Please check the vicinity of the {reshape_node} node of the onnx graph and minimize the Reshape "
                    f"class operations.")

    advice = {'Chinese': advice_Chinese, 'English': advice_English}
    return advice


def dict_onnx_csv(onnx_node, op_summary):
    result_dict = {}
    for j in range(len(op_summary)):
        result_dict[op_summary[j]['Op Name']] = []
    for i in range(len(onnx_node)):
        for j in range(len(op_summary)):
            if onnx_node[i].name in op_summary[j]['Op Name']:
                result_dict[op_summary[j]['Op Name']].append(onnx_node[i].name)
    return result_dict


def relevant_node(trans_node, op_summary, table_onnx_csv, direction='up'):
    index = op_summary.index(trans_node)
    name = ''
    op_type = ''
    if direction == 'up':
        for i in range(index - 1, 0, -1):
            if table_onnx_csv[op_summary[i]['Op Name']]:
                name = table_onnx_csv[op_summary[i]['Op Name']][-1]
                op_type = op_summary[i]['OP Type']
                break
        if name == '':
            name = 'input'
    if direction == 'down':
        for i in range(index, len(op_summary), 1):
            if table_onnx_csv[op_summary[i]['Op Name']]:
                name = table_onnx_csv[op_summary[i]['Op Name']][0]
                op_type = op_summary[i]['OP Type']
                break
        if name == '':
            name = 'output'
    return name, op_type


def find_op_Reshape(up_name, down_name, onnx_node, Reshape_type):
    if up_name == 'input':
        up_name = onnx_node[0].name
    if down_name == 'output':
        down_name = onnx_node[-1].name
    flag = False
    reshape_name = 'none'
    for i in range(len(onnx_node)):
        if onnx_node[i].name == up_name:
            flag = True
        if flag and onnx_node[i].op_type in Reshape_type:
            reshape_name = onnx_node[i].name
            break
        if onnx_node[i].name == down_name:
            break
    return reshape_name


def result_parse(advice, language):
    if len(advice) == 0:
        result = Result()
        result.class_type = class_type['model']
        result.error_code = error_code['optimized']
        if language == "English":
            result.summary = "There is no transdata operator to be optimized"
        else:
            result.summary = "不存在需要优化的transdata算子"
        return result.generate()

    result = Result()
    result.class_type = class_type['model']
    result.error_code = error_code['success']
    if language == "English":
        result.summary = "Transdata operator exists and needs to be optimized."
    else:
        result.summary = "存在transdata算子需要优化."

    extend_result = ExtendResult()
    extend_result.type = extend_type['list']
    if language == "English":
        extend_result.extend_title = "Suggestions on optimization of transdata operator"
    else:
        extend_result.extend_title = "transdata算子优化建议"
    extend_result.data_type.append(extend_data_type['str'])

    extend_result.value = advice

    result.extend_result.append(extend_result)
    return result.generate()
