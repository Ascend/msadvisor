#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

import json
import os
from get_data import get_csv_path, get_cce_path, get_csv_data
from compute_bound_jud import comp_bound
from cube_jud import cube_jud
from precision_jud import preci_jud
from block_jud import block_jud
from bank_conflic_jud import bank_jud
from repeat_jud import repeat_jud
from mask_jud import mask_jud
from compute_amount_jud import comput_jud
from high_performance_jud import high_jud
from exception import ProfilingDataLack, CCEFileNotFound


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
        outputstr = json.dumps(res, indent="\t")
        return outputstr


class_type = {'op': '0', 'model': '1'}
error_code = {'success': '0', 'optimized': '1'}
extend_type = {'list': '0', 'table': '1', 'sourcedata': '2'}
extend_data_type = {'str': '0', 'int': '1', 'double': '2'}


def evaluate(data_path, parameter='{}'):
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
    set_parameter = ['ArithmeticUtilization', 'PipeUtilization', 'MemoryL0', 'MemoryUB', 'ResourceConflictRatio']
    profiling_parameter = ['cube_fops', 'mac_time(us)', 'l0c_write_bw_cube(GB/s)', 'ub_write_bw_vector(GB/s)', 'vec_bankgroup_cflt_ratio']
    parameter = json.loads(parameter)
    topk = parameter.get('topk', 5) if isinstance(parameter, dict) else 5
    divice_id = parameter.get('divice_id', 0) if isinstance(parameter, dict) else 0    
    csv_path = get_csv_path(data_path, divice_id)
    cce_path = get_cce_path(data_path)
    list_total = get_csv_data(csv_path)
    data_path_last = data_path.split(os.path.sep)[-1]

    # 判断csv文件是否缺失，并给出提示
    lack_keys = []
    for i, profil in enumerate(profiling_parameter):
        if profil not in list_total[0].keys():
            lack_keys.append(profil)
    if len(lack_keys) > 0:
        raise ProfilingDataLack(lack_keys)

    # 判断CCE文件是否缺失，并给出提示
    if len(cce_path) == 0:
        raise CCEFileNotFound(f'{data_path_last}/kernel_meta/')

    list_comp_bound = comp_bound(list_total)
    if len(list_comp_bound) == 0:
        result = Result()
        result.class_type = class_type['model']
        result.error_code = error_code['optimized']
        result.summary = "All operators compute operation are well optimized"
        return result.generate()
    list_cube_jud = cube_jud(list_total)
    list_preci_jud = preci_jud(list_total)
    list_block_jud = block_jud(list_total)
    list_bank_jud = bank_jud(list_total)
    list_repeat_jud = []
    list_mask_jud = []
    list_comput_jud = []
    list_high_jud = []

    for path in cce_path:
        list_repeat_jud = list_repeat_jud + repeat_jud(path)
        list_mask_jud = list_mask_jud + mask_jud(path)
        list_high_jud = list_high_jud + high_jud(path)
    list_comput_jud = list_comput_jud + comput_jud(list_repeat_jud, list_mask_jud)

    # topk设置
    if list_cube_jud != []:
        list_cube_jud.sort(key=lambda x: (x['aicore_time']), reverse=True)
        if len(list_cube_jud) > topk:
            list_cube_jud = list_cube_jud[0: topk]
    if list_preci_jud != []:
        list_preci_jud.sort(key=lambda x: (x['aicore_time']), reverse=True)
        if len(list_preci_jud) > topk:
            list_preci_jud = list_preci_jud[0: topk]
    if list_block_jud != []:
        list_block_jud.sort(key=lambda x: (x['aicore_time']), reverse=True)
        if len(list_block_jud) > topk:
            list_block_jud = list_block_jud[0: topk]
    if list_bank_jud != []:
        list_bank_jud.sort(key=lambda x: (x['aicore_time']), reverse=True)
        if len(list_bank_jud) > topk:
            list_bank_jud = list_bank_jud[0: topk]
    if list_repeat_jud != []:
        list_repeat_jud = list_repeat_jud[0: topk]
    if list_mask_jud != []:
        list_mask_jud = list_mask_jud[0: topk]
    if list_high_jud != []:
        list_high_jud = list_high_jud[0: topk]

    # fill result
    result = Result()
    result.class_type = class_type['model']
    result.error_code = error_code['success']
    result.summary = "Operator compute operation need to be optimized"

    extend_result = ExtendResult()
    extend_result.type = extend_type['table']
    extend_result1 = ExtendResult()
    extend_result1.type = extend_type['table']
    extend_result2 = ExtendResult()
    extend_result2.type = extend_type['table']
    extend_result3 = ExtendResult()
    extend_result3.type = extend_type['table']
    extend_result4 = ExtendResult()
    extend_result4.type = extend_type['table']
    extend_result5 = ExtendResult()
    extend_result5.type = extend_type['table']
    extend_result6 = ExtendResult()
    extend_result6.type = extend_type['table']
    extend_result7 = ExtendResult()
    extend_result7.type = extend_type['table']

    # list type result
    if extend_result.type == '0':
        extend_result.extend_title = "Recommendations of Ops_Not_Support_Heavy_Format"
        extend_result.data_type.append(extend_data_type['str'])
        extend_result.value.append("Modify the operation to support light format")
        extend_result.value.append("Modify the operation to support heavy format")
        result.extend_result.append(extend_result)

    # table type result
    elif extend_result.type == '1':
        # 场景判定
        # cube计算场景
        if len(list_cube_jud) != 0:
            extend_result.extend_title = "It is compute bound. Suggest changing calculation units, for example, replace Vector with Cube"
            extend_result.key.append("Op Name")
            extend_result.key.append("compute bound type")
            extend_result.key.append("advice")

            extend_result.data_type.append(extend_data_type['str'])
            extend_result.data_type.append(extend_data_type['str'])
            extend_result.data_type.append(extend_data_type['str'])
            for i in range(len(list_cube_jud)):
                value = []
                value.append(list_cube_jud[i].get('Op name'))
                value.append("compute bound")
                value.append(list_cube_jud[i].get('adv'))
                extend_result.value.append(value)
            result.extend_result.append(extend_result)

        # 精度判定场景
        if len(list_preci_jud) != 0:
            extend_result1.extend_title = "It is compute bound. Suggest adopt low-precision computing"
            extend_result1.key.append("Op Name")
            extend_result1.key.append("compute bound type")
            extend_result1.key.append("advice")

            extend_result1.data_type.append(extend_data_type['str'])
            extend_result1.data_type.append(extend_data_type['str'])
            extend_result1.data_type.append(extend_data_type['str'])
            for i in range(len(list_preci_jud)):
                value = []
                value.append(list_preci_jud[i].get('Op name'))
                value.append("compute bound")
                value.append(list_preci_jud[i].get('adv'))
                extend_result1.value.append(value)
            result.extend_result.append(extend_result1)                    

        # 双核判定场景
        if len(list_block_jud) != 0:
            extend_result2.extend_title = "It is compute bound. Suggest use dual-core"
            extend_result2.key.append("Op Name")
            extend_result2.key.append("compute bound type")
            extend_result2.key.append("advice")

            extend_result2.data_type.append(extend_data_type['str'])
            extend_result2.data_type.append(extend_data_type['str'])
            extend_result2.data_type.append(extend_data_type['str'])
            for i in range(len(list_block_jud)):
                value = []
                value.append(list_block_jud[i].get('Op name'))
                value.append("compute bound")
                value.append(list_block_jud[i].get('adv'))
                extend_result2.value.append(value)
            result.extend_result.append(extend_result2)

        # 计算量过大判定场景
        if len(list_comput_jud) != 0:
            extend_result3.extend_title = "It is  compute bound. Suggest optimize the algorithms to reduce the computation amount"
            extend_result3.key.append("advice")
            extend_result3.key.append("compute bound type")

            extend_result3.data_type.append(extend_data_type['str'])
            extend_result3.data_type.append(extend_data_type['str'])
            value = []
            value.append(list_comput_jud[0].get('adv'))
            value.append("compute bound")
            extend_result3.value.append(value)
            result.extend_result.append(extend_result3)

        # repeat过小场景判定
        if len(list_repeat_jud) != 0:
            extend_result4.extend_title = "It is latency compute bound. Suggest increase the number of repeats computed by Vector instructions"
            extend_result4.key.append("file name")
            extend_result4.key.append("compute bound type")
            extend_result4.key.append("instruct name")
            extend_result4.key.append("line_num")
            extend_result4.key.append("repeat_time")
            extend_result4.key.append("advice")

            extend_result4.data_type.append(extend_data_type['str'])
            extend_result4.data_type.append(extend_data_type['str'])
            extend_result4.data_type.append(extend_data_type['str'])
            extend_result4.data_type.append(extend_data_type['str'])
            extend_result4.data_type.append(extend_data_type['str'])
            extend_result4.data_type.append(extend_data_type['str'])
            for i in range(len(list_repeat_jud)):
                value = []
                value.append(list_repeat_jud[i].get('file name'))
                value.append("Latency compute bound")
                value.append(list_repeat_jud[i].get('instruct name'))
                value.append(list_repeat_jud[i].get('line_num'))
                value.append(list_repeat_jud[i].get('repeat_time'))
                value.append(list_repeat_jud[i].get('adv'))
                extend_result4.value.append(value)
            result.extend_result.append(extend_result4)
        # mask设置不合理场景判定
        if len(list_mask_jud) != 0:
            extend_result5.extend_title = "It is latency compute bound. Suggest check whether the mask setting is proper"
            extend_result5.key.append("file name")
            extend_result5.key.append("compute bound type")
            extend_result5.key.append("line_num")
            extend_result5.key.append("mask actual value")
            extend_result5.key.append("advice")

            extend_result5.data_type.append(extend_data_type['str'])
            extend_result5.data_type.append(extend_data_type['str'])
            extend_result5.data_type.append(extend_data_type['str'])
            extend_result5.data_type.append(extend_data_type['str'])
            extend_result5.data_type.append(extend_data_type['str'])
            for i in range(len(list_mask_jud)):
                value = []
                value.append(list_mask_jud[i].get('file name'))
                value.append("Latency compute bound")
                value.append(list_mask_jud[i].get('line_num'))
                value.append(list_mask_jud[i].get('mask_actual'))
                value.append(list_mask_jud[i].get('adv'))
                extend_result5.value.append(value)
            result.extend_result.append(extend_result5)
        # bank冲突判定场景
        if len(list_bank_jud) != 0:
            extend_result6.extend_title = "It is latency compute bound. Suggest check bank conflict"
            extend_result6.key.append("Op Name")
            extend_result6.key.append("compute bound type")
            extend_result6.key.append("advice")

            extend_result6.data_type.append(extend_data_type['str'])
            extend_result6.data_type.append(extend_data_type['str'])
            extend_result6.data_type.append(extend_data_type['str'])
            for i in range(len(list_bank_jud)):
                value = []
                value.append(list_bank_jud[i].get('Op name'))
                value.append("Latency compute bound")
                value.append(list_bank_jud[i].get('adv'))
                extend_result6.value.append(value)
            result.extend_result.append(extend_result6)
        # 高性能指令场景判定
        if len(list_high_jud)!=0:
            extend_result7.extend_title = "It is latency compute bound. Suggest use high-performance instructions to replace low-performance instructions"
            extend_result7.key.append("file name")
            extend_result7.key.append("compute bound type")
            extend_result7.key.append("line_num")
            extend_result7.key.append("advice")

            extend_result7.data_type.append(extend_data_type['str'])
            extend_result7.data_type.append(extend_data_type['str'])
            extend_result7.data_type.append(extend_data_type['str'])
            extend_result7.data_type.append(extend_data_type['str'])
            for i in range(len(list_high_jud)):
                value = []
                value.append(list_high_jud[i].get('file name'))
                value.append("Latency compute bound")
                value.append(list_high_jud[i].get('line_num'))
                value.append(list_high_jud[i].get('adv'))
                extend_result7.value.append(value)
            result.extend_result.append(extend_result7)
    return result.generate()


if __name__ == "__main__":
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(data_path, 'data')
    ret = evaluate(data_path)
    print("sample in:{} out:{}".format(data_path, ret))
