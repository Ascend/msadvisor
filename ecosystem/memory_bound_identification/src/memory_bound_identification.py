#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

import os
import json
from get_profiling_data import get_profiling_data, Search_PROF_file_name
from get_bottlenck_pathway import get_bottleneck_pathway
from get_repeated_data_migration import get_repeated_data_migration
from get_data_migration_granularity import get_data_migration_granularity
from get_bottleneck_pipeline import get_bottleneck_pipeline
from result_parse import result_parse, Result


class_type = {'op': '0', 'model': '1'}
error_code = {'success': '0', 'optimized': '1'}
extend_type = {'list': '0', 'table': '1', 'sourcedata': '2'}
extend_data_type = {'str': '0', 'int': '1', 'double': '2'}


def evaluate(data_path, parameter='{"device_id": 0, "Topk": 5}'):
    """
    interface function called by msadvisor
    Args:
        data_path: string data_path
    Returns:
        json string of result info
        result must by ad_result
    """
    # do evaluate work by file data
    parameter_dict = json.loads(parameter)
    device_id = parameter_dict.get('device_id')
    Topk = parameter_dict.get('Topk')
    # 判断是否放入了PROF_  文件夹
    num_PROF_dir = Search_PROF_file_name(data_path)
    if not num_PROF_dir:
        result = Result()
        result.class_type = class_type['model']
        result.error_code = error_code['optimized']
        result.summary = "Please put in profiling data!"
        return result.generate()
    list = get_profiling_data(data_path, device_id)
    # 检测profiling数据是否齐全
    set_parameter = ['ArithmeticUtilization', 'PipeUtilization', 'Memory', 'MemoryL0', 'MemoryUB']
    profiling_parameter = ['cube_fops', 'mac_time(us)', 'main_mem_read_bw(GB/s)', 'l0c_write_bw_cube(GB/s)', 'ub_write_bw_vector(GB/s)']
    lack_metrics = []
    for i, profil in enumerate(profiling_parameter):
        if profil not in list[0].keys():
            lack_metrics.append(set_parameter[i])
    if len(lack_metrics) > 0:
        result = Result()
        result.class_type = class_type['model']
        result.error_code = error_code['optimized']
        result.summary = "Lack profiling data collected by metrics:" + str(lack_metrics)
        return result.generate()
    ensure_have_memory_bound = ensure_memory_bound(list)
    if ensure_have_memory_bound == 0:
        result = Result()
        result.class_type = class_type['model']
        result.error_code = error_code['optimized']
        result.summary = "All operators memory operation are well optimized"
        return result.generate()
    # 搬运瓶颈的具体原因识别 -> list
    op_bottleneck_pathway_list = get_bottleneck_pathway(list, Topk)
    op_redundancy_rate_list = get_repeated_data_migration(list, Topk)
    if ensure_have_dump(data_path):
        op_data_migration_granularity = get_data_migration_granularity(data_path, list, Topk)
    else:
        op_data_migration_granularity = []
    op_bottleneck_pipeline_list = get_bottleneck_pipeline(list, Topk)
    result = result_parse(op_bottleneck_pathway_list, op_redundancy_rate_list, op_data_migration_granularity, op_bottleneck_pipeline_list)
    return result


# 通过获得模型中所有算子的memory_bound，来确定模型是否存在搬运瓶颈
def ensure_memory_bound(data):
    list = data
    list_memory_bound = []
    for i in range(len(list)):
        t = float(list[i].get('memory_bound'))
        if t > 1:
            list_memory_bound.append(t)
    return len(list_memory_bound)


# 判断是否生成了仿真文件夹
def ensure_have_dump(data_path):
    dump_file = data_path + '/project/kernel_meta/dump'
    if os.path.exists(dump_file) and len(os.listdir(dump_file)) != 0:
        return True
    else:
        return False


if __name__ == "__main__":
    # 测试时以data所在的绝对路径为data_path
    # 若使用其他的device_id采集Profiling数据，需指定参数device_id。默认：0
    # Topk默认值5，意思是默认显示每个场景耗时前五的算子信息
    CURRENT_FILE_PATH = os.path.abspath(__file__)
    CURRENT_DIR = os.path.dirname(CURRENT_FILE_PATH)
    data_path = CURRENT_DIR + '/../data'
    ret = evaluate(data_path, '{"device_id": 0, "Topk": 5}')
    print("out:{}".format(ret))
