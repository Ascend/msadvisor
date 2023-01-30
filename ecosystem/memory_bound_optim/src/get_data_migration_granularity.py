#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

import linecache
import os
import re
from parse_om_model import om_model_parse_to_txt


# 1.2 潜在搬运瓶颈Latency Memory Bound。
# 1.数据搬运粒度是否过小导致。
def get_data_migration_granularity(data_path, profiling_data, Topk):
    threshold_value = 0  # 搬运粒度阈值暂时未设置，用0代替
    list = profiling_data
    line_number = 0
    granularity_list = []
    sorted_granularity_list = []
    om_model_parse_to_txt(data_path)
    dict_name_match = {}
    # 获得名字匹配对   {'Conv16':'te_op_xxx'}
    with open('om_model.txt', 'r', encoding='utf-8') as read_obj:
        for line in read_obj:
            line_number += 1
            for i in range(len(list)):
                Op_Name = list[i].get('Op Name')
                kernelname_in_om = Op_Name + '_kernelname'
                if kernelname_in_om in line:
                    text = str(linecache.getline('om_model.txt', line_number + 2))
                    simulation_filename = re.findall('s: "(.*?)__kernel0', text)
                    dict_name_match[Op_Name] = simulation_filename[0]
    read_obj.close()
    os.remove('om_model.txt')
    # 查找仿真文件名是否存在，存在就继续计算搬运粒度
    for i in range(len(list)):
        memory_bound = float(list[i].get('memory_bound'))
        if memory_bound > 1:
            Op_Name = list[i].get('Op Name')
            simulation_file = data_path + '/project/kernel_meta/dump/' + dict_name_match.get(Op_Name)
            if os.path.exists(simulation_file):
                dump_file_path = simulation_file + '/core0_instr_log.dump'
                with open(dump_file_path, "r", encoding="utf-8") as f_m:
                    content = f_m.read()
                    num_mov_out_to_ub = content.count("mov_out_to_ub")
                    num_mov_ub_to_out = content.count("mov_ub_to_out")
                    num_mov_l1_to_ub = content.count("mov_l1_to_ub")
                f_m.close()
                Model_Name = list[i].get('Model Name')
                OP_Type = list[i].get('OP Type')
                op_name = Model_Name + '/' + Op_Name + '/' + OP_Type
                main_mem_read_bw = float(list[i].get('main_mem_read_bw(GB/s)'))
                main_mem_write_bw = float(list[i].get('main_mem_write_bw(GB/s)'))
                ub_read_bw = float(list[i].get('ub_read_bw(GB/s)'))
                ub_write_bw = float(list[i].get('ub_write_bw(GB/s)'))
                l1_read_bw = float(list[i].get('l1_read_bw(GB/s)'))
                aicore_time = float(list[i].get('aicore_time(us)'))
                Block_Dim = float(list[i].get('Block Dim'))
                memory_bound = float(list[i].get('memory_bound'))
                out_to_ub_bw = min(main_mem_read_bw, ub_write_bw)
                ub_to_out_bw = min(ub_read_bw, main_mem_write_bw)
                l1_to_ub_bw = min(l1_read_bw, ub_write_bw)
                granularity_l12ub = 0
                granularity_out2ub = 0
                granularity_ub2out = 0
                if num_mov_out_to_ub:
                    granularity_out2ub = out_to_ub_bw / Block_Dim * aicore_time * 1e3 / num_mov_out_to_ub
                if num_mov_ub_to_out:
                    granularity_ub2out = ub_to_out_bw / Block_Dim * aicore_time * 1e3 / num_mov_ub_to_out
                if num_mov_l1_to_ub:
                    granularity_l12ub = l1_to_ub_bw / Block_Dim * aicore_time * 1e3 / num_mov_l1_to_ub
                if granularity_out2ub > threshold_value or granularity_ub2out > threshold_value or granularity_l12ub > threshold_value:
                    dict_temp6 = {}
                    dict_temp5 = {'op_name': op_name, 'granularity_out2ub': granularity_out2ub, 'granularity_ub2out': granularity_ub2out, 'granularity_l12ub': granularity_l12ub, 'aicore_time': aicore_time}
                    dict_temp6.update(dict_temp5)
                    granularity_list.append(dict_temp6.copy())
    if len(granularity_list) > 0:
        sorted_id = sorted(range(len(granularity_list)), key=lambda x: granularity_list[x].get('aicore_time'), reverse=True)
        for i in range(len(granularity_list)):
            sorted_granularity_list.append(granularity_list[sorted_id[i]])
        if len(granularity_list) <= Topk:
            return sorted_granularity_list
        else:
            return sorted_granularity_list[:Topk]
    else:
        return sorted_granularity_list
