#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""


# 2.识别重复搬运，减少数据重复搬移量。
def get_repeated_data_migration(data, Topk):
    list = data
    redundancy_rate_list = []
    sorted_redundancy_rate_list = []
    threshold_value = 0.5  # 冗余度阈值
    for i in range(len(list)):
        Model_Name = list[i].get('Model Name')
        Op_Name = list[i].get('Op Name')
        OP_Type = list[i].get('OP Type')
        op_name = Model_Name + '/' + Op_Name + '/' + OP_Type
        main_mem_read_bw = float(list[i].get('main_mem_read_bw(GB/s)'))
        l2_read_bw = float(list[i].get('l2_read_bw(GB/s)'))
        total_cycles = float(list[i].get('total_cycles'))
        vec_fp32_ratio = float(list[i].get('vec_fp32_ratio'))
        vec_fp16_ratio = float(list[i].get('vec_fp16_ratio'))
        vec_int32_ratio = float(list[i].get('vec_int32_ratio'))
        mac_fp16_ratio = float(list[i].get('mac_fp16_ratio'))
        mac_int8_ratio = float(list[i].get('mac_int8_ratio'))
        aicore_time = float(list[i].get('aicore_time(us)'))
        memory_bound = float(list[i].get('memory_bound'))
        dict_temp4 = {}
        # 计算量
        vec_caculation = total_cycles * vec_fp32_ratio * 4 * 64 * 2 + total_cycles * vec_fp16_ratio * 2 * 128 * 2 + total_cycles * vec_int32_ratio * 4 * 64 * 2
        cube_caculation = total_cycles * mac_fp16_ratio * 16 * 16 * 2 * 2 + total_cycles * mac_int8_ratio * 16 * 32 * 2 * 1
        if memory_bound > 1:
            aicore_redundancy_rate = 0
            if (vec_fp16_ratio or vec_fp32_ratio or vec_int32_ratio or mac_fp16_ratio or mac_int8_ratio):
                aicore_redundancy_rate = (main_mem_read_bw + l2_read_bw) * aicore_time * 1000 / (vec_caculation + cube_caculation)
            if aicore_redundancy_rate > threshold_value:
                dict_temp3 = {'op_name': op_name, 'aicore redundancy rate': aicore_redundancy_rate, 'aicore_time': aicore_time}
                dict_temp4.update(dict_temp3)
                redundancy_rate_list.append(dict_temp4.copy())
    if len(redundancy_rate_list) > 0:
        sorted_id = sorted(range(len(redundancy_rate_list)), key=lambda x: redundancy_rate_list[x].get('aicore_time'), reverse=True)
        for i in range(len(redundancy_rate_list)):
            sorted_redundancy_rate_list.append(redundancy_rate_list[sorted_id[i]])
        if len(redundancy_rate_list) <= Topk:
            return sorted_redundancy_rate_list
        else:
            return sorted_redundancy_rate_list[:Topk]
    else:
        return sorted_redundancy_rate_list
