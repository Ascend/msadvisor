#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

def comput_jud(repeat_jud, mask_jud):
    dic = {}
    list_comput_jud = []
    repeat_num = 0
    mask_num = 0
    if repeat_jud != []:
        for line in repeat_jud:
            if line['adv'] == 'Increase the number of repeats computed by Vector instructions':
                repeat_num = repeat_num + 1
    else:
        repeat_num = 0
    if mask_jud != []:
        for line in mask_jud:
            if line['mask_actual'] != 'you can print the actual mask':
                mask_num = mask_num + 1
    else:
        mask_num = 0
    if (repeat_num == 0) and (mask_num == 0):
        dic['adv'] = 'In this model, optimize the algorithms to reduce the computation amount'
        list_comput_jud.append(dic.copy())
    return list_comput_jud
