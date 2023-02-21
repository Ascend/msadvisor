#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

import re
def mask_jud(datapath):
    with open(datapath, 'r', encoding='utf-8') as fd:
        lines = fd.readlines()
    fd.close()
    mask_up = []
    dic = {}
    line_num = 0
    for line_num, line in enumerate(lines):
        num = -1
        pattern = r'(set_vector_mask)\((.+)\)'
        result1 = re.findall(pattern, line)
        if result1 != []:
            result = ','.join(result1[0])
            res = result.split(',')
            if len(res) == 3:
                a = res[1]
                b = res[2]
            else:
                dic['file name'] = datapath
                dic['line_num'] = str(line_num + 1)
                dic['mask_actual'] = 'you can print the actual mask'
                dic['adv'] = 'Maybe the mask setting is not proper'
                mask_up.append(dic.copy())
                return mask_up
            if a.find('0x') == -1:
                a = a.split('(uint64_t)')[-1]
            if b.find('0x') == -1:
                b = b.split('(uint64_t)')[-1]
            num = mask_calcu(a) + mask_calcu(b)
        if num > 0 and num < 64:
            dic['file name'] = datapath
            dic['line_num'] = str(line_num + 1)
            dic['mask_actual'] = str(num)
            dic['adv'] = 'Check whether the mask setting is proper'
            mask_up.append(dic.copy())
    return mask_up

def mask_calcu(data):
    cal1 = 0
    cal2 = 64
    if data == '-1':
        return cal2
    d1 = int(data, 16)
    if d1 == 0:
        return cal1
    d2 = bin(d1)
    cal1 = d2.count('1')
    return cal1
