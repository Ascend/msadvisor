#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

import re
def high_jud(datapath):
    instruct = ['vreduce_sum', 'vreduce_max', 'vreduce_min', 'vreduce_argmax', 'vreduce_argmin',
                 'vduplicate', 'vdata_move', 'vrelu', 'vabs', 'vnot', 'vexp', 'vln', 'vrec', 'vrsqrt',
                 'vadd', 'vsub', 'mulv', 'vdiv', 'vmax', 'vmin', 'vand', 'vor', 'vcmpv', 'vsel',
                 'vcast', 'vconv', 'vquant']
    high_instruct = ['h_reduce_sum', 'h_reduce_max', 'h_reduce_min', 'h_reduce_argmax',
                     'h_reduce_argmin', 'h_duplicate', 'h_data_move', 'h_relu', 'h_abs',
                     'h_not', 'h_exp', 'h_ln', 'h_rec', 'h_rsqrt', 'h_add', 'h_sub', 'mul',
                     'div', 'h_max', 'h_min', 'h_and', 'h_or', 'h_cmpv', 'h_sel', 'h_cast',
                     'h_conv', 'h_quant']
    file_name = datapath
    with open(datapath, 'r', encoding='utf-8') as fd:
        lines = fd.readlines()
    fd.close()
    high_jud = []
    dic = {}
    for line_num, line in enumerate(lines):
        pattern = r'([a-z]\w+)\((.+)\)'
        result1 = re.findall(pattern, line)
        if result1 != []:
            result = ','.join(result1[0])
            res = result.split(',')
            if res != None:
                if res[0] not in high_instruct:
                    if res[0] in instruct:
                        a = res[0]
                        b = 'h_' + a[1:-1] +a[-1] 
                        dic['file name'] = file_name
                        dic['line_num'] = str(line_num + 1)
                        dic['adv'] = f"please use high-performance instruction '{b}' to replace low-performance instruction '{res[0]}'"
                        high_jud.append(dic.copy())
    return high_jud