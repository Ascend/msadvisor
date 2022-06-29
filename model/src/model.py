#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
"""

import os
import json

# msadvisor识别的结果类型
CLASS_TYPE = {'op': '0', 'model': '1'}
ERROR_CODE = {'success': '0', 'optimized': '1'}
EXTEND_TYPE = {'list': '0', 'table': '1', 'sourcedata': '2'}
EXTEND_DATA_TYPE = {'str': '0', 'int': '1', 'double': '2'}

# msadvisor调用函数接口，调用时传入工程文件夹路径(传入绝对路径)
# 需要用户自行修改具体profiling数据的位置
def evaluate(datapath, parameter):
    """
    interface function called by msadvisor
    Args:
        datapath: string datapath
        parameter: string parameter
    Returns:
        json string of result info
        result must be ad_result
    """
    # do evaluate work by file data
    filename = 'task_time_0_1_1.json'
    target_path = "profiling/PROF_000001_20220120101335574_GIKHQGHIOIJDBKBC/device_0/timeline"
    data = get_data(filename, datapath, target_path)
    aicpu, aicore = process_data(data)
    cpu_tasks, count_and_record = get_blocking_aicpu_tasks(aicpu, aicore)
    result = result_parse(cpu_tasks, count_and_record)
    return result

# 根据路径获取解析数据json->python
def get_data(filename, dir_path='./', second_path=''):
    file_path = os.path.join(dir_path, second_path)
    file_path = os.path.join(file_path, filename)
    real_file_path = os.path.realpath(file_path)
    with open(real_file_path, 'r') as task_json_file:
        task_data = json.load(task_json_file)
    return task_data

# 将数据根据算子类型分为aicpu以及aicore两组
def process_data(data):
    group_tasks = {'AI_CPU': [], 'AI_CORE': []}
    for task in data:
        if task.get('ph') != 'M':
            if task.get('dur') == 0 or task.get('pid') != 0:
                continue
            task_type = task.get('args').get('Task Type')
            if not task_type:
                continue
            group_tasks[task_type].append(task)

    aicpu = group_tasks['AI_CPU']
    aicore = group_tasks['AI_CORE']

    aicpu = sorted(aicpu, key=lambda op: op.get('ts'))
    aicore = sorted(aicore, key=lambda op: op.get('ts'))
    return aicpu, aicore

# 计算aicpu与aicore的并行时间，识别串行占AICPU算子大于80%时长的算子
def get_blocking_aicpu_tasks(aicpu, aicore):
    cpu_tasks = []
    count_and_record = CountAndRecord()
    for task in aicpu:
        count_and_record.total_dur_time += task.get('dur')
    for task in aicore:
        count_and_record.total_dur_time += task.get('dur')
    index_of_aicpu = 0
    index_of_aicore = 0

    while index_of_aicpu < len(aicpu) and index_of_aicore < len(aicore):
        aicpu_op = aicpu[index_of_aicpu]
        aicpu_op_begin = aicpu_op.get('ts')
        aicpu_op_end = get_task_end_time(aicpu_op)

        aicore_op = aicore[index_of_aicore]
        aicore_op_begin = aicore_op.get('ts')
        aicore_op_end = get_task_end_time(aicore_op)

        coincident_time = 0
        if aicore_op_end < aicpu_op_begin:
            index_of_aicore += 1
            continue

        while aicore_op_begin <= aicpu_op_end:
            if aicore_op_begin <= aicpu_op_begin and aicore_op_end <= aicpu_op_end:
                coincident_time += aicore_op_end - aicpu_op_begin
            elif aicore_op_begin >= aicpu_op_begin and aicore_op_end <= aicpu_op_end:
                coincident_time += aicore_op.get('dur')
            elif aicore_op_begin >= aicpu_op_begin and aicore_op_end >= aicpu_op_end:
                coincident_time += aicpu_op_end - aicore_op_begin
            index_of_aicore += 1
            aicore_op = aicore[index_of_aicore]
            aicore_op_begin = aicore_op.get('ts')
            aicore_op_end = get_task_end_time(aicore_op)

        if coincident_time/aicpu_op.get('dur') <= 0.2:
            cpu_tasks.append(aicpu_op)
            count_and_record(aicpu_op)

        index_of_aicpu += 1

    while index_of_aicpu < len(aicpu):
        aicpu_op = aicpu[index_of_aicpu]
        cpu_tasks.append(aicpu_op)
        count_and_record(aicpu_op)
        index_of_aicpu += 1

    return cpu_tasks, count_and_record

# 计算算子结束时间
def get_task_end_time(task):
    return round(task.get('ts') + task.get('dur'))

# 记录算子阻塞次数，运行时间
class CountAndRecord:
    def __init__(self):
        self.total_dur_time = 0
        self.op_occur_times = dict()
        self.op_dur_time = dict()

    def __call__(self, aicpu_op):
        op_name = aicpu_op.get('name')
        if self.op_occur_times.get(op_name):
            self.op_occur_times[op_name] += 1
        else:
            self.op_occur_times[op_name] = 1
        if self.op_dur_time.get(op_name):
            self.op_dur_time[op_name] += aicpu_op.get('dur')
        else:
            self.op_dur_time[op_name] = aicpu_op.get('dur')

# 扩展结果类，调用generate后返回可转json的数据格式，该类用于填充至结果类的extends列中，不建议修改
class ExtendResult:
    def __init__(self):
        self.type = '0'
        self.data_type = []      # table type is an array with multiple elements, list type with only one element
        self.extend_title = ""
        self.identifier = ""
        self.key = []           # this field is only used for table type result
        self.value = []         # table type is a two-dimensional array, list type is a one-dimensional array

    def generate(self):
        res = {"type": self.type, "data_type": self.data_type, "extend_title": self.extend_title,
               "identifier": self.identifier, "key": self.key, "value": self.value}
        return res

# 结果类，调用generate后返回json格式结果，返回结果必须固定为此格式，不可修改。
class Result:
    def __init__(self):
        self.class_type = '0'
        self.error_code = '0'
        self.title = ""
        self.summary = ""
        self.extends = []

    def generate(self):
        res = {"classType": self.class_type, "errorCode": self.error_code, "title": self.title,
               "summary": self.summary, "extendResult": self.extends}
        return json.dumps(res, indent="\t")

# 结果处理，将识别出的aicpu算子排序，最后打包返回。如无异常则返回好。
def result_parse(cpu_tasks, count_and_record):
    if not cpu_tasks:
        result = Result()
        result.class_type = CLASS_TYPE['model']
        result.error_code = ERROR_CODE['optimized']
        result.summary = "Aicpu operations are well optimized"
        return result.generate()

    result = Result()
    result.class_type = CLASS_TYPE['model']
    result.error_code = ERROR_CODE['success']
    result.summary = "Aicpu operations need to be optimized"

    statis_identi_extend = ExtendResult()
    statis_identi_extend.type = EXTEND_TYPE['table']
    statis_identi_extend.identifier = "aicpu_statistics-identifications"
    statis_identi_extend.extend_title = "Operator statistics, sorted by duration percentage:"
    statis_identi_extend.data_types = [EXTEND_DATA_TYPE['str'], EXTEND_DATA_TYPE['int'], EXTEND_DATA_TYPE['double']]
    statis_identi_extend.key = ['OP Name', 'Count', 'Time(%)']
    sorted_op_dur_time = sorted(count_and_record.op_dur_time.items(), key=lambda d: d[1])
    for op_name, op_dur_time in sorted_op_dur_time:
        statis_identi_extend.value.append([op_name,
                                           count_and_record.op_occur_times.get(op_name),
                                           op_dur_time / count_and_record.total_dur_time * 100])

    op_identi_extend = ExtendResult()
    op_identi_extend.type = EXTEND_TYPE['table']
    op_identi_extend.identifier = "aicpu_operations-identifications"
    op_identi_extend.extend_title = "Identifications of aicpu operations optimization:"
    op_identi_extend.data_type = [EXTEND_DATA_TYPE['str'], EXTEND_DATA_TYPE['double'], EXTEND_DATA_TYPE['double']]
    op_identi_extend.key = ["Operator name", "Task Start Time(us)", "Task Duration(us)", "Task Duration Ratio(%)"]
    sorted_cpu_tasks = sorted(cpu_tasks, key=lambda d: d.get('dur'))
    for task in sorted_cpu_tasks:
        op_identi_extend.value.append([task.get('name'),
                                       task.get('ts'),
                                       task.get('dur'),
                                       task.get('dur') / count_and_record.total_dur_time * 100])

    recomm_extend = ExtendResult()
    recomm_extend.type = EXTEND_TYPE['list']
    recomm_extend.identifier = "aicpu-recommendations"
    recomm_extend.extend_title = "Recommendations of aicpu operations optimization:"
    recomm_extend.data_type = [EXTEND_DATA_TYPE['str']]
    recomm_extend.value = ["recommendation"]

    result.extends.append(statis_identi_extend.generate())
    result.extends.append(op_identi_extend.generate())
    result.extends.append(recomm_extend.generate())
    return result.generate()

# 主函数接口，在本地调试试使用
if __name__ == "__main__":
    datapath = './'
    ret = evaluate(datapath, parameter)
    print(ret)