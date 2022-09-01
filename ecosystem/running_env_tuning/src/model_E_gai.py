#!/usr/bin/env python3.7.5
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
"""

import os
import json
import function


# msadvisor识别的结果类型
# 扩展结果类，调用generate后返回可转json的数据格式，该类用于填充至结果类的extends列中，不建议修改
class ExtendResult:
    def __init__(self):
        self.type = '0'
        self.data_type = []  # table type is an array with multiple elements, list type with only one element
        self.extend_title = ""
        self.key = []  # this field is only used for table type result
        self.value = []  # table type is a two-dimensional array, list type is a one-dimensional array


# 结果类，调用generate后返回json格式结果，返回结果必须固定为此格式，不可修改。
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
        return json.dumps(res)


CLASS_TYPE = {'op': '0', 'model': '1'}
ERROR_CODE = {'success': '0', 'optimized': '1'}
EXTEND_TYPE = {'list': '0', 'table': '1', 'sourcedata': '2'}
EXTEND_DATA_TYPE = {'str': '0', 'int': '1', 'double': '2'}


# msadvisor调用函数接口，调用时传入工程文件夹路径(传入绝对路径)
# 需要用户自行修改具体profiling数据的位置
def Evaluate(datapath, parameter):
    """
    interface function called by msadvisor
    Args:
        data_path: string data_path
    Returns:
        json string of result info
        result must be ad_result
        :param datapath:
    """
    # do evaluate work by file data
    print(parameter)
    result = Result()
    sequence = 0  # summary次序
    result.class_type = CLASS_TYPE['model']
    result.summary = "Operating environment Tuning need to be optimized, "

    environment_filename = 'environmentConfig.json'
    user_filename = 'UserEnvironmentConfig.json'

    target_path = "knowledgeBase"
    environment_data = get_data(environment_filename, datapath, target_path)  # 获取系统配置文件的数据environmentConfig.json
    user_data = get_data(user_filename, datapath, target_path)  # 获取用户配置文件的数据UserEnvironmentConfig.json
    # 获取各个方向的ExtendResult,并处理各个方向的er
    # 方向1
    er1, optimizedsummary = direction1_process(user_data)
    result, sequence = result_generate(er1, result, "Direction1", optimizedsummary, sequence)
    # 方向2
    er2, optimizedsummary = direction2_process(user_data, datapath, target_path)  # 方向二
    result, sequence = result_generate(er2, result, "Direction2", optimizedsummary, sequence)
    # 方向3
    er3, optimizedsummary = direction3_process(environment_data, datapath, target_path)
    result, sequence = result_generate(er3, result, "Direction3", optimizedsummary, sequence)
    # 方向4_1
    er4_1, optimizedsummary = direction4_1_process(environment_data, user_data, datapath, target_path)
    result, sequence = result_generate(er4_1, result, "Direction4_1", optimizedsummary, sequence)

    # 方向4_2
    er4_2, optimizedsummary = direction4_2_process(environment_data, datapath, target_path)
    result, sequence = result_generate(er4_2, result, "Direction4_2", optimizedsummary, sequence)
    # 方向5
    er5, optimizedsummary = direction5_process(environment_data, datapath, target_path)
    result, sequence = result_generate(er5, result, "Direction5", optimizedsummary, sequence)

    return result.generate()


# result最终生成
# 处理各个方向的er   sequence序号
def result_generate(er, result, direction, OptimizedSummary, sequence):
    # 处理各个方向的er
    if er.data_type != []:
        sequence += 1
        result.summary += str(sequence) + ". " + OptimizedSummary
        result.summary += ' '
        result.extend_result.append(er)
    return result, sequence


# 根据路径获取解析数据json->python
def get_data(filename, dir_path='./', second_path=''):
    file_path = os.path.join(dir_path, second_path)  # ./second_path
    file_path = os.path.join(file_path, filename)
    real_file_path = os.path.realpath(file_path)
    with open(real_file_path, 'r', encoding='UTF-8') as task_json_file:
        task_data = json.load(task_json_file)
    return task_data


# 方向一：推理卡的选择
def direction1_process(user_data):
    er = ExtendResult()
    optimizedsummary = ""
    applicationSceneNum = user_data.get('direction_one')  # 获取应用场景编码
    temp_npu_name = function.getCard()
    if temp_npu_name == "d100":
        npu_name = "Atlas 300I"
    elif temp_npu_name == "d500":
        npu_name = function.getCard310P()
    else:
        npu_name = "others"
    if npu_name == "others" or npu_name == "Atlas 300I":
        er.data_type.append(EXTEND_DATA_TYPE['str'])
        er.extend_title = "Please replace the inference card with Atlas 300I Pro or Atlas 300V Pro"
        optimizedsummary = "The current inference card is not the target of migration or is not included in the expert system"
        return er, optimizedsummary
    else:
        if applicationSceneNum <= 5 and applicationSceneNum >= 1:
            if npu_name == "Atals 300I Pro":
                return er, optimizedsummary
            else:
                er.type = EXTEND_TYPE['list']
                er.extend_title = "Recommendation of Inference Card:"
                er.data_type.append(EXTEND_DATA_TYPE['str'])
                er.key.append("Atals 300I Pro key")
                er.value.append("Atals 300I Pro")
                optimizedsummary = "Inference card need to be optimized"
            return er, optimizedsummary
        elif applicationSceneNum == 6:
            if npu_name == "Atals 300V Pro":
                return er, optimizedsummary
            else:
                er.type = EXTEND_TYPE['list']
                er.extend_title = "Recommendation of Inference Card:"
                er.data_type.append(EXTEND_DATA_TYPE['str'])
                er.key.append("Atals 300V Pro key")
                er.value.append("Atals 300V Pro")
                optimizedsummary = "Inference card need to be optimized"
            return er, optimizedsummary
        else:
            er.data_type.append(EXTEND_DATA_TYPE['str'])
            er.extend_title = "The input sequence number of the application scene should be 1-6"
            optimizedsummary = "The input application scene parameter is incorrect"
            return er, optimizedsummary

# 方向二：推理服务器兼容校验，返回的推理服卡匹配的推理服务器信息
def direction2_process(user_data, datapath, target_path):
    target_path += '/Direction2/E'
    er = ExtendResult()
    optimizedsummary = ""
    servers_recommend_list = list()

    # 获取当前推理卡
    temp_pcie_Card = function.getCard()
    if temp_pcie_Card == "d100":
        pcie_Card = "Atlas 300I"
    elif temp_pcie_Card == "d500":
        pcie_Card = function.getCard310P()
    else:
        pcie_Card = "others"

    if pcie_Card == "others" or pcie_Card == "Atlas 300I":
        # 推理卡不为Atlas 300I Pro或Atlas 300V Pro，但是方向一已经输出此错误了，无需重复
        # er.data_type.append(EXTEND_DATA_TYPE['str'])
        # er.extend_title = "Please replace the inference card with Atlas 300i Pro or Atlas 300V Pro"
        # optimizedsummary = "The current inference card is not appropriate"
        return er, optimizedsummary
    else:   # 推理卡为Atlas 300I Pro或 Atlas 300V Pro
        server_name = user_data.get('direction_two')[0].get('servers_name')

        server_pcieCard_data = get_data('Server_PcieCard.json', datapath, target_path)  # 从对应服务器的json文件中获取数据
        server_piceCard_list = server_pcieCard_data['Server_PcieCard']
        for server_piceCard in server_piceCard_list:
            if server_piceCard['Server Model'] == server_name and server_piceCard['Ascend AI processor'] == pcie_Card:
                er.extend_title = 'Pcie Card Matching AI servers successfully'
                return er, optimizedsummary
            elif server_piceCard['Ascend AI processor'] == pcie_Card and server_piceCard['Cooperative partner'] == 'Huawei':
                servers_recommend_list.append(server_piceCard)
        if servers_recommend_list:
            er.type = EXTEND_TYPE['table']

            er.extend_title = 'The following are the recommended inference servers:'
            er.data_type = [EXTEND_DATA_TYPE['str'] * 8]
            er.key = ['Cooperative partner', 'Server Model', 'Ascend AI processor',
                      'Maximum number of AI processors per node',
                      'CPU family', 'Maximum number of CPUs per node', 'Server form', 'Expiration date']
            for servers_recommend in servers_recommend_list:
                er.value.append([servers_recommend.get(er.key[0]),
                                 servers_recommend.get(er.key[1]),
                                 servers_recommend.get(er.key[2]),
                                 servers_recommend.get(er.key[3]),
                                 servers_recommend.get(er.key[4]),
                                 servers_recommend.get(er.key[5]),
                                 servers_recommend.get(er.key[6]),
                                 servers_recommend.get(er.key[7])])
            optimizedsummary = "The inference card is incompatible with the inference server"
        return er, optimizedsummary


# 方向三 基础软件适配
def direction3_process(environment_data, datapath, target_path):
    er = ExtendResult()
    optimizedsummary = ""
    target_path += '/Direction3'
    inferenceCard_name = environment_data.get('direction_three')[0].get('inferenceCard_name')  # 真实推理卡信息
    inferenceCard_data = get_data(inferenceCard_name + '.json', datapath, target_path)  # 获取各推理卡兼容的操作系统数据
    status, npu_name, os_name = function.IsCompatible(inferenceCard_data)
    if status == 0:
        return er, optimizedsummary
    elif status == 1:
        er.type = EXTEND_TYPE['list']
        er.extend_title = "Recommendations of operating system:"
        er.data_type = [EXTEND_DATA_TYPE['str']]
        er.value = inferenceCard_data[npu_name]
        optimizedsummary = "Operating system need to be optimized"
        return er, optimizedsummary
    else:  # 推理卡不为Atlas 300I Pro或Atlas 300V Pro，但是方向一已经输出此错误了，无需重复
        # er.data_type.append(EXTEND_DATA_TYPE['str'])
        # er.extend_title = "Please replace the inference card with Atlas 300i pro or atlas 300V pro"
        return er, optimizedsummary


# 方向四_1:昇腾软件兼容性校验---媒体数据处理接口迁移指引
def direction4_1_process(environment_data, user_data, datapath, target_path):
    target_path += '/Direction4/E'
    er1 = ExtendResult()
    er2 = ExtendResult()
    transfer_version = user_data.get('direction_four')[0].get('transfer_version')  # 需要转化模型的版本
    transfer_V1_file = environment_data.get('direction_four')[0].get(
        'transfer_V1_file')  # 转化为310pV1的接口信息310_Transfer_v1
    transfer_V2_file = environment_data.get('direction_four')[0].get(
        'transfer_V2_file')  # 转化为310pV2的接口信息310_Transfer_v2
    # 获取目标文件下的所有文件内容
    target_file_address = user_data.get('direction_four')[0].get('target_file_address')  # 转化为310pV2的接口信息310_Transfer_v2
    target_file_address_list = list(str.split(target_file_address, ','))
    target_file_content = function.GetFileContent(target_file_address_list)

    flag = False
    needed_sketchy_function = user_data.get('direction_four')[1]  # 用户所需的模块功能列表

    transfer_V1_json = get_data(transfer_V1_file + '.json', datapath, target_path)  # 获取对应json中数据
    transfer_V2_json = get_data(transfer_V2_file + '.json', datapath, target_path)  # 获取对应json中数据

    if needed_sketchy_function['VPC'] + needed_sketchy_function['VDEC'] + needed_sketchy_function['VENC'] + \
            needed_sketchy_function['JPEGD'] + needed_sketchy_function['JPEGE'] == 0 \
            and needed_sketchy_function['PNGD'] == 1 and transfer_version == '310pV1':
        er1.extend_title += 'PNGD'
        flag = True

    if flag:  # 为True的话说明310p_v1_acldvpp版本满足不了需求 需要迁移到v2上
        er1.extend_title = er1.extend_title + " not supported when transfer to " + transfer_version
        er1.type = EXTEND_TYPE['list']
        er1.data_type = EXTEND_DATA_TYPE['str']
        er1.value.append('Recommended transfer to 310p_v2')
        optimizedsummary = "Transfer to 310p_v1 is unreasonable"
        return er1, optimizedsummary
    elif transfer_version == '310pV1':  # flag为False且转移的版本为V1版本的话说明迁移到310p_v1_acldvpp版本可以实现
        er1.extend_title = 'Relevant interface recommendations for transferring to 310p_v1:'
        er1.type = EXTEND_TYPE['table']
        # er1.data_type = [EXTEND_DATA_TYPE['str'] * 3]
        er1.key = ['The AscendCL interface name', 'The migration recommendations from 310_v1 to 310p_v1',
                   'Module']
        for key, value in needed_sketchy_function.items():
            if value == 1 and key != 'PNGD':  # v1版本中没有这个PNGD功能
                temps = transfer_V1_json['v1_' + key]
                for temp in temps:
                    interface_temp_list = str.split(temp.get(er1.key[0]), ',')
                    for interface in interface_temp_list:
                        # if re.findall(interface,target_file_content) != []:
                        if interface in target_file_content:
                            er1.value.append([temp.get(er1.key[0]),
                                              temp.get(er1.key[1]),
                                              key])
                            break  # 如果当前module下的一个功能中有了用户代码中的接口 则输出一次这个temp即可，避免多次输出
            else:
                continue
        if er1.value:
            er1.data_type = [EXTEND_DATA_TYPE['str'] * 3]
        optimizedsummary = "There are relevant transfer suggestions from 310_v1 to 310p_v1"
        return er1, optimizedsummary
    elif transfer_version == '310pV2':  # flag为False且转移的版本为V2版本的话说明迁移到310p_v2_hi_mpi版本可以实现
        er2.extend_title = 'Relevant interface recommendations for transferring to 310p_v2:'
        er2.type = EXTEND_TYPE['table']
        er2.data_type = [EXTEND_DATA_TYPE['str'] * 3]
        er2.key = ['The migration recommendations from 310_v1 to 310p_v2', 'The AscendCL interface name', 'Module']
        # 对必须处理的头文件进行单独处理
        temps = transfer_V2_json['v2_library file']
        for temp in temps:
            er2.value.append([temp.get(er2.key[0]),
                              temp.get(er2.key[1]),
                              'library file'])
        temps = transfer_V2_json['v2_header file']
        for temp in temps:
            er2.value.append([temp.get(er2.key[0]),
                              temp.get(er2.key[1]),
                              'header file'])
        for key, value in needed_sketchy_function.items():
            if value == 1:
                temps = transfer_V2_json['v2_' + key]
                for temp in temps:
                    if temp.get(er2.key[1])[0] == 'a' or temp.get(er2.key[1])[0] == 'r':  # 为a或者为r的话说明是一个接口列表
                        interface_temp_list = str.split(temp.get(er2.key[1]), ',')
                        for interface in interface_temp_list:
                            # if re.findall(interface,target_file_content) != []:
                            if interface in target_file_content:
                                er2.value.append([temp.get(er2.key[0]),
                                                  temp.get(er2.key[1]),
                                                  key])
                                break
                    else:
                        er2.value.append([temp.get(er2.key[0]),
                                          temp.get(er2.key[1]),
                                          key])
            else:
                continue
        optimizedsummary = "There are relevant transfer suggestions from 310_v1 to 310p_v2"
        return er2, optimizedsummary


# 方向四_2:昇腾软件兼容性校验---目标芯片选项参数差异
def direction4_2_process(environment_data, datapath, target_path):
    target_path += '/Direction4/E'
    er1 = ExtendResult()
    chip_type = environment_data.get('direction_four')[1].get('chip_type')
    chip_option_json = get_data('chip_option.json', datapath, target_path)  # 获取对应json中数据
    chip_option_list = chip_option_json["chip_option"]
    chip = function.getCard()
    if chip == "d100":
        chip_type = '310'
    elif chip == "d500":
        chip_type = '710'
    elif chip == "d801":
        chip_type = '910'
    if chip_type == '310' or chip_type == '710' or chip_type == '910':
        er1.extend_title = "Target chip options for ATC tools:"
        er1.type = EXTEND_TYPE['table']
        er1.data_type = [EXTEND_DATA_TYPE['str'] * 2]
        er1.key = ['options', 'format']
        for chip in chip_option_list:
            if chip[chip_type] == 0:
                continue
            else:
                er1.value.append([chip[er1.key[0]],
                                  chip[chip_type]])
        optimizedsummary = "ATC tool parameters need to be optimized"
    else:
        er1.extend_title = "There are no chip information about " + chip_type + "in ATC tool"
        er1.type = EXTEND_TYPE['table']
        er1.data_type = [EXTEND_DATA_TYPE['str']]
        er1.key = ['recommendation']
        er1.value = ['Please select Ascend 310 or 710 or 910 chip']
        optimizedsummary = "No chip can be matched in ATC tool"
    return er1, optimizedsummary


# 方向五：操作系统内核版本校验
def direction5_process(environment_data, datapath, target_path):
    er = ExtendResult()
    optimizedsummary = ""
    target_path += '/Direction5/E'
    inner_core = environment_data.get('direction_five')[0].get('inner_core')  # 真实推理卡信息
    innerCore_data = get_data(inner_core + '.json', datapath, target_path)  # 获取各推理卡兼容的操作系统数据
    status, version = function.IsInnerCoreAndOSCompatible_E(innerCore_data)
    if status == 0:
        return er, optimizedsummary
    elif status == 1:
        er.type = EXTEND_TYPE['table']
        er.extend_title = "Recommendations of OS kernel version:"
        er.data_type = [EXTEND_DATA_TYPE['str'] * 4]
        er.key = ['Operating System Version', 'Default OS kernel version', 'Operating System Architecture',
                  'Way to install']
        er.value.append([version.get(er.key[0]),
                         version.get(er.key[1]),
                         version.get(er.key[2]),
                         version.get(er.key[3])])
        optimizedsummary = "You need to change the OS kernel version"
        return er, optimizedsummary
    elif status == -1:
        data = innerCore_data.get(version)  # 获取知识库当前推理卡下的所有操作系统、操作系统架构和默认内核版本的对应关系
        er.type = EXTEND_TYPE['table']
        er.extend_title = "Recommendations of OS versions and architecture:"
        er.data_type = [EXTEND_DATA_TYPE['str'] * 2]
        er.key = ['Operating System Version', 'Operating System Architecture']
        for tmp in data:
            er.value.append([tmp.get(er.key[0]),
                             tmp.get(er.key[1])])
        optimizedsummary = "There is no suitable kernel version under the current inference card and operating system version and architecture. Please replace the operating system version and architecture"
        return er, optimizedsummary
    else:  # 推理卡不为Atlas 300I Pro或Atlas 300V Pro，但是方向一已经输出此错误了，无需重复
        return er, optimizedsummary

