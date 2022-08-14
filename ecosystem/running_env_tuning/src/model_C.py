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
def Evaluate(datapath):
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
    result = Result()
    sequence = 0  # summary次序
    result.class_type = CLASS_TYPE['model']
    result.summary = "操作环境需要优化调优, "

    environment_filename = 'environmentConfig.json'
    user_filename = 'UserEnvironmentConfig.json'

    target_path = "knowledgeBase"
    environment_data = get_data(environment_filename, datapath, target_path)  # 获取系统配置文件的数据environmentConfig.json
    user_data = get_data(user_filename, datapath, target_path)  # 获取用户配置文件的数据UserEnvironmentConfig.json
    # 获取各个方向的ExtendResult,并处理各个方向的er
    # 方向1
    er1 = direction1_process(user_data)
    SuccessSummary = "推理卡适合当前应用场景"
    OptimizedSummary = "推理卡需要调整"
    sequence += 1
    result = result_generate(SuccessSummary, er1, result, "Direction1", OptimizedSummary, sequence)
    # 方向2
    server_name, real_pcie, er2 = direction2_process(user_data, datapath, target_path)  # 方向二
    sequence += 1
    result = result_generate(server_name + "和" + real_pcie + "推理卡成功匹配", er2, result,
                             "Direction2", server_name + "和" + real_pcie + "推理卡不匹配", sequence)
    # 方向3
    er3 = direction3_process(environment_data, datapath, target_path)
    SuccessSummary = "推理卡和当前操作系统兼容"
    OptimizedSummary = "操作系统需要调整"
    sequence += 1
    result = result_generate(SuccessSummary, er3, result, "Direction3", OptimizedSummary, sequence)

    # 方向4_1
    transfer_version, er4_1 = direction4_1_process(environment_data, user_data, datapath, target_path)
    if isinstance(er4_1, list):  # 迁移版本2的时候会出现返回两个ExtendResist
        sequence += 1
        result = result_generate("可以迁移到310p_v1版本", er4_1[0], result,
                                 "Direction4-1", "", sequence)
        sequence += 1
        result = result_generate("可以迁移到" + transfer_version + "版本", er4_1[1], result,
                                 "Direction4_1", "", sequence)
    else:
        sequence += 1
        result = result_generate("可以迁移到" + transfer_version + "版本", er4_1, result,
                                 "Direction4_1", "无法迁移到" + transfer_version + "版本", sequence)
    # 方向4_2
    er4_2 = direction4_2_process(environment_data, datapath, target_path)
    sequence += 1
    result = result_generate("目标芯片识别成功", er4_2, result, "Direction4_2",
                             "没有芯片匹配", sequence)
    # 方向5
    er5 = direction5_process(environment_data, datapath, target_path)
    SuccessSummary = "当前的操作系统内核版本是合理的"
    OptimizedSummary = "你需要更换相关软硬件版本"
    sequence += 1
    result = result_generate(SuccessSummary, er5, result, "Direction5", OptimizedSummary, sequence)

    return result.generate()


# result最终生成
# 处理各个方向的er   sequence序号
def result_generate(SuccessSummary, er, result, direction, OptimizedSummary, sequence):

    result.summary += str(sequence) + ". "
    if direction == 'Direction4_2' and er.data_type == []:
        result.summary += OptimizedSummary
        result.summary += ' '
        result.extend_result.append(er)
    elif direction == 'Direction4_2' and er.data_type != []:
        result.summary += SuccessSummary
        result.summary += ' '
        result.extend_result.append(er)
    elif not er.data_type:
        result.summary += SuccessSummary
        result.summary += ' '
    elif er.extend_title == '迁移到310pV1版本的相关接口兼容性信息：' or er.extend_title == '迁移到310pV2版本的相关接口兼容性信息：':
        result.summary += SuccessSummary
        result.summary += ' '
        result.extend_result.append(er)
    else:
        result.summary += OptimizedSummary
        result.summary += ' '
        result.extend_result.append(er)
    return result


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
    applicationSceneNum = user_data.get('direction_one')  # 获取应用场景编码
    temp_npu_name = function.getCard()
    if temp_npu_name == "d100":
        npu_name = "Atlas 300I"
    elif temp_npu_name == "d500":
        npu_name = function.getCard310P()
    else:
        npu_name = ""
    if applicationSceneNum <= 5:
        if npu_name == "Atals 300I Pro":
            return er
        else:
            er.type = EXTEND_TYPE['list']
            er.extend_title = "推理卡推荐："
            er.data_type.append(EXTEND_DATA_TYPE['str'])
            er.key.append("Atals 300I Pro key")
            er.value.append("Atals 300I Pro")
        return er
    else:
        if npu_name == "Atals 300V Pro":
            return er
        else:
            er.type = EXTEND_TYPE['list']
            er.extend_title = "推理卡推荐："
            er.data_type.append(EXTEND_DATA_TYPE['str'])
            er.key.append("Atals 300V Pro key")
            er.value.append("Atals 300V Pro")
        return er


# 方向二：推理服务器兼容校验，返回的推理服卡匹配的推理服务器信息
def direction2_process(user_data, datapath, target_path):
    target_path += '/Direction2'
    er = ExtendResult()
    servers_recommend_list = list()

    pcie_Card = function.getCard310P()
    server_name = user_data.get('direction_two')[0].get('servers_name')

    server_pcieCard_data = get_data('Server_PcieCard.json', datapath, target_path)  # 从对应服务器的json文件中获取数据
    server_pcieCard_list = server_pcieCard_data['Server_PcieCard']
    for server_pcieCard in server_pcieCard_list:
        if server_pcieCard['服务器型号'] == server_name and server_pcieCard['昇腾AI处理器'] == pcie_Card:
            er.extend_title = '推理卡和推理服务器成功匹配'
            return server_name, pcie_Card, er
        elif server_pcieCard['昇腾AI处理器'] == pcie_Card and server_pcieCard['合作伙伴'] == '华为':
            servers_recommend_list.append(server_pcieCard)
    if servers_recommend_list:
        er.type = EXTEND_TYPE['table']
        er.extend_title = '推理卡与推理服务器不兼容，以下是推荐的推理服务器：'
        er.data_type = [EXTEND_DATA_TYPE['str'] * 8]
        er.key = ['合作伙伴', '服务器型号', '昇腾AI处理器', '每节点最大AI处理器数', 'CPU系列', '每节点最大CPU数', '服务器形态', '有效期']
        for servers_recommend in servers_recommend_list:
            er.value.append([servers_recommend.get(er.key[0]),
                             servers_recommend.get(er.key[1]),
                             servers_recommend.get(er.key[2]),
                             servers_recommend.get(er.key[3]),
                             servers_recommend.get(er.key[4]),
                             servers_recommend.get(er.key[5]),
                             servers_recommend.get(er.key[6]),
                             servers_recommend.get(er.key[7])])
    return server_name, pcie_Card, er


# 方向三 基础软件适配
def direction3_process(environment_data, datapath, target_path):
    er = ExtendResult()
    target_path += '/Direction3'
    inferenceCard_name = environment_data.get('direction_three')[0].get('inferenceCard_name')  # 真实推理卡信息
    inferenceCard_data = get_data(inferenceCard_name + '.json', datapath, target_path)  # 获取各推理卡兼容的操作系统数据
    status, npu_name, os_name = function.IsCompatible(inferenceCard_data)
    if status == 0:
        return er
    elif status == 1:
        er.type = EXTEND_TYPE['list']
        er.extend_title = "推荐的操作系统："
        er.data_type = [EXTEND_DATA_TYPE['str']]
        er.value = inferenceCard_data[npu_name]
        return er
    else:
        er.extend_title = "该调优方向不包含此推理卡"
        return er


# 方向四_1:昇腾软件兼容性校验---媒体数据处理接口迁移指引
def direction4_1_process(environment_data, user_data, datapath, target_path):
    target_path += '/Direction4'
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
            and needed_sketchy_function['PNGD'] == 1 and transfer_version == '310p_v1_acldvpp':
        er1.extend_title += 'PNGD'
        flag = True

    if flag:  # 为True的话说明310p_v1_acldvpp版本满足不了需求 需要迁移到v2上
        er1.extend_title = "当转移到" + transfer_version + "时" + er1.extend_title + "不被支持"
        er1.type = EXTEND_TYPE['list']
        er1.data_type = EXTEND_DATA_TYPE['str']
        er1.value.append('建议转移到310p_ v2')
        return transfer_version, er1
    else:  # 为False的话说明迁移到310p_v1_acldvpp或者310p_v2版本满足了需求
        er1.extend_title = '迁移到310pV1版本的相关接口兼容性信息：'
        er1.type = EXTEND_TYPE['table']
        er1.key = ['功能或约束', '涉及的AscendCL接口', 'Ascend 310的实现', 'Ascend 310pV1的实现', '310->310p_V1迁移时，对用户的影响', '所属模块']
        for key, value in needed_sketchy_function.items():
            if value == 1 and key != 'PNGD':  # v1版本中没有这个PNGD功能
                temps = transfer_V1_json['v1_' + key]
                for temp in temps:
                    interface_temp_list = str.split(temp.get(er1.key[1]), ',')
                    for interface in interface_temp_list:
                        # if re.findall(interface,target_file_content) != []:
                        if interface in target_file_content:
                            er1.value.append([temp.get(er1.key[0]),
                                              temp.get(er1.key[1]),
                                              temp.get(er1.key[2]),
                                              temp.get(er1.key[3]),
                                              temp.get(er1.key[4]),
                                              key])
                            break  # 如果当前module下的一个功能中有了用户代码中的接口 则输出一次这个temp即可，避免多次输出
            else:
                continue
        if er1.value:
            er1.data_type = [EXTEND_DATA_TYPE['str'] * 6]
    if transfer_version == '310p_v2_hi_mpi':  # 如果是迁移到V2版本上,那么就要返回V1和V2的所有接口信息
        er2.extend_title = '迁移到310pV2版本的相关接口兼容性信息：'
        er2.type = EXTEND_TYPE['table']
        er2.data_type = [EXTEND_DATA_TYPE['str'] * 5]
        er2.key = ['310->310pV2迁移时，对用户的影响', 'Ascend 310 acldvpp接口', 'Ascend 710 hi_mpi接口', '功能或约束', '所属模块']
        # 对必须处理的头文件进行单独处理
        temps = transfer_V2_json['v2_library file']
        for temp in temps:
            er2.value.append([temp.get(er2.key[0]),
                              temp.get(er2.key[1]),
                              temp.get(er2.key[2]),
                              temp.get(er2.key[3]),
                              '库文件'])
        temps = transfer_V2_json['v2_header file']
        for temp in temps:
            er2.value.append([temp.get(er2.key[0]),
                              temp.get(er2.key[1]),
                              temp.get(er2.key[2]),
                              temp.get(er2.key[3]),
                              '头文件'])
        for key, value in needed_sketchy_function.items():
            if value == 1:
                temps = transfer_V2_json['v2_' + key]
                for temp in temps:
                    if temp.get(er2.key[1])[0] == 'a' or 'r':
                        interface_temp_list = str.split(temp.get(er2.key[1]), ',')
                        for interface in interface_temp_list:
                            # if re.findall(interface,target_file_content) != []:
                            if interface in target_file_content:
                                er2.value.append([temp.get(er2.key[0]),
                                                  temp.get(er2.key[1]),
                                                  temp.get(er2.key[2]),
                                                  temp.get(er2.key[3]),
                                                  key])
                                break
                    else:
                        er2.value.append([temp.get(er2.key[0]),
                                          temp.get(er2.key[1]),
                                          temp.get(er2.key[2]),
                                          temp.get(er2.key[3]),
                                          key])
            else:
                continue
        return transfer_version, [er1, er2]

    return transfer_version, er1


# 方向四_2:昇腾软件兼容性校验---目标芯片选项参数差异
def direction4_2_process(environment_data, datapath, target_path):
    target_path += '/Direction4'
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
        er1.extend_title = "目标芯片选项"
        er1.type = EXTEND_TYPE['table']
        er1.data_type = [EXTEND_DATA_TYPE['str'] * 2]
        er1.key = ['选项', '格式']
        for chip in chip_option_list:
            if chip[chip_type] == 0:
                continue
            else:
                er1.value.append([chip[er1.key[0]],
                                  chip[chip_type]])
    else:
        er1.extend_title = "没有" + chip_type + "芯片的信息"

    return er1


# 方向五：操作系统内核版本校验
def direction5_process(environment_data, datapath, target_path):
    er = ExtendResult()
    target_path += '/Direction5'
    inner_core = environment_data.get('direction_five')[0].get('inner_core')  # 真实推理卡信息
    innerCore_data = get_data(inner_core + '.json', datapath, target_path)  # 获取各推理卡兼容的操作系统数据
    status, InnerCore_version = function.IsInnerCoreAndOSCompatible(innerCore_data)
    if status == 0:
        return er
    elif status == 1:
        er.type = EXTEND_TYPE['table']
        er.extend_title = "操作系统内核版本的推荐："
        er.data_type = [EXTEND_DATA_TYPE['str'] * 4]
        er.key = ['操作系统版本', '操作系统架构', '操作系统内核默认版本', '安装方式']
        er.value.append([InnerCore_version.get(er.key[0]),
                         InnerCore_version.get(er.key[1]),
                         InnerCore_version.get(er.key[2]),
                         InnerCore_version.get(er.key[3])])
        return er
    elif status == -1:
        er.data_type.append(EXTEND_DATA_TYPE['str'])
        er.extend_title = "此推理卡下未查询到相应操作系统的默认内核版本信息"
        return er
    else:
        er.data_type.append(EXTEND_DATA_TYPE['str'])
        er.extend_title = "该调优方向不包含此推理卡"
        return er
