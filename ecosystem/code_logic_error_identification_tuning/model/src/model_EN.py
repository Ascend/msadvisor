import os
import sys
import json
import warnings
import function
import re
import subprocess
import math
import warnings
import time
import pandas as pd


class ExtendResult:
    def __init__(self):
        self.type = '0'
        self.extend_title = ""
        # table type is an array with multiple elements, list type with only
        # one element
        self.data_type = []
        self.key = []           # this field is only used for table type result
        # table type is a two-dimensional array, list type is a one-dimensional
        # array
        self.value = []


class Result:
    def __init__(self):
        self.class_type = '0'
        self.error_code = '0'
        self.summary = ""
        self.extend_result = []

    def generate(self):
        extend_data = []
        for item in self.extend_result:
            data = {
                "type": item.type,
                "extendTitle": item.extend_title,
                "dataType": item.data_type,
                "key": item.key,
                "value": item.value}
            extend_data.append(data)
        res = {"classType": self.class_type, "errorCode": self.error_code,
               "summary": self.summary, "extendResult": extend_data}
        outputstr = json.dumps(res)
        return outputstr


# msadvisor识别的结果类型
CLASS_TYPE = {'op': '0', 'model': '1'}
ERROR_CODE = {'success': '0', 'optimized': '1'}
EXTEND_TYPE = {'list': '0', 'table': '1', 'sourcedata': '2'}
EXTEND_DATA_TYPE = {'str': '0', 'int': '1', 'double': '2'}

# msadvisor调用函数接口，调用时传入工程文件夹路径(传入绝对路径)
# 需要用户自行修改具体profiling数据的位置


def Evaluate(datapath, API):
    """
    interface function called by msadvisor
    Args:
        data_path: string data_path
    Returns:
        json string of result info
        result must be ad_result
    """
    tyTitle = (
        "In the view parsing business, ShengTeng 310p application migration knowledgebase will give tuning suggestions"
        "from the following aspects based on the feedback data obtained from the user application"
        "1. The memory ECC enable state and the ratio of AI CPU and Ctrl CPU can be optimized"
        "2. The selection and use of the DVPP VPC interface can be optimized."
        "For example, the corresponding batch interface functions can be applied in the target-intensive scenarios "
        "for services such as crop,resize,and paste"
        "3. The selection and use of the DVPP VDEC interface can be optimized. For example, the application of "
        "the VDEC frame extraction function can greatly improve the application performance of most scenarios"
        "4. The development of AI CPU custom operators can be optimized, and optimization suggestions are provided "
        "for operators that may block service flow performance"
        "5. DVPP VPC output YUV 400 format can be optimized, ShengTeng 310p has carried out a good encapsulation of "
        "this aspect of the business, users can make changes according to the optimization suggestions")
    os.chdir(sys.path[0])
    result = Result()
    sequence = 0
    result.class_type = CLASS_TYPE['model']
    result.summary = "After the optimization analysis of this knowledge base, we can get:"
    # 获取各个方向的ExtendResult,并处理各个方向的er
    # 方向1
    sequence += 1
    er1 = direction2_1_process(datapath)
    SuccessSummary_1 = str(sequence) + "." + "In the user migration applications, the memory ECC enabled status " \
        "and the ratio of AI CPU and Ctrl CPU are in good condition."\
        "There is no tuning suggestion for this aspect."
    OptimizedSummary_1 = str(sequence) + "." + "In user migration applications, the memory ECC enable status " \
        "and the ratio of AI CPU to Ctrl CPU""need to be adjusted and optimized."
    result = result_generate(SuccessSummary_1, er1, result, OptimizedSummary_1)
    # 方向2
    sequence += 1
    er2 = direction3_1_process(datapath, API)
    SuccessSummary_2 = str(sequence) + "." + "In the user migration applications, " \
        "the selection and usage of the DVPP VPC interface is relatively good, " \
        "and there is no optimization suggestion in this aspect."
    OptimizedSummary_2 = str(sequence) + "." + "In user migration applications, " \
        "the selection and use of DVPP VPC interfaces need to be adjusted and optimized."
    result = result_generate(
        SuccessSummary_2,
        er2,
        result,
        OptimizedSummary_2)
    # 方向3
    sequence += 1
    er3 = direction3_2_process(datapath, API)
    SuccessSummary_3 = str(sequence) + "." + "In the user migration applications, the selection and usage of " \
        "the DVPP Vdec interface is relatively good, there is no tuning suggestion " \
        "for this aspect."
    OptimizedSummary_3 = str(sequence) + "." + "In user migration applications, the selection and use of DVPP " \
        "VPC interfaces need to be adjusted and optimized."
    result = result_generate(SuccessSummary_3, er3, result, OptimizedSummary_3)
    # 方向4
    sequence += 1
    er4 = direction3_3_process(datapath)
    SuccessSummary_4 = str(sequence) + "." + "In the user migration applications," \
        " the development of AI CPU custom operators is in good condition," \
        "and there is no tuning suggestion for this aspect."
    OptimizedSummary_4 = str(sequence) + "." + "In user migration applications, " \
        "the development of AI CPU custom operators need to be adjusted and optimized."
    result = result_generate(SuccessSummary_4, er4, result, OptimizedSummary_4)
    # 方向5
    sequence += 1
    er5 = direction3_4_process(datapath, API)
    SuccessSummary_5 = str(sequence) + "." + "In the user migration applications, the output of DVPP VPC " \
        "in YUV 400 format is in good condition, and there is no tuning suggestion " \
        "for this aspect."
    OptimizedSummary_5 = str(sequence) + "." + "In user migration applications, DVPP VPC output YUV " \
        "400 format needs to be adjusted and optimized."
    result = result_generate(SuccessSummary_5, er5, result, OptimizedSummary_5)
    return result.generate()


def result_generate(SuccessSummary, er, result, OptimizedSummary):
    if er.data_type != []:
        result.summary += OptimizedSummary
        result.extend_result.append(er)
    else:
        result.summary += SuccessSummary
        result.extend_result.append(er)
    return result


def direction2_1_process(profiling_path):
    ER = ExtendResult()
    # 读取采集到的cpu与内存数据
    try:
        f = open(r"../../data/ctrlCpuAndMemoryData.txt", "r", encoding='utf-8')
    except BaseException:
        ER.type = EXTEND_TYPE['list']
        ER.extend_title = "The memory ECC enable state and the ratio of AI CPU and Ctrl CPU"
        ER.data_type = []
        ER.value.append(
            "Warning! CTRL CPU and memory utilization data have not failed or collected.")
        ER.value.append(
            "Please note that if you need to optimize the CPU ratio and memory enable application in the migration business, "
            "you should use the script file provided by the knowledge base to collect ctrlCpuAndMemoryData.txt文件")
        return ER
    read_txt = f.read()
    f.close()
    # 开始分析
    usage_list = []
    data_list = read_txt.split("end\n")
    if data_list[-1] == '0':  # 若为0,采集数据不成功,另做处理！
        ER.type = EXTEND_TYPE['list']
        ER.extend_title = "The memory ECC enable state and the ratio of AI CPU and Ctrl CPU"
        ER.data_type = []
        ER.value.append(
            "Ctrl CPU and memory utilization data is not successfully collected")
        return ER
    data_list.pop()   # 删除最后的用于判断当前数据是否可用的0/1

    for data in data_list:
        sub_data_list = data.split("\n\n\t")
        one_data_lst = []
        for sub_data in sub_data_list:
            if "NPU ID" in sub_data:
                continue
            last_data_list = sub_data.split("\n\t")
            tmp_lst = []
            for last_data in last_data_list:
                if "Memory Usage Rate" in last_data:
                    mem_usage = last_data.split(":")[1].strip()
                    mem_usage = int(mem_usage)
                    tmp_lst.append(mem_usage)
                if "Ctrlcpu Usage" in last_data:
                    ctrlcpu_usage = last_data.split(":")[1].strip()
                    ctrlcpu_usage = int(ctrlcpu_usage)
                    tmp_lst.append(ctrlcpu_usage)
            one_data_lst.append(tmp_lst)
        usage_list.append(one_data_lst)

    max_mem_usage = 0
    max_ctrlcpu_usage = 0

    device_list = os.listdir(profiling_path)
    device_list = [int(x[-1]) for x in device_list]  # 模型所使用的所有芯片

    for usage_data in usage_list:
        for device in device_list:
            if usage_data[device][0] > max_mem_usage:
                max_mem_usage = usage_data[device][0]
            if usage_data[device][1] > max_ctrlcpu_usage:
                max_ctrlcpu_usage = usage_data[device][1]

    ctrlcpu_need_optimize = 0
    mem_need_optimize = 0

    if max_ctrlcpu_usage > 90:
        ctrlcpu_need_optimize = 1

    if max_mem_usage > 80:
        mem_need_optimize = 1

    if ctrlcpu_need_optimize == 1 and mem_need_optimize == 1:
        ER.type = EXTEND_TYPE['table']
        ER.extend_title = "The memory ECC enable state and the ratio of AI CPU and Ctrl CPU"
        ER.data_type.append(extend_data_type['str'])
        ER.key.append("The suggestions of AI CPU and Ctrl CPU ratio")
        value = []
        value.append(
            "It is suggested to modify the ratio of AI CPU to Ctrl CPU to increase the proportion of Ctrl CPU")
        ER.value.append(value)
        ER.data_type.append(extend_data_type['str'])
        ER.key.append("The suggestions for Memory ECC Enable Status")
        value1 = []
        value1.append(
            "It is suggested to modify the memory ECC enable state in the Ascend 310p environment")
        ER.value.append(value1)
        return ER

    if ctrlcpu_need_optimize == 1:
        ER.type = EXTEND_TYPE['list']
        ER.extend_title = "The memory ECC enable state and the ratio of AI CPU and Ctrl CPU "
        ER.data_type.append(EXTEND_DATA_TYPE["str"])
        ER.value.append(
            "It is suggested to modify the ratio of AI CPU to Ctrl CPU to increase the number of Ctrl CPUs")
        return ER

    if mem_need_optimize == 1:
        ER.type = EXTEND_TYPE['list']
        ER.extend_title = "The memory ECC enable state and the ratio of AI CPU and Ctrl CPU "
        ER.data_type.append(EXTEND_DATA_TYPE["str"])
        ER.value.append(
            "It is suggested to modify the memory ECC enable state in the Ascend 310p environment")
        return ER
    if ctrlcpu_need_optimize == 0 and mem_need_optimize == 0:
        ER.type = EXTEND_TYPE['list']
        ER.extend_title = "The memory ECC enable state and the ratio of AI CPU and Ctrl CPU "
        ER.data_type = []
        ER.value.append(
            "The ratio of AI CPU to Ctrl CPU, memory ECC enabled status is normal, there is no tuning suggestion")
        return ER
    return ER


def direction3_1_process(profiling_path, API):
    ER = ExtendResult()
    acl_statistic_data = function.get_statistic_profile_data(profiling_path)
    #statistic_contend = acl_statistic_data.readline()
    data = pd.read_csv(acl_statistic_data)
    countCrop = 0
    countResize = 0
    countCropResize = 0
    countCropPaste = 0
    countCropResizePaste = 0
    countMakeBorder = 0
    for line in data.itertuples():
        if (line[1] == API["Crop"]):
            countCrop = line[5]
        if (line[1] == API["Resize"]):
            countResize = line[5]
        if (line[1] == API["CropResize"]):
            countCropResize = line[5]
        if (line[1] == API["CropPaste"]):
            countCropPaste = line[5]
        if (line[1] == API["CropResizePaste"]):
            countCropResizePaste = line[5]
        if (line[1] == API["MakeBorder"]):
            countMakeBorder = line[5]
    if (countCrop != 0 or countResize != 0 or countCropResize !=
            0 or countCropPaste != 0 or countCropResizePaste != 0 or countMakeBorder != 0):
        ER.type = EXTEND_TYPE["list"]
        ER.extend_title = "The selection and use of the DVPP VPC interface"
        ER.data_type = [EXTEND_DATA_TYPE["str"]]
        ER.value.append("On this AI processor, VPCAPI is already used")
        if (countCrop >= 2):
            ER.value.append(
                "It is detected that the " +
                API["Crop"] +
                "interface is used, and the picture "
                "is processed cyclically. It is suggested to use the " +
                API["CropBatch"] +
                " interface")
        if (countCrop >= 2 and countResize >= 2 and countCrop == countResize):
            ER.value.append(
                "It is detected that the " +
                API["Crop"] +
                "and" +
                API["Resize"] +
                "interfaces are used, and the picture is processed "
                "cyclically.It is suggested to use the " +
                API["CropResizeBatch"] +
                " interface")
        if (countCropResize >= 2):
            ER.value.append(
                "It is detected that the " +
                API["CropResize"] +
                "interface is used, and the picture is processed "
                "cyclically.It is suggested to use the " +
                API["CropResizeBatch"] +
                " interface")
        if (countCropPaste >= 2):
            ER.value.append(
                "It is detected that the " +
                API["CropPaste"] +
                "interface is used, and the picture is processed "
                "cyclically.It is suggested to use the " +
                API["CropPasteBatch"] +
                " interface")
        if (countCropResizePaste >= 2):
            ER.value.append(
                "It is detected that the " +
                API["CropResizePaste"] +
                "interface is used, and the picture is processed "
                "cyclically.It is suggested to use the " +
                API["CropResizePasteBatch"] +
                " interface")
        if (countMakeBorder >= 2 and (countCrop != 0 or countResize != 0) and (
                countMakeBorder == countCrop or countMakeBorder == countResize)):
            ER.value.append(
                "It is detected that the " +
                API["Crop"] +
                "and " +
                API["Resize"] +
                "and " +
                API["MakeBorder"] +
                "interface is used, and the picture is processed "
                "cyclically.It is suggested to use the " +
                API["MakeBorderBatch"] +
                " interface")
        return ER
    else:
        ER.type = EXTEND_TYPE["list"]
        ER.extend_title = "The selection and use of the DVPP VPC interface"
        ER.data_type = []
        ER.value.append(
            "On this AI processor, the VPCAPI interface may not be used, "
            "so in this direction, the knowledgebase has no tuning suggestion")
        return ER
    return ER


def direction3_2_process(profiling_path, API):
    ER = ExtendResult()
    acl_statistic_data = function.get_statistic_profile_data(profiling_path)
    data = pd.read_csv(acl_statistic_data)
    countVpcCCA = 0
    countVdecSF = 0
    for line in data.itertuples():
        if (line[1] == API["VdecCCA"]):
            countVpcCCA = line[5]
        if (line[1] == API["VdecSF"]):
            countVdecSF = line[5]
    if (countVpcCCA != 0 or countVdecSF != 0):
        ER.type = EXTEND_TYPE["list"]
        ER.extend_title = "The selection and use of the DVPP VDEC interface"
        ER.data_type = [EXTEND_DATA_TYPE["str"]]
        ER.value.append("On this AI processor, VDECAPI has been used")
        if (countVpcCCA >= 1 & countVdecSF == 0):
            ER.value.append(
                "Detected that the " + API["VdecCCA"] + " interface is used")
            ER.value.append(
                "If you are using the Ascend 710 AI processor"
                "The processor's video decoding interface " +
                API["VdecSF"] +
                " supports output in YUV420SP format "
                "or RGB888 format, and interface parameters can be set to output different formats"
                "It is suggested to omit the step of calling " +
                API["VdecCCA"] +
                " for format conversion "
                "and reduce interface calls")
            ER.value.append(
                "At the same time, in the scenario of video decoding + model inference, if the number of video frames is "
                "large, and not every frame needs to be inferred, for frames that do not need inference"
                "It is suggested that you use the " + API["VdecSkip"] + " interface for decoding, "
                "and the decoding result is not output.")
        if (countVdecSF >= 1 & countVpcCCA == 0):
            ER.value.append(
                "Detected the use of the " + API["VdecSF"] + " interface")
            ER.value.append(
                "On the Ascend 710 AI processor"
                "VPC image processing function supports output YUV400 format (grayscale image)"
                "If the input image for model inference is a grayscale image, it is suggested that "
                "you use the VPC function directly instead of using the AIPP color gamut conversion function.")
            ER.value.append(
                "At the same time, in the scenario of video decoding + model inference, if the number of video frames is "
                "large, and not every frame needs to be inferred, for frames that do not need inference"
                "It is suggested that you use the " + API["VdecSkip"] + " interface for decoding, "
                "and the decoding result is not output.")
        if (countVpcCCA >= 1 & countVdecSF >= 1):
            ER.value.append(
                "Detect that both the " +
                API["VdecCCA"] +
                " interface and the " +
                API["VdecSF"] +
                " interface "
                "are both used at the same time")
            ER.value.append(
                "On the Ascend 710 AI processor"
                "The video decoding interface " +
                API["VdecSF"] +
                " supports output in YUV420SP format or RGB888 format, "
                "and you can set interface parameters to output different formats"
                "It is suggested to omit the step of calling " +
                API["VdecCCA"] +
                ""
                " for format conversion and reduce interface calls")
            ER.value.append(
                "At the same time, in the scenario of video decoding + model inference, if the number of video frames "
                "is large, and not every frame needs to be inferred, for frames that do not need inference"
                "It is suggested that you use the " + API["VdecSkip"] + " interface for decoding, "
                "and the decoding result is not output.")
        return ER
    else:
        ER.type = EXTEND_TYPE["list"]
        ER.extend_title = "The selection and use of the DVPP VDEC interface"
        ER.data_type = []
        ER.value.append(
            "The VDECAPI interface is not used on this AI processor"
            "so the knowledge base has no tuning suggestions in this direction.")
        ER.value.append(
            "But in the scenario of video decoding + model inference,"
            "if the user video has a large number of frames and not every frame needs to be inferred,"
            "it is suggested that you use the " +
            API["VdecSkip"] +
            " interface to improve the experience")
        return ER
    return ER


def direction3_3_process(profiling_path):
    ER = ExtendResult()
    project_path = function.check_project_data("../../data/project")
    if project_path == -1:
        ER.type = EXTEND_TYPE['list']
        ER.extend_title = "The development of AI CPU custom operators"
        ER.data_type = []
        ER.value.append(
            "Warning! Can't find project documents.\n"
            "Please note that if you need to optimize the Optimization of the Migration Business, "
            "the Project file should be placed in the data file directory")
        return ER
    om_json_path = function.find_om_json("../../data/project")
    if om_json_path == -1:
        ER.type = EXTEND_TYPE['list']
        ER.extend_title = "The development of AI CPU custom operators"
        ER.data_type = []
        ER.value.append(
            "Warning! Can't find the model JSON information.\n "
            "Please note that if you need to optimize the operator part in the migration business, "
            "the OM format model should be converted to JSON format and placed in the same position as the OM model")
        return ER
    # 将自定义算子存入集合中，若无自定义算子，直接返回空
    custom_op = set()
    om_info = function.get_data(om_json_path)
    op_info_list = om_info.get("graph")[0]["op"]
    for op_info in op_info_list:
        attrs = op_info.get("attr")
        for attr in attrs:
            if attr.get("key", 0) == "_is_custom_op":
                if attr.get("value").get("b"):
                    custom_op.add(op_info.get("name"))
    if len(custom_op) == 0:
        ER.type = EXTEND_TYPE['list']
        ER.extend_title = "The development of AI CPU custom operators"
        ER.data_type = []
        ER.value.append(
            "AI CPU custom operators are not detected")
        return ER

    # 保存相关数据路径
    json_list = []
    csv_list = []
    for device in os.listdir(profiling_path):
        summary_path = os.path.join(profiling_path, device, "summary")
        timeline_path = os.path.join(profiling_path, device, "timeline")
        for file in os.listdir(timeline_path):
            if "task_time" in file and os.path.splitext(file)[-1] == ".json":
                json_list.append(os.path.join(timeline_path, file))
        for file in os.listdir(summary_path):
            if "task_time" in file and os.path.splitext(file)[-1] == ".csv":
                csv_list.append(os.path.join(summary_path, file))

    # 处理数据
    ai_cpu_ops_infos = []  # 用于存放json文件中ai cpu算子的时间信息
    aicore_time_infos = []  # 用于存放json文件中ai core算子的时间信息
    task_starttime_list = []
    for task in json_list:
        task_start_time = 0
        task_start_time_flag = 0
        task_data = function.get_data(task)
        ai_cpu_ops_info = []
        aicore_time_info = []
        for task in task_data:
            if task["args"].get("Task Type", -1) == -1:
                continue
            if task_start_time_flag == 0:
                task_start_time = task["ts"] / 1000  # 单位转换为ms
                task_starttime_list.append(task_start_time)
                task_start_time_flag = 1
            if task["args"].get("Task Type", -1) == "AI_CORE":
                aicore_time_info.append([task["ts"] /
                                         1000 -
                                         task_start_time, task["dur"] /
                                         1000 +
                                         task["ts"] /
                                         1000 -
                                         task_start_time])
            elif task["args"].get("Task Type", -1) == "AI_CPU":
                ai_cpu_ops_info.append([task["name"], [task["ts"] /
                                                       1000 -
                                                       task_start_time, task["dur"] /
                                                       1000 +
                                                       task["ts"] /
                                                       1000 -
                                                       task_start_time]])
        aicore_time_infos.append(aicore_time_info)
        ai_cpu_ops_infos.append(ai_cpu_ops_info)

    ai_cpu_ops_infos2 = []  # 用于存放csv文件中的aicpu算子时间信息

    try:  # 若task_time.csv文件中的数据为可利用的格式，则亦可提取AI CPU算子时间数据
        for task in csv_list:
            ai_cpu_ops_info2 = []
            task_data = pd.read_csv(task)
            ops_name = task_data["kernel_name"]
            ops_type = task_data["kernel_type"]
            ops_start_time = task_data["task_start(ns)"]
            ops_end_time = task_data["task_stop(ns)"]
            nan = pd.isnull(task_data["kernel_name"])
            task_start_time = task_starttime_list[0]
            task_starttime_list.pop(0)

            for index in range(len(ops_type)):
                if ops_type[index] == "KERNEL_AICPU" and not nan[index]:
                    op_start_time = float(
                        ops_start_time[index][1:-1]) / 1000000 - task_start_time
                    op_end_time = float(
                        ops_end_time[index][1:-1]) / 1000000 - task_start_time
                    ai_cpu_ops_info2.append(
                        [ops_name[index], [op_start_time, op_end_time]])
            ai_cpu_ops_infos2.append(ai_cpu_ops_info2)
    except BaseException:
        pass

    # 将aicore算子运行过程中相交的时间段结合起来
    for aicore_time_info in aicore_time_infos:
        index = 1
        while (index < len(aicore_time_info)):
            if aicore_time_info[index][0] < aicore_time_info[index - 1][1]:
                aicore_time_info[index - 1][1] = aicore_time_info[index][1]
                aicore_time_info.pop(index)
            else:
                index += 1

    # 处理json文件中的aicpu算子时间信息，计算每一个aicpu算子与aicore算子并行率
    for index in range(len(ai_cpu_ops_infos)):
        ai_cpu_ops_info = ai_cpu_ops_infos[index]
        aicore_time_info = aicore_time_infos[index]
        for i in range(len(ai_cpu_ops_info)):
            parallel_time = 0
            key, ai_cpu_time = ai_cpu_ops_info[i]
            for ai_core_time in aicore_time_info:

                aicpu_start_time = ai_cpu_time[0]
                aicpu_end_time = ai_cpu_time[1]

                aicore_start_time = ai_core_time[0]
                aicore_end_time = ai_core_time[1]

                if aicore_start_time >= aicpu_end_time:
                    break
                if aicpu_start_time >= aicore_end_time:
                    continue

                if aicpu_start_time >= aicore_start_time and aicpu_end_time <= aicore_end_time:
                    parallel_time += aicpu_end_time - aicpu_start_time
                elif aicpu_start_time >= aicore_start_time and aicpu_end_time >= aicore_end_time:
                    parallel_time += aicore_end_time - aicpu_start_time
                elif aicpu_start_time <= aicore_start_time and aicpu_end_time <= aicore_end_time:
                    parallel_time += aicpu_end_time - aicore_start_time
                elif aicpu_start_time <= aicore_start_time and aicpu_end_time >= aicore_end_time:
                    parallel_time += aicore_end_time - aicore_start_time

            ai_cpu_ops_info[i][1].append(
                parallel_time / (ai_cpu_ops_info[i][1][1] - ai_cpu_ops_info[i][1][0]))

    # 处理csv文件中的aicpu算子时间信息，计算每一个aicpu算子与aicore算子并行率
    for index in range(len(ai_cpu_ops_infos2)):
        ai_cpu_ops_info2 = ai_cpu_ops_infos2[index]
        aicore_time_info = aicore_time_infos[index]
        for i in range(len(ai_cpu_ops_info2)):
            parallel_time = 0
            key, ai_cpu_time = ai_cpu_ops_info2[i]
            for ai_core_time in aicore_time_info:

                aicpu_start_time = ai_cpu_time[0]
                aicpu_end_time = ai_cpu_time[1]

                aicore_start_time = ai_core_time[0]
                aicore_end_time = ai_core_time[1]

                if aicore_start_time >= aicpu_end_time:
                    break
                if aicpu_start_time >= aicore_end_time:
                    continue

                if aicpu_start_time >= aicore_start_time and aicpu_end_time <= aicore_end_time:
                    parallel_time += aicpu_end_time - aicpu_start_time
                elif aicpu_start_time >= aicore_start_time and aicpu_end_time >= aicore_end_time:
                    parallel_time += aicore_end_time - aicpu_start_time
                elif aicpu_start_time <= aicore_start_time and aicpu_end_time <= aicore_end_time:
                    parallel_time += aicpu_end_time - aicore_start_time
                elif aicpu_start_time <= aicore_start_time and aicpu_end_time >= aicore_end_time:
                    parallel_time += aicore_end_time - aicore_start_time

            ai_cpu_ops_info2[i][1].append(
                parallel_time / (ai_cpu_ops_info[i][1][1] - ai_cpu_ops_info[i][1][0]))

    # 将阻塞的算子提取出来
    set_aicpu_block = set()  # 用于保存阻塞的aicpu算子
    for ai_cpu_ops_info in ai_cpu_ops_infos:
        for aicpuinfo in ai_cpu_ops_info:
            if aicpuinfo[1][-1] < 0.8:
                set_aicpu_block.add(aicpuinfo[0])
    for ai_cpu_ops_info2 in ai_cpu_ops_infos2:
        for aicpuinfo in ai_cpu_ops_info2:
            if aicpuinfo[1][-1] < 0.8:
                set_aicpu_block.add(aicpuinfo[0])

    aicpu_custom_block_op = set_aicpu_block & custom_op  # 对自定义aicpu算子和所有阻塞的aicpu算子取交集
    # 即为阻塞的aicpu自定义算子

    if len(aicpu_custom_block_op) == 0:
        # 无阻塞的aicpu自定义算子
        ER.type = EXTEND_TYPE['list']
        ER.extend_title = "The development of AI CPU custom operators"
        ER.data_type = []
        ER.value.append(
            "AI CPU custom operators are running well")
        return ER
    else:
        ER.type = EXTEND_TYPE['list']
        ER.extend_title = "The development of AI CPU custom operators"
        ER.data_type.append(EXTEND_DATA_TYPE["str"])
        ER.value.append(
            "It is suggested to implement " +
            str(aicpu_custom_block_op) +
            " ai CPU custom operators in parallel execution of multi-processor")
        return ER
    return ER


def direction3_4_process(profiling_path, API):
    root_path = r'../../data/project/'
    ER = ExtendResult()
    is_vpc_used = 0

    for device in os.listdir(profiling_path):
        path = os.path.join(profiling_path, device, "summary")
        for file in os.listdir(path):
            if "acl_statistic" in file:
                data = pd.read_csv(os.path.join(path, file))
                for api in data["Name"]:
                    if "Vpc" in api or "vpc" in api:
                        is_vpc_used = 1
    if is_vpc_used == 0:
        ER.type = EXTEND_TYPE["list"]
        ER.extend_title = "DVPP VPC output YUV 400 format"
        ER.data_type = []
        ER.value = [
            "VPC is not be used"]
        return ER

    cfg_list = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.split('.')[-1] == 'cfg':
                cfg_list.append(os.path.join(root, file))
    if len(cfg_list) == 0:
        ER.type = EXTEND_TYPE["list"]
        ER.extend_title = "DVPP VPC output YUV 400 format"
        ER.data_type = []
        ER.value = [
            "There is no action files containing AIPP in the path ../../data/project,the function of this direction may not be used"]
        return ER

    csc_cfgs = {"matrix_r0c0": "256",
                "matrix_r0c1": "0",
                "matrix_r0c2": "0",
                "matrix_r1c0": "0",
                "matrix_r1c1": "0",
                "matrix_r1c2": "0",
                "matrix_r2c0": "0",
                "matrix_r2c1": "0",
                "matrix_r2c2": "0",
                "input_bias_0": "0",
                "input_bias_1": "0",
                "input_bias_2": "0"}
    aipp_cfg = ""
    for cfg in cfg_list:
        f = open(cfg, 'r', encoding='utf-8')
        cntxt = f.readline()
        while (cntxt == '\n'):
            cntxt = f.readline()
        if "aipp_op" not in cntxt:
            cfg_list.remove(cfg)
            f.close
        else:
            aipp_cfg = cfg
            f.close

    if len(cfg_list) > 1:
        ER.type = EXTEND_TYPE["list"]
        ER.extend_title = "DVPP VPC output YUV 400 format"
        ER.data_type = []
        ER.value = [
            "An error occurred, reason: There are multiple action files containing AIPP in the path ../../data/project"]
        return ER
    f = open(aipp_cfg, 'r', encoding='utf-8')
    cntxt_list = f.readlines()
    f.close()

    is_static = 1
    is_true = 1
    for cntxt in cntxt_list:
        if cntxt == "\n":
            continue
        if "aipp_op" in cntxt:
            continue
        if "}" in cntxt:
            continue
        if "aipp_mode" in cntxt and "static" not in cntxt:
            is_static = 0
            break
        if "input_format" in cntxt and "YUV420SP_U8" not in cntxt:
            is_true = 0
            break
        if "csc_switch" in cntxt and "false" in cntxt:
            is_true = 0
            break

        ret = re.match("[\\s]*[a-zA-Z0-9_]*[ ]?:[ ]?[a-zA-Z0-9_-]*\n", cntxt)
        i = 0
        ret_list = list(ret.group())
        if ret_list[-1] == '\n':
            ret_list.pop(-1)
        ret_list = [x for x in ret_list if x != ' ' and x != '\t']
        csc_key = "".join(ret_list).split(":")[0]
        csc_value = "".join(ret_list).split(":")[1]
        if (csc_cfgs.get(csc_key, "NULL") !=
                "NULL" and csc_cfgs[csc_key] != csc_value):
            is_true = 0

    if is_true == 0:
        ER.type = EXTEND_TYPE["list"]
        ER.extend_title = "DVPP VPC output YUV 400 format"
        ER.data_type = []
        ER.value = [
            "On the Ascend 310p AI processor,the image of this VPC has been output reasonably "]
        return ER

    if is_static:
        ER.type = EXTEND_TYPE["list"]
        ER.extend_title = "DVPP VPC output YUV 400 format"
        ER.data_type = [EXTEND_DATA_TYPE["str"]]
        ER.value = [
            "On the Ascend 310p AI processor, "
            "VPC can directly output images in YUV400 format without any color gamut conversion"]
        return ER
    else:
        src_list = []
        for root, dirs, files in os.walk(root_path):
            for file in files:
                code_suffix = file.split('.')[1]
                if code_suffix == 'cpp' or code_suffix == 'C' or code_suffix == 'cc' or code_suffix == 'py':
                    src_list.append(os.path.join(root, file))
        if len(src_list) == 0:
            ER.type = EXTEND_TYPE["list"]
            ER.extend_title = "DVPP VPC output YUV 400 format"
            ER.data_type = [EXTEND_DATA_TYPE["str"]]
            ER.value = [
                "C++ or python source files not found in project files"]
            return ER
        for file in src_list:
            f = open(file, "r", encoding='utf-8')
            read_txt = f.read()
            f.close()

            if os.path.splitext(os.path.split(file)[1])[1] == "py":
                input_api = API["AippInputFormat_py"]
                csc_api = API["AippCscParams_py"]
            else:
                input_api = API["AippInputFormat_cpp"]
                csc_api = API["AippCscParams_cpp"]

            is_input_yuv420 = 0
            indexs = [each.end() for each in re.finditer(input_api, read_txt)]
            for index in indexs:
                start = 0
                end = 0
                while (index < len(read_txt)):
                    if read_txt[index] == '(':
                        start = index + 1
                    if read_txt[index] == ')':
                        end = index
                        break
                    index += 1
                if "YUV420SP" in read_txt[start: end]:
                    is_input_yuv420 = 1
                    break

            if is_input_yuv420:
                is_csc_yuv400 = 0
                indexs = [each.end()
                          for each in re.finditer(csc_api, read_txt)]
                for index in indexs:
                    start = 0
                    end = 0
                    while (index < len(read_txt)):
                        if read_txt[index] == '(':
                            start = index + 1
                        if read_txt[index] == ')':
                            end = index
                            break
                        index += 1
                    csc_parm = read_txt[start: end].split(',')
                    csc_parm = [x.strip() for x in csc_parm]
                    if csc_parm[1] == '0':
                        continue
                    if csc_parm[2] != '256':
                        continue
                    flag_tmp = 1
                    for parm in csc_parm[3:]:
                        if parm != '0':
                            flag_tmp = 0
                            break
                    if flag_tmp:
                        is_csc_yuv400 = 1
                        break

                if is_csc_yuv400:
                    ER.type = EXTEND_TYPE["list"]
                    ER.extend_title = "DVPP VPC output YUV 400 format"
                    ER.data_type = [EXTEND_DATA_TYPE["str"]]
                    ER.value = [
                        "On the Ascend 310p AI processor, VPC can directly output images in YUV400 format without any color gamut conversion"]
                    return ER
                else:
                    ER.type = EXTEND_TYPE["list"]
                    ER.extend_title = "DVPP VPC output YUV 400 format"
                    ER.data_type = []
                    ER.value = [
                        "On the Ascend 310p AI processor, the image of this VPC has been output reasonably in YUV400 format"]
                    return ER
        return ER
