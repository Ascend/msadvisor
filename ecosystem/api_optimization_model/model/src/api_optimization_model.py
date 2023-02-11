import os
import time
import sys
import json

# define datatype

class_type = {'op': '0', 'model': '1'}
error_code = {'success': '0', 'optimized': '1'}
extend_type = {'list': '0', 'table': '1', 'sourcedata': '2'}
extend_data_type = {'str': '0', 'int': '1', 'double': '2'}
invalid_result = ""


class ExtendResult:
    def __init__(self):
        self.type = '0'
        self.extend_title = ""
        self.data_type = []      # table type is an array with multiple elements, list type with only one element
        self.key = []           # this field is only used for table type result
        self.value = []         # table type is a two-dimensional array, list type is a one-dimensional array


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
        outputstr = json.dumps(res, indent='\t')
        return output_str




def evaluate(dataPath, parameter):
    profilepath = os.path.join(dataPath, 'profiler')
    if os.path.isdir(profilepath):
        subdirs = os.listdir(profilepath)
        if subdirs:
            profilepath = os.path.join(os.path.join(profilepath, subdirs[0]), 'device_0')
    project_dir = os.path.join(dataPath, 'project')

    acl_profile_fp = get_profile_fp(profilepath)
    acl_statistic_fp = get_statistic_profile_fp(profilepath)
    extend_result = init_extent_result()
    if acl_profile_fp and acl_statistic_fp:
        extend_result = process_memory_suggestions(acl_profile_fp, acl_statistic_fp, extend_result)
    elif acl_profile_fp:
        extend_result = process_profiling_file(acl_profile_fp, extend_result)
    elif acl_statistic_fp:
        extend_result = process_profiling_file_with_json(acl_statistic_fp, extend_result)
        extend_result = process_async_suggestions(acl_statistic_fp, extend_result)
    extend_result = datatype_process(project_dir, extend_result)

    if acl_profile_fp:
        acl_profile_fp.close()
    if acl_statistic_fp:
        acl_statistic_fp.close()

    result = class_result.Result()
    return result_parse(result, extend_result)


# read profiling file
def get_profile_fp(profilepath):
    try:
        return open(profilepath + "/summary/acl_0_1_1.csv")
    except IOError:
        return None

def get_statistic_profile_fp(profilepath):
    try:
        return open(profilepath + "/summary/acl_statistic_0_1_1.csv")
    except IOError:
        return None


def get_code_data(codepath):
    return open(codepath)


# define extend_result
def init_extent_result():
    extend_result = class_result.ExtendResult()
    extend_result.type = extend_type['table']
    extend_result.data_type.append(extend_data_type['str'])
    extend_result.data_type.append(extend_data_type['str'])
    extend_result.data_type.append(extend_data_type['str'])
    extend_result.key.append("API Name")
    extend_result.key.append("Optimization Suggestion")
    extend_result.key.append("API Location")
    return extend_result


# Device管理、Context管理、内存管理
def process_profiling_file(profile_fp, extend_result):
    stream_num = 0
    max_stream = 0
    aclrtSetDevice_num = 0
    aclrtCreateContext_first = 0
    for line in profile_fp.readlines():
        if aclrtSetDevice_num == 0:
            if line.count('aclrtCreateContext') and aclrtCreateContext_first == 0:
                stream_num += 2
                aclrtCreateContext_first = 1
            elif line.count('aclrtCreateStream') or line.count('aclrtCreateContext'):
                stream_num += 1
            if stream_num > max_stream:
                max_stream = stream_num
            elif line.count('aclrtDestroyStream') and stream_num > 0:
                stream_num -= 1
        else:
            if line.count('aclrtCreateStream') or line.count('aclrtCreateContext') or line.count(
                    'aclrtSetDevice'):
                stream_num += 1
            if stream_num > max_stream:
                max_stream = stream_num
            elif line.count('aclrtDestroyStream') and stream_num > 0:
                stream_num -= 1

    if stream_num >= 1024:
        value = []
        value.append("aclrtCreateStream")
        value.append(f"The max num of streams is 1024, current is {stream_num}")
        value.append('-')
        extend_result.value.append(value)
    return extend_result


def process_memory_suggestions(profile_fp, statistic_fp, extend_result):
    profile_fp.seek(0)
    statistic_fp.seek(0)
    if all(map(lambda l: 'aclrtMemcpy' not in l.split(',')[0], statistic_fp.readlines())):
        return extend_result

    can_access_peer_call = False
    enable_peer_access_call = False
    for line in profile_fp.readlines():
        api = line.split(',')[0]
        if 'aclrtDeviceCanAccessPeer' in api:
            can_access_peer_call = True
        if 'aclrtDeviceEnablePeerAccess' in api:
            enable_peer_access_call = True
        if can_access_peer_call and enable_peer_access_call:
            return extend_result
        if 'aclrtMemcpy' in api:
            break

    value = []
    value.append("aclrtMemcpy")
    if not can_access_peer_call and not enable_peer_access_call:
        value.append(
            "Please use aclrtDeviceCanAccessPeer and aclrtDeviceEnablePeerAccess to check whether supported memory copy")
    elif not can_access_peer_call:
        value.append(
            "Please use aclrtDeviceCanAccessPeer to check whether supported memory copy")
    elif not enable_peer_access_call:
        value.append(
            "Please use aclrtDeviceEnablePeerAccess to check whether supported memory copy")
    value.append('-')
    extend_result.value.append(value)
    return extend_result


def process_async_suggestions(statistic_fp, extend_result):
    statistic_fp.seek(0)
    line_num_stack = []
    for line_num, line in enumerate(statistic_fp.readlines()):
        api = line.split(',')[0]
        if 'aclmdlExecuteAsync' in api:
            line_num_stack.append(line_num)
        if 'aclrtSynchronizeStream' in api:
            backtrace = line_num
            while len(line_num_stack) > 0 and backtrace == line_num_stack[-1] + 1:
                backtrace = line_num_stack.pop()

    if len(line_num_stack) == 0:
        return extend_result

    value = []
    value.append("aclmdlExecuteAsync")
    value.append('Please use aclrtSynchronizeStream to block the Host run')
    value.append('Line:' + ', '.join(map(str, line_num_stack)))
    extend_result.value.append(value)
    return extend_result


# 昇腾310 AI处理器媒体数据处理V1->昇腾310P AI处理器媒体数据处理V1迁移指引
def process_profiling_file_with_json(profile_fp, extend_result):
    for line_num, line in enumerate(profile_fp.readlines()):
        api = str(line).split(',')[0]
        if api in V1_transformer.keys():
            value = []
            value.append(api)
            value.append(V1_transformer[api])
            value.append(f'Line:{line_num}')
            extend_result.value.append(value)
    return extend_result

def result_parse(result, extend_result):
    if not extend_result.value:
        result.class_type = class_type['op']
        result.error_code = error_code['optimized']
        result.summary = "310B API operations are well optimized"
        return result.generate()
    result.class_type = class_type['op']
    result.error_code = error_code['success']
    result.summary = "310B API operations need to be optimized"
    result.extend_result.append(extend_result)
    return result.generate()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('path argument required!')
    print(evaluate(sys.argv[1], 1))