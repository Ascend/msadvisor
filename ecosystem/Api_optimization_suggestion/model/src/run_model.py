import os
import time

import class_result

# define datatype

class_type = {'op': '0', 'model': '1'}
error_code = {'success': '0', 'optimized': '1'}
extend_type = {'list': '0', 'table': '1', 'sourcedata': '2'}
extend_data_type = {'str': '0', 'int': '1', 'double': '2'}

V1_transformer = {
    "acldvppSetPicDescWidth": "For yuv420sp format, both width and height need 2 alignment, and for yuv422sp or yuv422packed format,the width needs 2 alignment",
    "acldvppSetPicDescHeight": "For yuv420sp format, both width and height need 2 alignment, and for yuv440sp format, the height needs 2 alignment",
    "acldvppSetPicDescSize": "Ascend 710:Support yuv400 format image processing, directly set the format of the input image to yuv400,and the memory size to widthstripe*heightslide. VPC will verify the memory size according to the image format",
    "aclvdecSetChannelDescOutPicWidth": "Input code stream buffer size: maximum width of decoded code stream * maximum height of decoded code stream *2,Ascend 710:calculate the cache size of the input code stream using the formula, needs to modify the code and call the aclvdecsetchanneldescoutpicwidth and aclvdecsetchanneldescoutpicheight interfaces to set the correct width and height.",
    "aclvdecSetChannelDescOutPicHeight": "Input code stream buffer size: maximum width of decoded code stream * maximum height of decoded code stream *2,Ascend 710:calculate the cache size of the input code stream using the formula, needs to modify the code and call the aclvdecsetchanneldescoutpicwidth and aclvdecsetchanneldescoutpicheight interfaces to set the correct width and height.",
    "aclvdecSetChannelDescRefFrameNum": "The default value of reference frame is 8, which is compatible with decoding most code streams, but for code streams with a large number of reference frames, decoding may fail.",
    "aclvdecSetChannelDescBitDepth": "For a 10bit code stream, if it is not set, decoding may fail. You need to call the aclvdecsetchanneldescbitdepth interface to set the bit width to 10bit",
    "aclvencSetChannelDescBufAddr": "The user needs to set the output buffer, and there is no need to copy the encoded output results again.",
    "aclvencSetChannelDescMaxBitRate": "If the user does not explicitly call the aclvencsetchanneldescmaxbitrate interface or aclvencsetchanneldescparam interface, the encoder will use the default output code rate for encoding. After migrating to ascend 710, you need to explicitly call the aclvencsetchanneldescmaxbitrate interface or aclvencsetchanneldescparam interface to set the output code rate to 300, otherwise the default value of 2000 will be used on ascend 710.",
    "acldvppSetChannelDescMode": "It supports setting the channel mode. If the channel mode is not set, the channel of vpc+jpegd+jpege+pngd will be created by default when creating the channel, which may occupy resources.",
    "acldvppCreateChannel": "710: vdec and jpegd share the channel number and support 256 channels at most. VPC supports 256 channels at most. Jpege and Venc share the same channel and the maximum number of channels is 128.",
    "aclvdecCreateChanne": "710: vdec and jpegd share the channel number and support 256 channels at most. VPC supports 256 channels at most. Jpege and Venc share the same channel and the maximum number of channels is 128.",
    "aclvencCreateChannel": "710: vdec and jpegd share the channel number and support 256 channels at most. VPC supports 256 channels at most. Jpege and Venc share the same channel and the maximum number of channels is 128."
}




task_data={
"ACL_VENC_MAX_BITRATE_UINT32": "310:range 0 or [10,30000] default 300,if set 0,means use default value 300.910:range 0 or [10,30000] default 300,if set 0,means use default value 300.710:range [2,614400] default 2000,if set 0,means use default value 2000.",
"ACL_VENC_MAX_IP_PROP_UINT32":"Ratio of the number of bits of a single I frame to the number of bits of a single P frame in a GOP,range 0 or [1,100] .If this parameter is not set,VBR mode default 80, CBR mode default 70.If set 0,use default value.",
"ACL_VENC_BUF_SIZE_UINT32": "710:Default=8M,Min=5M.310 and 910:not support to set this parameter,default 3686400 Byte.",
"ACL_VENC_RC_MODE_UINT32": "default 0.310:value 0 equal CBR mode.910:value 0 equal CBR mode.710:value 0 means VBR mode",
"ACL_VENC_SRC_RATE_UINT32": "310:range 0 or [1,120].910:range 0 or [1,120].710:range 0 or [1,240].If this parameter is not set,default is 30.If set 0,use default 30.If the difference between this value and the actual input bitstream frame rate is too large, the output bitrate will be affected."}


def evaluate(dataPath,parameter):
    profilepath = dataPath + "/profiler/"+os.listdir(dataPath + "/profiler")[0]+'/device_0'
    project_dir = dataPath + '/project'
    result = class_result.Result()
    acl_profile_data = get_profile_data(profilepath)
    acl_statistic_data = get_statistic_profile_data(profilepath)
    extend_result = init_extent_result()
    extend_result = process_profiling_file(acl_profile_data, acl_statistic_data, extend_result)
    extend_result = process_profiling_file_with_json(acl_statistic_data, extend_result)
    extend_result = datatype_process(project_dir, extend_result)
    return result_parse(result, extend_result)


# read profiling file
def get_profile_data(profilepath):
    return open(profilepath + "/summary/acl_0_1_1.csv")


def get_statistic_profile_data(profilepath):
    return open(profilepath + "/summary/acl_statistic_0_1_1.csv")


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
def process_profiling_file(profile_data, statistic_profile_data, extend_result):
    stream_num = 0
    max_stream = 0
    aclrtSetDevice_num = 0
    aclrtCreateContext_first = 0
    contend = profile_data.readline()
    while contend:
        if aclrtSetDevice_num == 0:
            if contend.count('aclrtCreateContext') and aclrtCreateContext_first == 0:
                stream_num += 2
                aclrtCreateContext_first = 1
            elif contend.count('aclrtCreateStream') or contend.count('aclrtCreateContext'):
                stream_num += 1
            if stream_num > max_stream:
                max_stream = stream_num
            elif contend.count('aclrtDestroyStream') and stream_num > 0:
                stream_num -= 1
            if stream_num >= 1024:
                value = []
                value.append("aclrtCreateStream")
                value.append("The max num of streams is 1024")
                value.append('-')
                extend_result.value.append(value)
        else:
            if contend.count('aclrtCreateStream') or contend.count('aclrtCreateContext') or contend.count(
                    'aclrtSetDevice'):
                stream_num += 1
            if stream_num > max_stream:
                max_stream = stream_num
            elif contend.count('aclrtDestroyStream') and stream_num > 0:
                stream_num -= 1
            if stream_num >= 1024:
                value = []
                value.append("aclrtCreateStream")
                value.append("The max num of streams is 1024")
                value.append('-')
                extend_result.value.append(value)
        contend = profile_data.readline()
    return extend_result


# 昇腾310 AI处理器媒体数据处理V1->昇腾310P AI处理器媒体数据处理V1迁移指引
def process_profiling_file_with_json(profile_data, extend_result):
    contend = profile_data.readline()
    while contend:
        api = str(contend).split(',')[0]
        if api in V1_transformer.keys():
            value = []
            value.append(api)
            value.append(V1_transformer[api])
            value.append('-')
            extend_result.value.append(value)
        contend = profile_data.readline()
    return extend_result


def datatype_process(file_pathname, extend_result):
    # 遍历该目录下的所有code文件
    dvppmem = []
    for filename in os.listdir(file_pathname):
        if filename.endswith('.cpp') or filename.endswith('.py') or filename.endswith('.h'):
            line_num = 0
            path = os.path.join(file_pathname, filename)
            with open(path, encoding='UTF-8') as f:
                contents = f.readlines()
                synch_need = 0
                ACL_VENC_BUF_SIZE_UINT32_flag = 0
                ACL_VENC_MAX_BITRATE_UINT32_flag = 0
                ACL_VENC_RC_MODE_UINT32_flag = 0
                EnablePeerAccess_num = 1
                CanAccessPeer_num = 1
                memcpy_num = 1
                for line in contents:
                    line_num += 1
                    if line.count('aclrtDeviceCanAccessPeer'):
                        CanAccessPeer_num += 1
                    if line.count('aclrtDeviceEnablePeerAccess'):
                        EnablePeerAccess_num += 1
                    if line.count('aclrtMemcpy'):
                        memcpy_num += 1
                        if memcpy_num > EnablePeerAccess_num and memcpy_num == CanAccessPeer_num:
                            EnablePeerAccess_num += 1
                            value = []
                            value.append("aclrtMemcpy")
                            value.append(
                                "Please use aclrtDeviceEnablePeerAccess to check whether supported memory copy")
                            value.append(filename + ' Line:' + str(line_num))
                            extend_result.value.append(value)
                        elif memcpy_num == EnablePeerAccess_num and memcpy_num > CanAccessPeer_num:
                            CanAccessPeer_num += 1
                            value = []
                            value.append("aclrtMemcpy")
                            value.append(
                                "Please use aclrtDeviceCanAccessPeer to check whether supported memory copy")
                            value.append(filename + ' Line:' + str(line_num))
                            extend_result.value.append(value)
                        elif memcpy_num > EnablePeerAccess_num and memcpy_num > CanAccessPeer_num:
                            EnablePeerAccess_num += 1
                            CanAccessPeer_num += 1
                            value = []
                            value.append("aclrtMemcpy")
                            value.append(
                                "Please use aclrtDeviceCanAccessPeer and aclrtDeviceEnablePeerAccess to check whether supported memory copy")
                            value.append(filename + ' Line:' + str(line_num))
                            extend_result.value.append(value)


                    if synch_need == 1 and line.count('aclrtSynchronizeStream') == 0:
                        synch_need = 0
                        value = []
                        value.append("aclmdlExecuteAsync")
                        value.append('Please use aclrtSynchronizeStream to block the Host run')
                        value.append(filename + ' Line:' + str(line_num))
                        extend_result.value.append(value)
                        memcpy_num += 1
                    if synch_need == 1 and line.count('aclrtSynchronizeStream'):
                        synch_need = 0
                    if line.count('aclmdlExecuteAsync'):
                        synch_need = 1


                    # 0 copy
                    if line.count('acldvppMalloc'):
                        para = line.split('(')[1]
                        para = para.split(',')[0]
                        para = para.split(')')[0]
                        dvppmem.append(para)
                    if line.count('aclrtMemcpy') and dvppmem != []:
                        para = line.split('aclrtMemcpy')[1]
                        para = para.split(',')[2]
                        print(para)
                        print(dvppmem)
                        if para in dvppmem:
                            value = []
                            value.append('aclrtMemcpy')
                            value.append(
                                'There is not need to copy the data on the dvpp output memory to the non dvpp device memory')
                            value.append(filename + ' Line:' + str(line_num))
                            extend_result.value.append(value)

                    #V1 transform
                    if line.count('acldvppSetResizeConfigInterpolation'):
                        interpolation = line.split('(')[1]
                        interpolation = interpolation.split(')')[0]
                        interpolation= interpolation.split(',')[1]
                        if int(interpolation)==0:
                            value = []
                            value.append('acldvppSetResizeConfigInterpolation')
                            value.append('Ascend 710 0:(default)Bilinear algorithm.1:Bilinear algorithm')
                            value.append(filename + ' Line:' + str(line_num))
                            extend_result.value.append(value)
                        if int(interpolation)==3 or int(interpolation)==4 :
                            value = []
                            value.append('acldvppSetResizeConfigInterpolation')
                            value.append('Ascend 710 only support 0:(default)Bilinear algorithm. 1:Bilinear algorithm  2:Nearest neighbor algorithm')
                            value.append(filename + ' Line:' + str(line_num))
                            extend_result.value.append(value)
                    if line.count('aclvencSetChannelDescParam'):
                        value = []
                        value.append('aclvencSetChannelDescParam')
                        value.append(
                            'Ascend 710:cannot use the set IP ratio function,needs to set the output buffer, and there is no need to copy the encoded output results again. Set the output code rate to 300, otherwise the default value of 2000 will be used on ascend 710.')
                        value.append(filename + ' Line:' + str(line_num))
                        extend_result.value.append(value)

                    for datatypeparam in task_data.keys():
                        if datatypeparam in line:
                            if datatypeparam == "ACL_VENC_BUF_SIZE_UINT32":
                                ACL_VENC_BUF_SIZE_UINT32_flag = 1
                                ACL_VENC_BUF_SIZE_UINT32_para = line.split('=')[1]
                                ACL_VENC_BUF_SIZE_UINT32_para = int(ACL_VENC_BUF_SIZE_UINT32_para.split(',')[0])
                                if ACL_VENC_BUF_SIZE_UINT32_para < 5:
                                    value = []
                                    value.append(datatypeparam)
                                    value.append(task_data[datatypeparam])
                                    value.append(filename + ' Line:' + str(line_num))
                                    extend_result.value.append(value)

                            elif datatypeparam == "ACL_VENC_MAX_BITRATE_UINT32":
                                ACL_VENC_MAX_BITRATE_UINT32_flag = 1
                                ACL_VENC_MAX_BITRATE_UINT32_para = line.split('=')[1]
                                ACL_VENC_MAX_BITRATE_UINT32_para = int(ACL_VENC_MAX_BITRATE_UINT32_para.split(',')[0])
                                if ACL_VENC_MAX_BITRATE_UINT32_para > 614400 or ACL_VENC_MAX_BITRATE_UINT32_para < 2:
                                    if ACL_VENC_MAX_BITRATE_UINT32_para == 0:
                                        value = []
                                        value.append(datatypeparam)
                                        value.append("The parameter value defaults to 2000")
                                        value.append(filename + ' Line:' + str(line_num))
                                        extend_result.value.append(value)
                                    else:
                                        value = []
                                        value.append(datatypeparam)
                                        value.append(task_data[datatypeparam])
                                        value.append(filename + ' Line:' + str(line_num))
                                        extend_result.value.append(value)

                            elif datatypeparam == "ACL_VENC_SRC_RATE_UINT32":

                                ACL_VENC_SRC_RATE_UINT32_para = line.split('=')[1]
                                ACL_VENC_SRC_RATE_UINT32_para = int(ACL_VENC_SRC_RATE_UINT32_para.split(',')[0])
                                if ACL_VENC_SRC_RATE_UINT32_para > 240 or ACL_VENC_SRC_RATE_UINT32_para < 1 or (
                                        ACL_VENC_SRC_RATE_UINT32_para != 0):
                                    value = []
                                    value.append(datatypeparam)
                                    value.append(task_data[datatypeparam])
                                    value.append(filename + ' Line:' + str(line_num))
                                    extend_result.value.append(value)


                            elif datatypeparam == "ACL_VENC_RC_MODE_UINT32":
                                ACL_VENC_RC_MODE_UINT32_flag = 1
                                ACL_VENC_RC_MODE_UINT32_para = line.split('=')[1]
                                ACL_VENC_RC_MODE_UINT32_para = ACL_VENC_RC_MODE_UINT32_para.split(',')[0]
                                if ACL_VENC_RC_MODE_UINT32_para == 0:
                                    value = []
                                    value.append(datatypeparam)
                                    value.append("The default value of 0 indicates VBR mode")
                                    value.append(filename + ' Line:' + str(line_num))
                                    extend_result.value.append(value)

                            elif datatypeparam == "ACL_VENC_RC_MODE_UINT32":
                                value.append(datatypeparam)
                                value.append(task_data[datatypeparam])
                                value.append(filename + ' Line:' + str(line_num))
                                extend_result.value.append(value)
                            if ACL_VENC_BUF_SIZE_UINT32_flag == 0:
                                value = []
                                value.append("ACL_VENC_BUF_SIZE_UINT32")
                                value.append("The parameter value defaults to 8m")
                                value.append(filename + ' Line:' + str(line_num))
                                extend_result.value.append(value)
                            if ACL_VENC_MAX_BITRATE_UINT32_flag == 0:
                                value = []
                                value.append("ACL_VENC_MAX_BITRATE_UINT32")
                                value.append("The default value of 0 indicates VBR mode")
                                value.append(filename + ' Line:' + str(line_num))
                                extend_result.value.append(value)
                            if ACL_VENC_RC_MODE_UINT32_flag == 0:
                                value = []
                                value.append("ACL_VENC_RC_MODE_UINT32")
                                value.append("The parameter value defaults to 2000")
                                value.append(filename + ' Line:' + str(line_num))
                                extend_result.value.append(value)

    return extend_result


def result_parse(result, extend_result):
    if not extend_result.value:
        result.class_type = class_type['op']
        result.error_code = error_code['optimized']
        result.summary = "710 API operations are well optimized"
        return result.generate()
    result.class_type = class_type['op']
    result.error_code = error_code['success']
    result.summary = "710 API operations need to be optimized"
    result.extend_result.append(extend_result)
    return result.generate()