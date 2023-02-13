#import knowledges
import os

knowledges = {
    "aclopCreateKernel": "检查aclopCreateKernel接口是否用到了aclEngineType枚举，如果使用请修改代码并重新编译",
    "aclmdlGetOutputNameByIndex": "代码使用了aclmdlGetOutputNameByIndex接口，请重新进行atc模型转换，如果模型包含top名称则还需要适配返回值并重新编译",
    "aclrtFreeHost": "代码使用了aclrtFreeHost，但未使用aclrtCreateContext或aclrtSetCurrentContext接口显示设置context，请增加代码适配，并重新编译",
    "aclgrphBuildModel": "",
    "aclcreateEvent": "检查aclcreateEvent接口创建的event资源是否超过了1023，若超过了会创建失败",
    "aclmdlGetOutputNameByIndex": "代码使用了aclmdlGetOutputNameByIndex接口，需要重新进行atc模型转换，如果模型包含top名称还需要适配返回值并重新编译",
    "aclgrphParseCaffe": "",
    "aclgrphParseTensorFlow": "",
    "aclgrphParseONNX": "",
    "aclgrphParseONNXFromMem": "",
}


def data_process(file_pathname, extend_result):
    # 遍历该目录下的所有code文件
    if not os.path.isdir(file_pathname):
        return extend_result

    for root, dirs, files in os.walk(file_pathname):
        for file in files:
            if file.endswith('.cpp') or file.endswith('.py') or file.endswith('.h'):
                line_num = 0
                path = os.path.join(root, file)
                with open(path, encoding='UTF-8') as file:
                    for line in file.readlines():
                        line_num += 1
                        for k, r in knowledges.item(): #遍历知识库
                            if k in line:
                                value = []
                                value.append(k)
                                value.append(r)
                                value.append(str(file.name) + ' Line:' + str(line_num))
                                extend_result.value.append(value)

    return extend_result