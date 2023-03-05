# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

knowledges_api_change = {
    "aclopCreateKernel(": "检查aclopCreateKernel接口是否用到了aclEngineType枚举，如果使用请修改代码并重新编译",
    "aclmdlGetOutputNameByIndex(": "代码使用了aclmdlGetOutputNameByIndex接口，请重新进行atc模型转换，"
                                  "如果模型包含top名称则还需要适配返回值并重新编译",
    "aclrtFreeHost(": "代码使用了aclrtFreeHost，但未使用aclrtCreateContext或aclrtSetCurrentContext接口显示设置context，"
                     "请增加代码适配，并重新编译",
    "aclgrphBuildModel(": "检查aclgrphBuildModel接口是否有传入PRECISION_MODE、EXEC_DISABLE_REUSED_MEMORY、"
                         "AUTO_TUNE_MODE三个参数，若没有，则无需处理；则检查下init中是否已配置，若未配置，"
                         "建议在build接口的option中进行删除，逻辑跟原先一致，若已配置，需要用户根据需要进行配置修改，若全局仅生效一次，"
                         "则只在init中配置即可，若每次build希望采用不同的option，则每次配置即可",
    "aclgrphBuildModel(": "检查aclgrphBuildModel接口INPUT_SHAPE参数设置，若涉及检查INPUT_SHAPE的传入值是否符合预期，如若不符请修改",
    "aclgrphBuildModel(": "检查aclgrphBuildModel接口INPUT_SHAPE参数设置，若涉及检查INPUT_SHAPE的传入值是否符合预期，如若不符请修改",
    "aclcreateEvent(": "检查aclcreateEvent接口创建的event资源是否超过了1023，若超过了会创建失败",
    "aclmdlGetOutputNameByIndex(": "代码使用了aclmdlGetOutputNameByIndex接口，需要重新进行atc模型转换，如果模型包含top名称还需要适配"
                                  "返回值并重新编译",
    "aclgrphParseCaffe(": "请检查参数是否有配置INPUT_FORMAT、INPUT_SHARP、INPUT_DYNAMIC_INPUT、OP_NAME_MAP、OUTPUT_TYPE、"
                         "LOG_LEVEL，如果有则需要删除这些参数",
    "aclgrphParseTensorFlow(": "请检查参数是否有配置INPUT_FORMAT、INPUT_SHARP、INPUT_DYNAMIC_INPUT、OP_NAME_MAP、OUTPUT_TYPE、"
                              "LOG_LEVEL，如果有则需要删除这些参数",
    "aclgrphParseONNX(": "请检查参数是否有配置INPUT_FORMAT、INPUT_SHARP、INPUT_DYNAMIC_INPUT、OP_NAME_MAP、OUTPUT_TYPE、"
                        "LOG_LEVEL，如果有则需要删除这些参数",
    "aclgrphParseONNXFromMem(": "请检查参数是否有配置INPUT_FORMAT、INPUT_SHARP、INPUT_DYNAMIC_INPUT、OP_NAME_MAP、OUTPUT_TYPE、"
                               "LOG_LEVEL，如果有则需要删除这些参数",
    "CreateDvppApi(": "如果使用了该接口，并且原先调用失败返回值使用-1判断，因为增加细化了返回值，需要修改代码适配",
    "DvppCtl(": "如果使用了该接口，并且原先调用失败返回值使用-1判断，因为增加细化了返回值，需要修改代码适配",
    "DestroyDvppApi(": "如果使用了该接口，并且原先调用失败返回值使用-1判断，因为增加细化了返回值，需要修改代码适配",
    "DvppGetOutParameter(": "如果使用了该接口，并且原先调用失败返回值使用-1判断，因为增加细化了返回值，需要修改代码适配",
    "CreateVdecApi(": "如果使用了该接口，并且原先调用失败返回值使用-1判断，因为增加细化了返回值，需要修改代码适配",
    "VdecCtl(": "如果使用了该接口，并且原先调用失败返回值使用-1判断，因为增加细化了返回值，需要修改代码适配",
    "DestroyVdecApi(": "如果使用了该接口，并且原先调用失败返回值使用-1判断，因为增加细化了返回值，需要修改代码适配",
    "CreateVenc(": "如果使用了该接口，并且原先调用失败返回值使用-1判断，因为增加细化了返回值，需要修改代码适配",
    "SetVencParam(": "如果使用了该接口，并且原先调用失败返回值使用-1判断，因为增加细化了返回值，需要修改代码适配",
    "RunVenc(": "如果使用了该接口，并且原先调用失败返回值使用-1判断，因为增加细化了返回值，需要修改代码适配",
    "DestroyVenc(": "如果使用了该接口，并且原先调用失败返回值使用-1判断，因为增加细化了返回值，需要修改代码适配",
    "AclfvRepoAdd(": "如果使用了该接口，入参删除aclrtStream，新版本不需要传递stream入参",
    "AclfvRepoDel(": "如果使用了该接口，入参删除aclrtStream，新版本不需要传递stream入参",
    "AclfvDel(": "如果使用了该接口，入参删除aclrtStream，新版本不需要传递stream入参",
    "AclfvModify(": "如果使用了该接口，入参删除aclrtStream，新版本不需要传递stream入参",
    "AclfvSearch(": "如果使用了该接口，入参删除aclrtStream，新版本不需要传递stream入参",
    "acl.util.numpy_to_ptr(": "如果使用了该接口，若不想修改代码，则需要升级运行环境为python>=3.8及numpy>=1.22.0，"
                             "若要修改代码，则可用acl.util.bytes_to_ptr替代",
    "acl.util.numpy_contiguous_to_ptr(": "如果使用了该接口，若不想修改代码，则需要升级运行环境为python>=3.8及numpy>=1.22.0，"
                                        "若要修改代码，则可用acl.util.bytes_to_ptr替代",
    "acl.util.ptr_to_numpy(": "如果使用了该接口，若不想修改代码，则需要升级运行环境为python>=3.8及numpy>=1.22.0，若要修改代码，"
                             "则可用acl.util.ptr_to_bytes替代",
    "acl.util.set_attr_list_int(": "如果使用了该接口，若不想修改代码，则需要升级运行环境为python>=3.8及numpy>=1.22.0，若要修改代码，"
                                  "接口的numpy输入可以替换成list输入",
    "acl.util.set_attr_list_bool(": "如果使用了该接口，若不想修改代码，则需要升级运行环境为python>=3.8及numpy>=1.22.0，若要修改代码，"
                                   "接口的numpy输入可以替换成list输入",
    "acl.util.set_attr_list_float(": "如果使用了该接口，若不想修改代码，则需要升级运行环境为python>=3.8及numpy>=1.22.0，若要修改代码，"
                                    "接口的numpy输入可以替换成list输入",
    "acl.util.set_attr_list_list_int(": "如果使用了该接口，若不想修改代码，则需要升级运行环境为python>=3.8及numpy>=1.22.0，若要修改代码，"
                                        "接口的numpy输入可以替换成list输入",
}

knowledges_zero_memory_copy = {
    "aclrtMallocHost(": "使用了aclrtMallocHost，在RC模型下需要替换成aclrtMalloc",
    "aclrtMemcpyAsync(": "使用了aclrtMemcpyAsync接口，在RC模型下kind输入需要替换成ACL_MEMCPY_DEVICE_TO_DEVICE宏",
    "aclrtMemcpy(": "使用了aclrtMemcpy接口，在RC模型下kind输入需要替换成ACL_MEMCPY_DEVICE_TO_DEVICE宏",
    "aclrtFreeHost(": "使用了aclrtFreeHost，在RC模型下需要替换成aclrtFree",
}
