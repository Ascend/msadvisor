/*
 * Copyright (c) 2020.Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <memory>
#include <chrono>
#include <acl/acl.h>
#include "Log/Log.h"
#include "ErrorCode/ErrorCode.h"
#include "ConfigParser/ConfigParser.h"

using namespace std;

namespace {
// Currently useless, ACL framework reservation
const char *g_aclConfigPath = "./data/config/acl.json";

/*
 * @description: Get the device ID from the configuration parsing
 * @param: configParser specify the configuration parsing
 * @param: deviceId is used to save the device ID
 * @return: aclError which is error code of ACL API
 */
APP_ERROR GetDeviceID(ConfigParser &configParser, int32_t &deviceId)
{
    // Obtain the ID and check its validity
    int temp = 0;
    APP_ERROR ret = configParser.GetIntValue("device_id", temp);
    if (ret != APP_ERR_OK) {
        LogError << "deviceId is not digit, please check, ret = " << ret;
        return APP_ERR_COMM_INVALID_PARAM;
    }
    if (temp < 0) {
        LogError << "deviceId < 0, not valid, please check, ret = " << ret;
        return APP_ERR_COMM_INVALID_PARAM;
    }
    deviceId = temp;
    LogInfo << "The device ID is got successfully, deviceId = " << deviceId;
    return APP_ERR_OK;
}

/*
 * @description: Initialize the Ascend310 chip and create resources
 * @param: deviceID specify the NPU number which will be initialized
 * @param: context specify the custom context which will be create
 * @param: create specify the custom stream which will be create
 * @return: aclError which is error code of ACL API
 */
APP_ERROR InitAscend(int32_t deviceID, aclrtContext &context, aclrtStream &stream)
{
    APP_ERROR ret = aclInit(g_aclConfigPath);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to initialize the ACL, ret = " << ret;
        return ret;
    }
    LogInfo << "The ACL is initialized successfully";

    ret = aclrtSetDevice(deviceID);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to set the device, ret = " << ret;
        return ret;
    }
    LogInfo << "The device is set successfully, deviceID = " << deviceID;

    ret = aclrtCreateContext(&context, deviceID);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to create the context, ret = " << ret;
        return ret;
    }
    LogInfo << "The context is created successfully";

    ret = aclrtCreateStream(&stream);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to create the stream, ret = " << ret;
        return ret;
    }
    LogInfo << "The stream is created successfully";

    aclrtRunMode runMode;
    ret = aclrtGetRunMode(&runMode);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to obtain the running mode, ret = " << ret;
        return ret;
    }
    std::string mode = (runMode == ACL_DEVICE) ? "Device" : "Host";
    LogInfo << "Running mode obtained successfully, mode = " << mode;

    return APP_ERR_OK;
}

/*
 * @description: Destroy resources and reset the Ascend310 chip
 * @param: deviceID specify the NPU number which will be reset
 * @param: context specify the custom context which will be destory
 * @param: stream specify the custom stream which will be destory
 * @return: aclError which is error code of ACL API
 */
APP_ERROR DestroyAscend(int32_t deviceID, aclrtContext &context, aclrtStream &stream)
{
    APP_ERROR ret = APP_ERR_OK;
    if (stream != nullptr) {
        ret = aclrtDestroyStream(stream);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to destroy the stream, ret = " << ret;
            return ret;
        }
        stream = nullptr;
    }
    LogInfo << "The stream is destroyed successfully";

    if (context != nullptr) {
        ret = aclrtDestroyContext(context);
        if (ret != APP_ERR_OK) {
            LogError << "Failed to destroy the context, ret = " << ret;
            return ret;
        }
        context = nullptr;
    }
    LogInfo << "The context is destroyed successfully";

    ret = aclrtResetDevice(deviceID);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to reset the device, ret = " << ret;
        return ret;
    }
    LogInfo << "Device reset successfully, deviceID = " << deviceID;

    ret = aclFinalize();
    if (ret != APP_ERR_OK) {
        LogError << "Failed to deinitialize the ACL, ret = " << ret;
        return ret;
    }
    LogInfo << "The ACL is deinitialized successfully";

    return APP_ERR_OK;
}

/*
 * @description: Transfer data between the host and the device
 */
APP_ERROR DataInteraction()
{
    const std::string str("Hello Davinci");
    // Allocate memory on the device
    char *devPtr = nullptr;
    APP_ERROR ret = aclrtMalloc((void **)&devPtr, str.length() + 1, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to allocate memory on the device, ret = " << ret;
        return ret;
    }
    std::shared_ptr<char> devBuf(devPtr, aclrtFree);

    // Allocate memory on the host
    char *hostPtr = nullptr;
    ret = aclrtMallocHost((void **)&hostPtr, str.length() + 1);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to allocate memory on the host, ret = " << ret;
        return ret;
    }
    std::shared_ptr<char> hostBuf(hostPtr, aclrtFreeHost);

    // Send string to the device from the host
    ret = aclrtMemcpy(devBuf.get(), str.length() + 1, str.c_str(), str.length() + 1, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to copy data from the host to the device, ret = " << ret;
        return ret;
    }
    LogInfo << "Send the \"Hello Davinci\" to device memory successfully";

    // Receive string from the device to the host.
    ret = aclrtMemcpy(hostBuf.get(), str.length() + 1, devBuf.get(), str.length() + 1, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to copy data from the device to the host, ret = " << ret;
        return ret;
    }
    LogInfo << "Receive the \"" << std::string(hostBuf.get()) << "\" from device memory successfully";
    return APP_ERR_OK;
}
}

/*
 * @description: Program entry
 * @param: argc specify the number of argv
 * @param: argv contains the path and command parameters of the program
 * @return: program exit status
 */
int main(int argc, char **argv)
{
    AtlasAscendLog::Log::LogInfoOn();

    // Parse the configuration file
    ConfigParser config;
    APP_ERROR ret = config.ParseConfig("./data/config/setup.config");
    if (ret != APP_ERR_OK) {
        LogError << "Failed to parse the configuration file, ret = " << ret;
        return ret;
    }

    // Get the device ID from the configuration parsing
    int32_t deviceId = 0;
    ret = GetDeviceID(config, deviceId);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to Get the device ID. , ret = " << ret;
        return ret;
    }

    // Initial the Ascend, context, stream
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    ret = InitAscend(deviceId, context, stream);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to initialize the Ascend, ret = " << ret;
        return APP_ERR_COMM_INIT_FAIL;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    ret = DataInteraction();
    if (ret != APP_ERR_OK) {
        LogError << "Failed to perform data interaction, ret = " << ret;
    }

    // Destroy the Ascend, context, stream
    ret = DestroyAscend(deviceId, context, stream);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to destroy the Ascend, ret = " << ret;
        return ret;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double costTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    LogInfo << "TotalRuntime: " << costTime << "ms";

    return APP_ERR_OK;
}
