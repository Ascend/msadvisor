knowledges = {
    "aclopCreateKernel": "检查aclopCreateKernel接口是否用到了aclEngineType枚举，如果使用请修改代码并重新编译",
    "aclmdlGetOutputNameByIndex": "代码使用了aclmdlGetOutputNameByIndex接口，请重新进行atc模型转换，如果模型包含top名称则还需要适配返回值并重新编译",
    "aclrtFreeHost": "代码使用了aclrtFreeHost，但未使用aclrtCreateContext或aclrtSetCurrentContext接口显示设置context，请增加代码适配，并重新编译"
}
