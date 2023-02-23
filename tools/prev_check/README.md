# 训前检查工具

## 1 功能介绍

主要针对pytorch训练场景，对pytorch版本和cann版本配套关系和环境变量进行检查，并给出检查结果和对应的版本更新建议。



## 2 使用介绍

（1）下载工具

执行指令git clone https://gitee.com/ascend/msadvisor.git

下载到本地后，进入到目录msadvisor/tools/prev_check

（2）运行工具进行检查

执行指令python3 main.py

（3）执行结果

如果‘check succeed’，则说明版本检查成功；

如果出现检查失败，会提示失败原因；

