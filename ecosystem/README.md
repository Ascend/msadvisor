# 310迁移310P调优知识库

## 业务背景

昇腾310处理器硬件迁移至配置昇腾 310P处理器硬件的任务，在满足具体场景功能要求的基本前提下，需要关注整体性能目标的达成情况。
不同应用场景，对性能的要求往往是不同的，常见的推理性能指标包括时延、吞吐两项；建议在没有特殊要求的场景下，以吞吐作为迁移目标。
因此需要进行必要的前置分析，主要包括业务分析、硬件选型、软硬件兼容性评估、推理业务迁移流程等部分。


## 组件使用说明

- [x]  [运行环境调优知识库](#1-运行环境调优知识库)
- [x]  [API使用问题识别及自动调优知识库](#2-API使用问题识别及自动调优知识库)
- [x]  [代码逻辑错误识别及自动调优知识库](#3-代码逻辑错误识别及自动调优知识库)

## 安装教程

```shell
git clone https://gitee.com/ascend/msadvisor.git

```
## 1 运行环境调优知识库
### 知识库调优方向
![知识库调优方向](running_env_tuning/doc/running_env_tuning.PNG)
### 数据准备

### 命令行执行
```shell
msadvisor -c xxx/msadvisor/ecosystem/running_env_tuning/RunningEnvTuning.json -d xxx/msadvisor/ecosystem/running_env_tuning/data/knowledge

```
## 2 API使用问题识别及自动调优知识库
### 知识库调优方向
![输入图片说明](Api_optimization_suggestion/doc/Api_optimization_suggestion.PNG)

## 3 代码逻辑错误识别及自动调优知识库

## 许可证

[Apache License 2.0](LICENSE)

## 免责声明

msadvisor仅提供调优参考，不对其质量或维护负责。
如果您遇到了问题，Gitee/Ascend/msadvisor提交issue，我们将根据您的issue跟踪解决。
衷心感谢您对我们社区的理解和贡献。
