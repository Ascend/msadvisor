#  迁移调优知识库
## 介绍
专家系统迁移调优工具，提供训练脚本迁移分析调优能力，提升模型迁移后的训练性能。

目前包含知识库：
> -  Pytorch训练脚本优化知识库

## 环境准备
```bash
# 首先需要安装CANN包 >= 6.0
./Ascend-cann-toolkit*.run --install  # 安装CANN包
./Ascend-cann-toolkit*.run --upgrade  # 更新CANN包
# 配置环境变量
source {CANN_INSTALL_PATH}/Ascend/ascend-toolkit/set_env.sh

# 下载知识库&安装依赖
git clone https://gitee.com/ascend/msadvisor.git
cd msadvisor/migrating/
pip install -r requirements.txt
```

> 添加python软链接：LD_LIBRARY_PATH添加当前使用的python的lib目录路径
> `export LD_LIBRARY_PATH=/usr/local/python*.*/lib:$LD_LIBRARY_PATH`

## Pytorch训练脚本优化知识库

**命令行使用**

```bash
msdavisor -c PytorchTrainingOptimizer/pytorchoptimizer.json -d training_dir_path
```

- -c 指定该知识库运行配置文件
- -d 指定训练工程目录路径

> 该知识库会自动分析`-d`指定目录下的所有`.py`文件，给出相应优化建议

