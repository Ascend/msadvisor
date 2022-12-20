# 搬运瓶颈识别

##  **简述**  

功能：专家系统Roofline模型，通过分析模型执行的profiling等数据，得到算子所属的瓶颈问题类型，以及对应的专家系统优化建议。

输入：

profiling，dump等相关性能数据。

输出：

1、 算子搬运瓶颈的具体的原因

2、 对应的优化建议

## 代码目录结构

```bash
root@root:# tree -L 2
.
├── data
│   ├── knowledge
│   ├── log
│   ├── profiling
│   └── project
├── doc
│   ├── 模型开发设计文档.md
│   ├── 模型说明文档.md
│   └── 测试设计文档.md
├── ecosystem.json
├── README.md
├── requirements.txt
└── src
    ├── ge_ir_pb2.py
    ├── get_bottlenck_pathway.py
    ├── get_bottleneck_pipeline.py
    ├── get_data_migration_granularity.py
    ├── get_profiling_data.py
    ├── get_repeated_data_migration.py
    ├── model.py
    ├── parse_om_model.py
    ├── result_parse.py
    └── test
```

其中，data、src/test/data需要用户根据搬运瓶颈识别中的输入数据要求自行提供。

## 运行环境

### 软件依赖

- protobuf==3.20.1

- pytest


## 运行及测试

### 1、单独运行知识库

首先，用户提供data目录的应用数据，然后运行下面命令即可。

```bash
cd src
python3 model.py
```



### 2、测试用例的运行

首先，用户提供src/test/data测试用例应用数据，然后运行下面命令即可。

```bash
cd src/test
python3 -m pytest -s test.py
```



### 3、专家系统测试

首先，用户提供data目录的应用数据，然后，根据具体路径修改并运行下面三行命令即可。

`source /xxx/env.sh`

`cd /xxx/Ascend/ascend-toolkit/latest/tools/msadvisor/bin`

`./msadvisor -d /xxx/msadvisor/ecosystem/memory_bound_optim/data -c /xxx/msadvisor/ecosystem/memory_bound_optim/ecosystem.json`

