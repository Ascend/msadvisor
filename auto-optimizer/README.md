# auto-optimizer工具指南

## 介绍

auto-optimizer（自动调优工具）使能ONNX模型在昇腾芯片的优化，并提供基于ONNX的改图功能。

**软件架构**

![architecture](./docs/img/architecture.png)

auto-optimizer主要通过graph_optimizer、knowledge、graph_refactor和inference_component接口提供专家系统工具的自动调优能力。

接口详细介绍请参见如下手册：

- [x]  [knowledge](docs/knowledge_optimizer/knowledge_optimizer_framework.md)
- [x]  [graph_refactor](auto_optimizer/graph_refactor/README.md)
- [x]  [inference_component](auto_optimizer/inference_engine/README.md)



## 工具安装

```shell
git clone https://gitee.com/ascend/msadvisor.git
cd msadvisor/auto-optimizer

# it's recommended to upgrade pip and install wheel

# is's also recommended to use conda/venv/... to manage python enviornments
# for example: `python3 -m venv .venv && source .venv/bin/activate`

# is's also recommended to use `python3 -m pip` to avoid python env issue

python3 -m pip install --upgrade pip
python3 -m pip install wheel

# installation
# optional features: inference, simplify

# without any optional feature
python3 -m pip install .

# with inference feature
python3 -m pip install .[inference]

# with inference and simplify feature
python3 -m pip install .[inference,simplify]
```

- inference：提供推理组件，如果需要使用optimize命令的--infer-test选项进行优化，需要安装该特性。
- simplify：提供onnx.simplify接口。

## 工具使用

### 命令格式说明

auto-optimizer工具可通过auto_optimizer可执行文件方式启动，若安装工具时未提示Python的HATH变量问题，或手动将Python安装可执行文件的目录加入PATH变量，则可以直接使用如下命令格式：

```bash
auto_optimizer <COMMAND> [OPTIONS] [ARGS]
```

或直接使用如下命令格式：

```bash
python3 -m auto_optimizer <COMMAND> [OPTIONS] [ARGS]
```

其中<COMMAND>为auto_optimizer执行模式参数，取值为list、evaluate和optimize；[OPTIONS]和[ARGS]为evaluate和optimize命令的额外参数，详细介绍请参见后续“evaluate命令”和“optimize命令”章节。

### 使用流程

auto-optimizer工具建议按照list、evaluate和optimize的顺序执行。

操作流程如下：

1. 执行**list**命令列举当前支持自动调优的所有知识库。
2. 执行**evaluate**命令搜索可以被指定知识库优化的ONNX模型。
3. 执行**optimize**命令使用指定的知识库来优化指定的ONNX模型。

### list命令

命令示例如下：

```bash
python3 -m auto_optimizer list
```

输出示例如下：

```bash
Available knowledges:
   0 KnowledgeConv1d2Conv2d
   1 KnowledgeMergeConsecutiveSlice
   2 KnowledgeTransposeLargeInputConv
   3 KnowledgeMergeConsecutiveConcat
   4 KnowledgeTypeCast
   5 KnowledgeSplitQKVMatmul
   6 KnowledgeSplitLargeKernelConv
   7 KnowledgeResizeModeToNearest
   8 KnowledgeTopkFix
   9 KnowledgeMergeCasts
  10 KnowledgeEmptySliceFix 
  11 KnowledgeDynamicReshape
  12 KnowledgeGatherToSplit
  13 KnowledgeAvgPoolSplit
  14 KnowledgeBNFolding
```

列举的知识库按照“序号”+“知识库名称”的格式展示，**evaluate**或**optimize**命令通过**knowledges**参数指定知识库时，可指定知识库序号或名称。关于具体知识库的详细信息，请参见[知识库文档](docs/knowledge_optimizer/knowledge_optimizer_rules.md)。

注意：序号是为了方便手动调用存在的，由于知识库可能存在被删除或修改等情况，序号可能会变化。

### evaluate命令

命令格式如下：

```bash
python3 -m auto_optimizer evaluate [OPTIONS] PATH
```

evaluate可简写为eva。

参数说明：

| 参数    | 说明                                                         | 是否必选 |
| ------- | ------------------------------------------------------------ | -------- |
| OPTIONS | 额外参数。可取值：<br/>    -k/--knowledges：知识库列表。可指定知识库名称或序号，以英文逗号“,”分隔。默认启用除修复性质以外的所有知识库。<br/>    -r/--recursive：在PATH为文件夹时是否递归搜索。默认关闭。<br/>    -v/--verbose：打印更多信息，目前只有搜索进度。默认关闭。<br/>    -p/--processes: 使用multiprocess并行搜索，指定进程数量。默认1。<br/>    --help：工具使用帮助信息。 | 否       |
| PATH    | evaluate的搜索目标，可以是.onnx文件或者包含.onnx文件的文件夹。 | 是       |

命令示例及输出如下：

```bash
python3 -m auto_optimizer evaluate aasist_bs1_ori.onnx
```

```
aasist_bs1_ori.onnx    KnowledgeConv1d2Conv2d,KnowledgeMergeConsecutiveSlice,KnowledgeTransposeLargeInputConv,KnowledgeTypeCast,KnowledgeMergeCasts
```

### optimize命令

命令格式如下：

```bash
python3 -m auto_optimizer optimize [OPTIONS] INPUT_MODEL OUTPUT_MODEL
```

optimize可简写为opt。

参数说明：

| 参数         | 说明                                                         | 是否必选 |
| ------------ | ------------------------------------------------------------ | -------- |
| OPTIONS      | 额外参数。可取值：<br/>    -k/--knowledges：知识库列表。可指定知识库名称或序号，以英文逗号“,”分隔。默认启用除修复性质以外的所有知识库。<br/>    -t/--infer-test：当启用这个选项时，通过对比优化前后的推理速度来决定是否使用某知识库进行调优，保证可调优的模型均为正向调优。启用该选项需要安装额外依赖[inference]，并且需要安装CANN。<br/>    -s/--soc：使用的昇腾芯片版本。默认为Ascend310P3。仅当启用infer-test选项时有意义。<br/>    -d/--device：NPU设备ID。默认为0。仅当启用infer-test选项时有意义。<br/>    -l/--loop：测试推理速度时推理次数。仅当启用infer-test选项时有意义。默认为100。<br/>    --threshold：推理速度提升阈值。仅当知识库的优化带来的提升超过这个值时才使用这个知识库，可以为负，负值表示接受负优化。默认为0，即默认只接受推理性能有提升的优化。仅当启用infer-test选项时有意义。<br/>    --input-shape：静态Shape图输入形状，ATC转换参数，可以省略。仅当启用infer-test选项时有意义。<br/>    --input-shape-range：动态Shape图形状范围，ATC转换参数。仅当启用infer-test选项时有意义。<br/>    --dynamic-shape：动态Shape图推理输入形状，推理用参数。仅当启用infer-test选项时有意义。<br/>    --output-size：动态Shape图推理输出实际size，推理用参数。仅当启用infer-test选项时有意义。<br/>    --help：工具使用帮助信息。 | 否       |
| INPUT_MODEL  | 输入ONNX待优化模型，必须为.onnx文件。                        | 是       |
| OUTPUT_MODEL | 输出ONNX模型名称，用户自定义，必须为.onnx文件。优化完成后在当前目录生成优化后ONNX模型文件。 | 是       |

命令示例及输出如下：

```bash
python3 -m auto_optimizer optimize aasist_bs1_ori.onnx aasist_bs1_ori_out.onnx
```

```bash
Optimization success
Applied knowledges:
  KnowledgeConv1d2Conv2d
  KnowledgeMergeConsecutiveSlice
  KnowledgeTransposeLargeInputConv
  KnowledgeTypeCast
  KnowledgeMergeCasts
Path: aasist_bs1_ori.onnx -> aasist_bs1_ori_out.onnx
```

## 许可证

[Apache License 2.0](LICENSE)

## 免责声明

auto-optimizer仅提供基于ONNX的改图及调优参考，不对其质量或维护负责。
如果您遇到了问题，Gitee/Ascend/auto-optimizer提交issue，我们将根据您的issue跟踪解决。
衷心感谢您对我们社区的理解和贡献。
