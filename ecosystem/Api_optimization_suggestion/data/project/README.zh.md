中文|[英文](README.md)
# CompileDemo

## 介绍

本开发样例演示 `CompileDemo` 程序

该Sample处理流程为：
```bash
创建 context、stream -> 内存复制 -> 销毁 context、stream
```

## 支持的产品

Atlas 800 (Model 3000), Atlas 800 (Model 3010), Atlas 300 (Model 3010), Atlas 300I (Model 6000)

## 支持的ACL版本

1.73.5.1.B050, 1.73.5.2.B050, 1.75.T11.0.B116, 20.1.0, 20.2.0

查询ACL版本号的方法是，在Atlas产品环境下，运行以下命令：
```bash
npu-smi info
```

## 依赖条件

代码依赖：

版本包中各个Sample都依赖ascendbase目录

编译时如果不是整包拷贝，请确保ascendbase和CompileDemo目录都拷贝到了编译环境的同一路径下，否则会编译失败；如果是整包拷贝，不需要关注。


设置环境变量：
*  `ASCEND_HOME`      Ascend安装的路径，一般为 `/usr/local/Ascend`
*  `LD_LIBRARY_PATH`  指定Sample程序运行时依赖的动态库查找路径

```bash
export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=${ASCEND_HOME}/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH
```

## 配置

需要在 `data/config/setup.config` 配置芯片id

修改芯片id
```bash
#chip config
device_id = 0 #use the device to run the program
```

## 编译
```bash
bash build.sh
```

## 运行
```bash
cd dist
./main
```

## 结果

打印输出每一步的执行结果
