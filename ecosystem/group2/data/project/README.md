EN|[CN](README.zh.md)
# CompileDemo

## Introduction

This sample demonstrates `CompileDemo` program

Process Framework:
```bash
Create context and stream -> memory copy -> Destory context and stream
```

## Supported Products

Atlas 800 (Model 3000), Atlas 800 (Model 3010), Atlas 300 (Model 3010), Atlas 300I (Model 6000)

## Supported ACL Version

1.73.5.1.B050, 1.73.5.2.B050, 1.75.T11.0.B116, 20.1.0, 20.2.0

Run the following command to check the version in the environment where the Atlas product is installed:
```bash
npu-smi info
```

## Dependency

Code dependency:

Each sample in the version package depends on the ascendbase directory.

If the whole package is not copied, ensure that the ascendbase and CompileDemo directories are copied to the same directory in the compilation environment. Otherwise, the compilation will fail. If the whole package is copied, ignore it.

Set the environment variable:
*  `ASCEND_HOME`      Ascend installation path, which is generally `/usr/local/Ascend`
*  `LD_LIBRARY_PATH`  Specifies the dynamic library search path on which the sample program depends

```bash
export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=$ASCEND_HOME/ascend-toolkit/latest/acllib/lib64:$LD_LIBRARY_PATH
```

## Configuration

configure the device_id in `data/config/setup.config`

set device id
```bash
#chip config
device_id = 0 #use the device to run the program
```

## Compilation
```bash
bash build.sh
```

## Execution
```bash
cd dist
./main
```

## Result

Print the result of each step
