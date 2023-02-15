#  训练调优知识库
## 介绍
专家系统训练调优工具，提供自动训练瓶颈分析调优能力，大幅提升训练调优效率。

目前包含知识库：
> -  训练通信算子优化知识库

## 环境准备
```bash
# 首先需要安装CANN包 >= 6.0
./Ascend-cann-toolkit*.run --install  # 安装CANN包
./Ascend-cann-toolkit*.run --upgrade  # 更新CANN包
# 配置环境变量
source {CANN_INSTALL_PATH}/Ascend/ascend-toolkit/set_env.sh

# 下载知识库&安装依赖
git clone https://gitee.com/ascend/msadvisor.git
cd msadvisor/training
pip install -r requirements.txt
```

> 添加python软链接：LD_LIBRARY_PATH添加当前使用的python的lib目录路径
> `export LD_LIBRARY_PATH=/usr/local/python*.*/lib:$LD_LIBRARY_PATH`


## 训练通信算子调优知识库

训练通信算子调优知识库需要从OBS上收集数据，需要安装 [ModelArts SDK](https://support.huaweicloud.com/sdkreference-modelarts/modelarts_04_0004.html)


**1、配置obs_access_config**
```bash
vim obs_access_config.json  
```

`obs_access_config.json`结构如下：
```
{
  "access_config": {           // 访问obs的配置参数；如果是计算中心，相关参数请联系运维同事获取
      "access_key": "",        // 登录obs所需的ak、sk信息
       "secret_access_key": "", 
       "server": "",            // 连接obs的服务地址；
       "region_name": "",       // 区域ID，获取方式参考链接：https://support.huaweicloud.com/api-iam/iam_17_0002.html
       "project_id": "",        // 项目ID 
       
       // 如下配置针对计算中心等专有云 通用云不需要设置 设置为空 请咨询相关维护同事
       // 设置该信息后 需要设置相关的域名解析地址
       "iam_endpoint": "",
       "obs_endpoint": "",
       "modelarts_endpoint": ""
}
```
> 域名解析，请咨询ModelArts所在云环境的运维，获取该云相关服务（obs、modelarts、swr）域名和IP的映射关系并写入/etc/hosts


```
比如武汉云相关服务obs、modelarts、swr域名映射关系如下：
58.48.42.196 obs.cn-central-221.ovaijisuan.com
58.48.42.193 modelarts.cn-central-221.ovaijisuan.com
58.48.42.198 swr.cn-central-221.ovaijisuan.com
```

**2、命令行使用**
- 运行知识库前，需要下载profiling数据
	```bash
	bash msadvisor.sh --rank_size=24 --bucket_name="obs://path/to/profiler"
	```
	- rank_size: 集群训练用到的卡数
	- bucket_name: obs上存放训练产生的profiling数据目录路径
	> profiling数据存放在和`msadvisor`同层的`profiler`目录下。如果已存在`profiler`目录，会删除该目录，避免数据污染。**请注意数据备份！**
  
	> 分析结果保存在`profiler/recommendation/visualization`

- 使用现有数据
	```bash
	bash msadvisor.sh --rank_size=24 --data="path/to/profiler"
	```
	- rank_size: 同上
	- data: 本地环境存放profiling数据目录路径
	
	> 知识库分析结果保存在`${data}/recommendation/visualization`

- 查看分析结果
    
	1、通信算子的分析结果概览通过打屏展示

	2、详细信息，请下载`visualization`目录到本地，通过该目录下的`hccl_analysis_result.html`查看通信算子的详细分析结果

**通信算子优化知识库使用须知**
- profiling保存路径格式统一
  > 请在训练脚本中将profiling数据传输到OBS上的路径格式设置为`profiler_path/{rank_id}`，和知识库解析obs数据的路径格式统一

- hccl_parser组件解析存在上限
	> hccl_parser组件目前解析数据上限为500，可能造成部分通信算子的信息遗漏，导致知识库分析失败。需要在modelarts训练时， 重新安装调整了解析上限的hccl_parser组件

	重新安装：
	
	1、修改hccl_parser组件的解析上限，将修改后的hccl_parser-*.whl上传到训练代码目录。修改细节可以参考下方FAQ
	
	2、在ma-pre-start.sh中添加以下下代码
	```bash
	sudo pip install  ${LOCAL_DIR}/${WORK_DIR}/hccl_parser-*.whl --force-reinstall
	export PYTHONPATH=/usr/local/ma/python3.7/lib/python3.7/site-packages:$PYTHONPATH
	```



## 训练性能数据精简

> Modelarts大模型训练时，`mindspore.profiler`会产生大量训练数据，一般使用`mox`模块拷贝传输到`OBS`中，但巨大的数据量是一个挑战；
> 故提供一个可以在传输数据过程中过滤掉冗余数据的脚本，达到数据精简的目的，减小训练容器=>OBS的数据传输压力.

数据精简脚本位于`training/utils/data_slim.py`，数据精简主要针对**原始数据**和**profiler解析数据**两部分：
- 原始数据`PROF_*`，过滤数据如下 
  - mindspore生成的hwst.data.join 
  - HCCL.* 数据
- `profiler`解析数据，只保留了知识库分析必要数据，如下：
  - ascend_timeline_display_{rank_id}.json
  - step_trace_raw_{rank_id}_detail_time
  - hccl_info_{rank_id}

> 通过此种方式，传输数据量可以减少 **50%** 左右

**使用方式：**

在训练脚本中，使用`collect_data()`, 直接替换`mox.file.copy_parallel()`
```python
from data_slim import collect_data

# mox.file.copy_parallel(srt_url, dst_url)
collect_data(srt_url, dst_url, data='HCCL')
```
collect_data()参数描述
- `srt_url` : 训练容器内保存profiling数据路径
- `dst_url` : OBS的数据保存路径, 路径格式建议：`obs_path/{rank_id}`, 如下方使用示例所示
- `data` : 数据传输策略，可选ALL、HCCL
  - `ALL` :  `collect_data(srt_url, dst_url, data='ALL')` 等价于 `mox.file.copy_parallel(srt_url, dst_url)`
  - `HCCL` : 针对profiling数据采取上述精简策略

**使用示例：**
```python
import os
import mox

from mindspore.profiler import Profiler
from data_slim import collect_data


profiler = Profiler(profile_communication=True, output_path="/cache/profiler_data" + str(rank_id))
model.train()
profiler.analyse()

profiler_path = "obs_path/profiler"
mox.file.make_dirs(profiler_path)

collect_data(
	src_url="/cache/profiler_data" + str(rank_id),
	dst_url=os.path.join(profiler_path, str(rank_id)),
	data="HCCL"
)
	
```

## FAQ

### 如何修改hccl_parser解析上限

1、从安装的CANN包中取出hccl_parser-*.whl，路径为`/usr/local/Ascend/ascend-toolkit/latest/tools/hccl_parser-0.1-py3-none-any.whl`

2、本地pip install hccl_parser-*.whl

3、找到hccl_paser源码，修改hccl_parser.entry中的hccl_parse_op()函数，其中参数num，表示解析上限值，修改该值为期望值即可

4、在hccl_parser同级目录新建setup.py，代码如下：
	
```python
from setuptools import setup

setup(
	name='hccl_parser',
	version='0.1',
	description='hccl parser tool',
	packages=['hccl_parser'],
	python_requires='>=3'
)
```

5、通过命令`python setup.py bdist_wheel` 重新构建hccl_paser-*.whl，
	


