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
source {CANN_INSTALL_PATH}/Ascend/ascend-toolkit/set_evn.sh

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
	> **profiling数据存放在和`msadvisor`同级的`profiler`目录下**
  > **知识库分析结果保存在`profiler/recommendation/visualization`**

- 使用现有数据
	```bash
	bash msadvisor.sh --rank_size=24 --data="path/to/profiler"
	```
	- rank_size: 同上
	- data: 本地环境存放profiling数据目录路径
	
	> **知识库分析结果保存在`${data}/recommendation/visualization`**

- 查看分析结果
    
	1、通信算子的分析结果概览通过打屏展示

	2、详细信息，请下载`visualization`目录到本地，通过该目录下的`hccl_analysis_result.html`查看通信算子的详细分析结果
