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
# 设置专家系统执行时间限制为1800s
bash common.sh
```

> 专家系统工具有限制执行时间`<20s`，但是训练调优知识库训练执行时间都是分钟级的，所以需要更改一下执行时间限制， 修改`max_run_time=1800`
> 运行shell脚本`common.sh`自动设置

> 添加python软链接：LD_LIBRARY_PATH添加当前使用的python的lib目录路径
> `export LD_LIBRARY_PATH=/usr/local/python*.*/lib:$LD_LIBRARY_PATH`



## 训练通信算子调优知识库

训练通信算子调优知识库需要从OBS上收集数据，需要安装 [ModelArts SDK](https://support.huaweicloud.com/sdkreference-modelarts/modelarts_04_0004.html)


**1、配置access_config**
```bash
cd  hccl_analysis_model/
vim hcclanalysis.json  # 修改其中access_config、bucket_name、rank_size等参数
```

`hcclanalysis.json`结构如下：
```
{
  "model_list": [
    {
      "model_name": "hcclanalysis",
      "session_list": [
        {
          "python_model_path": "./",
          "parameter":{
              "step_num": null,            // 具体分析的step
              "rank_size": 1,              // 集群训练用到的卡数
              "bucket_name": "obs://**/",  // obs上存放profiling数据的路径
              "download": 1,               // 是否开启profilig数据收集功能{0：关闭， 1：开启}     
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
          }
        }
      ]
    }
  ]
}
```
> 域名解析
> 请咨询ModelArts所在云环境的运维，获取该云相关服务（obs、modelarts、swr）域名和IP的映射关系并写入/etc/hosts


```
比如武汉云相关服务obs、modelarts、swr域名映射关系如下：
58.48.42.196 obs.cn-central-221.ovaijisuan.com
58.48.42.193 modelarts.cn-central-221.ovaijisuan.com
58.48.42.198 swr.cn-central-221.ovaijisuan.com
```

**2、命令行使用**
```bash
msadvisor -c path/hcclanalysis.json -d datapath
```

- 打开数据下载功能时(download设置为1)，`-d datapath` 指定下载目录，profiling数据保存在`datapath/profiler`，可视化结果保存在`datapath/profiler/recommendation/visualization`
- 关闭数据下载功能时(download设置为0)，`-d datapath` 指定保存profiling数据的目录,可视化结果保存在`datapath/recommendation/visualization`
> 请下载`visualization`目录到本地，通过该目录下的`hccl_analysis_result.html`查看知识库详细分析结果

可以通过-p参数快捷设置部分参数，使用方法：`-p "model_name.key_1=value_1;model_name.key_2=value_2"`，示例：
```bash
msadvisor -c hcclanalysis.json -d datapath -p "hcclanalysis.download=1"     # 运行知识库，关闭数据下载
```

