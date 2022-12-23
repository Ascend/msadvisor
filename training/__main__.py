# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import argparse


from training.hccl_analysis_model.hcclanalysis import evaluate


def run_hcclanalysis(rank_size, bucket_name, data_path):
    pra = {
        "step_num": 1,
        "download": 1,
        "access_config": {
            # 登录需要的ak sk信息
            'access_key': '',
            'secret_access_key': '',
            # 连接OBS的服务地址。可包含协议类型、域名、端口号。（出于安全性考虑，建议使用https协议）
            # 如果是计算中心，需要联系运维同事获取
            'server': 'obs.cn-south-222.ai.pcl.cn',
            # project_id/region_name:
            # 项目ID/区域ID，获取方式参考链接
            # https://support.huaweicloud.com/api-iam/iam_17_0002.html
            # 如果是计算中心,请咨询相关维护同事
            'region_name': 'cn-south-222_chenj',
            'project_id': 'c5bd8b7d9a8b4b94a2a82eab402c27c8',

            # 如下配置针对计算中心等专有云 通用云不需要设置 设置为空 请咨询相关维护同事
            # 设置该信息后 需要设置相关的域名解析地址
            'iam_endpoint': 'https://iam-pub.cn-south-222.ai.pcl.cn',
            'obs_endpoint': 'https://obs.ai.pcl.cn',
            'modelarts_endpoint': 'https://modelarts.cn-south-222.ai.pcl.cn',
        },
    }
    pra.update({"rank_size": rank_size})
    pra.update({"bucket_name": bucket_name})
    pra = json.dumps(pra)
    result = evaluate(data_path, parameter=pra)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parameters for collecting profiling data from obs")
    parser.add_argument("--rank_size", type=int, default=None)
    parser.add_argument("--bucket_name", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    args = parser.parse_args()

    run_hcclanalysis(args.rank_size, args.bucket_name, args.data_path)
