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
import sys
import json
import time
import argparse

from obs import ObsClient
from modelarts.session import Session
from log import AD_INFO, AD_ERROR, ad_log, ad_print_and_log

try:
    import moxing as mox
    moxing_import_flag = True
except Exception:
    moxing_import_flag = False

# data collector identifier
DATA_COLLECT_OK = 0
DATA_COLLECT_ERROR = 1

OBS_ACCESS_CONFIG = [
    "access_key",
    "secret_access_key",
    "server",
    "region_name",
    "project_id",
    "iam_endpoint",
    "obs_endpoint",
    "modelarts_endpoint"
]

class ModelartsHandler:

    def __init__(self):
        self.session = None
        self.obsClient = None

    def create_obs_handler(self, access_config):
        if not moxing_import_flag:
            # Create OBS login handle
            self.obsClient = ObsClient(
                access_key_id=access_config.get("access_key"),
                secret_access_key=access_config.get("secret_access_key"),
                server=access_config.get("server")
            )

    def create_session(self, access_config):
        # 如下配置针对计算中心等专有云 通用云不需要设置
        if access_config.get("iam_endpoint") and access_config.get("obs_endpoint") \
                and access_config.get("modelarts_endpoint"):
            Session.set_endpoint(iam_endpoint=access_config.get("iam_endpoint"),
                                 obs_endpoint=access_config.get("obs_endpoint"),
                                 modelarts_endpoint=access_config.get("modelarts_endpoint"),
                                 region_name=access_config.get("region_name"))
        # Create modelarts session
        self.session = Session(access_key=access_config.get("access_key"),
                               secret_key=access_config.get("secret_access_key"),
                               project_id=access_config.get("project_id"),
                               region_name=access_config.get("region_name"))

        ad_log(AD_INFO, "Create modelarts session succeed")

    def obs_data_collected(self, rank_size, bucket_name, datapath, access_config):
        ad_print_and_log(AD_INFO, "Profiling data Downloading...")
        download_start = time.time()
        # create modelarts session for downloading
        try:
            self.create_session(access_config)
        except Exception as e:
            ad_print_and_log(AD_ERROR, f"Create modelarts session failed, error:{e}")
            return DATA_COLLECT_ERROR

        # download profiling data
        for rank_id in range(rank_size):
            hccl_info_dir = os.path.join(bucket_name, f"hccl_info_{rank_id}")
            step_trace = os.path.join(bucket_name, f"step_trace_raw_{rank_id}_detail_time.csv")
            ascend_timeline = os.path.join(bucket_name, f"ascend_timeline_display_{rank_id}.json")
            try:
                local_dir = f"{datapath}/"
                self.session.obs.download_dir(src_obs_dir=hccl_info_dir, dst_local_dir=local_dir)
                self.session.obs.download_file(src_obs_file=ascend_timeline, dst_local_dir=local_dir)
                self.session.obs.download_file(src_obs_file=step_trace, dst_local_dir=local_dir)
            except Exception as e:
                ad_print_and_log(AD_ERROR, f"rank:{rank_id} data collected failed, error:{e}")
                return DATA_COLLECT_ERROR
        cost_time = time.time() - download_start
        ad_print_and_log(AD_INFO, f"File download succeeded !, cost time: {cost_time}")
        return DATA_COLLECT_OK


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parameters for collecting profiling data from obs")
    parser.add_argument("--rank_size", type=int, default=1)
    parser.add_argument("--bucket_name", type=str, default="obs://")
    parser.add_argument("--download_path", type=str, default=None)
    args = parser.parse_args()

    # load obs access_configs
    access_configs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    access_configs_path = os.path.join(access_configs_path, "obs_access_config.json")
    with open(access_configs_path) as f:
        config = json.load(f)
    access_configs = config["access_config"]
    
    modelarts_handler = ModelartsHandler()
    identifier = modelarts_handler.obs_data_collected(args.rank_size, args.bucket_name, args.download_path, access_configs)
    sys.exit(identifier)

