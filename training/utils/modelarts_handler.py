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

from modelarts.session import Session
from obs import ObsClient
from log import AD_INFO, ad_log

try:
    import moxing as mox
    moxing_import_flag = True
except Exception:
    moxing_import_flag = False


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
        # Create modelars handle
        self.session = Session(access_key=access_config.get("access_key"),
                               secret_key=access_config.get("secret_access_key"),
                               project_id=access_config.get("project_id"),
                               region_name=access_config.get("region_name"))

        ad_log(AD_INFO, "Create modelarts session succeed")

