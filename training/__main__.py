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
        "step_num": None,
    }
    pra.update({"rank_size": rank_size})
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
