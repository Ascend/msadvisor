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

import json
import os
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from training.utils import log
from hccl_analysis_tool.hccl_analysis_tool_v2 import HcclAnalysisTool
from critical_path_analysis.ascend_timeline_analysis_v2 import run_critical_path_analysis
from utils.constant import Constant
from utils.generate_html import generate_html

INVALID_RESULT = ""


def evaluate(datapath, parameter):
    """
    interface function called by msadvisor
    Args:
        datapath: string data_path
        parameter: input parameter
    Returns:
        json string of result info
        result must be ad_result
    """
    log.ad_log(log.AD_INFO, f"hccl analysis Model start running, log and recommendation will be saved in {datapath}")
    parameters = json.loads(parameter)
    cluster_rank = parameters.get("rank_size")
    cluster_rank = int(cluster_rank) \
        if cluster_rank and (not isinstance(cluster_rank, str) or cluster_rank.isdigit()) else None
    step_num = parameters.get("step_num")
    step_num = int(step_num) if step_num and (not isinstance(step_num, str) or step_num.isdigit()) else None

    if cluster_rank is None or cluster_rank <= 0:
        log.ad_print_and_log(log.AD_ERROR, "Input rank size is invalid, please check")
        return INVALID_RESULT
    

    # Critical path analysis and Hccl oprator analysis
    log.ad_print_and_log(log.AD_INFO, "Critical path analyzing...")
    start = time.time()
    top_com_op = run_critical_path_analysis(datapath, step_num)
    if top_com_op == Constant.DATA_PARSE_ERROR:
        return INVALID_RESULT
    critical_analysis_time = time.time() - start
    hccl_start_time = time.time()
    log.ad_print_and_log(log.AD_INFO, "Operator hccl data analyzing...")
    hcclanalysistool = HcclAnalysisTool(cluster_rank, datapath, True)
    for _, comm_op in top_com_op.items():
        if hcclanalysistool.run(comm_op, True, step_num) == Constant.HCCL_ANALYSIS_ERROR:
            log.ad_print_and_log(log.AD_ERROR, "The Hccl Analysis Model runs Failed")
            return INVALID_RESULT
    log.ad_log(log.AD_INFO, f"Critical path analysis time: {critical_analysis_time}, "
                            f"Hccl operator analysis time: {time.time() - hccl_start_time}")

    save_path = os.path.join(os.path.realpath(datapath), "recommendation", "visualization")
    generate_html(body=hcclanalysistool.html_body, save_path=save_path)
    log.ad_print_and_log(log.AD_INFO, f"visualization and hccl_analysis_result.html saved in {save_path}")
    log.ad_log(log.AD_INFO, f"Overall run time: {time.time() - start}")
    return hcclanalysistool.result.generate()
