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

from ..utils.result import ExtendResult
from ..utils.constant import Constant
from ..utils.log import AD_ERROR, ad_print_and_log


def op_communication_time_analysis(op_time_info, iteration_num):
    rank_list = []
    elapse_time_list, transit_time_list, wait_time_list, wait_ratio_list = [], [], [], []
    synchronization_time_list, synchronization_ratio_list = [], []
    for rank_id, op_info in op_time_info:
        rank_list.append(int(rank_id))
        cur_op_time_info = op_info.get(iteration_num)
        elapse_time_list.append(float(format(cur_op_time_info[0], ".2f")))
        transit_time_list.append(float(format(cur_op_time_info[1], ".2f")))
        wait_time_list.append(float(format(cur_op_time_info[2][1], ".2f")))
        synchronization_time_list.append(float(format(cur_op_time_info[2][0], ".2f")))
        try:
            wait_ratio = float(format(cur_op_time_info[2][1] / (cur_op_time_info[1] + cur_op_time_info[2][1]), ".2f"))
            synchronization_ratio = \
                float(format(cur_op_time_info[2][0] / (cur_op_time_info[1] + cur_op_time_info[2][1]), ".2f"))
        except ZeroDivisionError as e:
            ad_print_and_log(AD_ERROR, f"The iter trace in profiling data of rank {rank_id} is abnormal, please check")
            return Constant.DATA_PARSE_ERROR, e
        wait_ratio_list.append(wait_ratio)
        synchronization_ratio_list.append(synchronization_ratio)
    max_wait_ratio = max(wait_ratio_list)
    max_synchronization_ratio = max(synchronization_ratio_list)
    slow_rank_flag = max_wait_ratio > Constant.WAIT_TIME_THRESHOLD
    min_wait_ratio = min(wait_ratio_list)
    slow_rank_id = rank_list[wait_ratio_list.index(min_wait_ratio)]
    bottleneck_info = [slow_rank_flag, slow_rank_id, max_wait_ratio, max_synchronization_ratio]
    return iteration_num, [rank_list, elapse_time_list, transit_time_list, synchronization_time_list, wait_time_list,
                           synchronization_ratio_list, wait_ratio_list, bottleneck_info]


def get_op_communication_time_analysis_result(op_time_analysis_result, analysis_op_name):
    op_hccl_analysis_extent_result = ExtendResult()
    op_time_analysis_html_result = []
    op_hccl_analysis_extent_result.type = Constant.EXTEND_TYPE.get(Constant.LIST)
    if op_time_analysis_result is not None:
        op_hccl_analysis_extent_result.extend_title = \
            f"{analysis_op_name} hccl analysis result:"
        op_hccl_analysis_extent_result.data_type.append(Constant.EXTEND_DATA_TYPE.get(Constant.STR))
        bottleneck_info = op_time_analysis_result[-1]
        if bottleneck_info[0]:
            analysis_result_0 = "There is a slow rank in the current communication, " \
                                "and slowest rank id is: %d, max wait ratio is: %.2f" \
                                % (bottleneck_info[1], bottleneck_info[2])
            op_hccl_analysis_extent_result.value.append(analysis_result_0)
            op_time_analysis_html_result.append(analysis_result_0)
            if bottleneck_info[3] > Constant.WAIT_TIME_THRESHOLD:
                op_time_analysis_html_result.append("The reason for this is that the ranks are not "
                                                    "synchronized, max synchronization ratio is: %.2f, "
                                                    "Please check whether the load of each card is balanced."
                                                    % (bottleneck_info[3]))
            else:
                op_time_analysis_html_result.append("The possible reason for this is that the communication "
                                                    "bandwidth between ranks is inconsistent")
        else:
            analysis_result_1 = "There is no slow rank in the current communication"
            op_hccl_analysis_extent_result.value.append(analysis_result_1)
            op_time_analysis_html_result.append(analysis_result_1)
    else:
        ad_print_and_log(AD_ERROR, f"Communication operator {analysis_op_name} communication time analysis failed")
        return Constant.DATA_PARSE_ERROR, Constant.DATA_PARSE_ERROR
    return op_hccl_analysis_extent_result, op_time_analysis_html_result
