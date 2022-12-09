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

from utils.constant import Constant
from .hccl_analysis_utils import parse_data, update_record_dict, determine_rdma, get_rdma_communication_info
from .hccl_analysis_utils import HcclConfig


def get_communication_matrix_info(trace_file_list, communication_maxtrix_info_record, iteration_num):
    for trace_file in trace_file_list:
        with open(trace_file) as f:
            cur_traces = json.load(f)
        iteration = int(cur_traces.get(Constant.ITERATION))
        if iteration != iteration_num:
            continue
        trace_events = cur_traces.get(Constant.TRACEEVENTS)
        get_communication_info(trace_events, communication_maxtrix_info_record)


def get_communication_info(trace_events, communication_info_record):
    for trace_event_idx, trace_event in enumerate(trace_events):
        src_rank = trace_event.get(Constant.ARGS).get(Constant.SRC_RANK)
        dst_rank = trace_event.get(Constant.ARGS).get(Constant.DST_RANK)
        if src_rank is None or dst_rank is None:
            continue
        if src_rank == int("0xffffffff", 16) or dst_rank == int("0xffffffff", 16):
            continue
        link_key = "{}-{}".format(src_rank, dst_rank)
        if link_key not in communication_info_record.keys():
            communication_info_record[link_key] = dict()
        transport_type = trace_event.get(Constant.ARGS).get(Constant.TRANSPORT_TYPE)
        task_type = trace_event.get(Constant.ARGS).get(Constant.TASK_TYPE)
        event_dur_time = parse_data(trace_event.get(Constant.DUR)) / Constant.US_TO_MS
        transit_size = parse_data(trace_event.get(Constant.ARGS).get(Constant.SIZE))
        cur_transport_type = None
        if transport_type == Constant.LOCAL and task_type == Constant.NOTIFY_WAIT:
            update_record_dict(communication_info_record[link_key], Constant.WAIT_TIME, event_dur_time)
        if transport_type == Constant.LOCAL and task_type == Constant.REDUCE_TBE:
            update_record_dict(communication_info_record[link_key], Constant.TRANSIT_TIME, event_dur_time)
            update_record_dict(communication_info_record[link_key], Constant.TRANSIT_SIZE, transit_size)
            cur_transport_type = Constant.LOCAL
        if transport_type == Constant.SDMA and task_type in Constant.SDMA_TRANSIT_ITEMS:
            update_record_dict(communication_info_record[link_key], Constant.TRANSIT_TIME, event_dur_time)
            update_record_dict(communication_info_record[link_key], Constant.TRANSIT_SIZE, transit_size)
            _, src_rand_os_id = HcclConfig.get_server_and_os_id(src_rank)
            _, dst_rank_os_id = HcclConfig.get_server_and_os_id(dst_rank)
            if src_rand_os_id == dst_rank_os_id and src_rand_os_id is not None and dst_rank_os_id is not None:
                cur_transport_type = Constant.HCCS
            elif src_rand_os_id != dst_rank_os_id and src_rand_os_id is not None and dst_rank_os_id is not None:
                cur_transport_type = Constant.PCIE
            else:
                cur_transport_type = Constant.SDMA
        if transport_type == Constant.RDMA and determine_rdma(trace_events, trace_event_idx):
            cur_transport_type = Constant.RDMA
            rdma_info = get_rdma_communication_info(trace_events, trace_event_idx)
            update_record_dict(communication_info_record[link_key], Constant.TRANSIT_TIME, rdma_info[0])
            update_record_dict(communication_info_record[link_key], Constant.TRANSIT_SIZE, rdma_info[1])
            for rdma_wait_link_key in rdma_info[3].keys():
                if rdma_wait_link_key not in communication_info_record.keys():
                    communication_info_record[rdma_wait_link_key] = dict()
                cur_wait_time = -1 * rdma_info[3][rdma_wait_link_key]
                update_record_dict(communication_info_record[rdma_wait_link_key], Constant.WAIT_TIME, cur_wait_time)
        if cur_transport_type and Constant.TRANSPORT_TYPE_ not in communication_info_record[link_key].keys():
            communication_info_record[link_key][Constant.TRANSPORT_TYPE_] = cur_transport_type