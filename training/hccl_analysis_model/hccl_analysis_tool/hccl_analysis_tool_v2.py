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

import glob
import json
import os
import numpy as np

from utils.constant import Constant
from utils.profiling_data_check import iter_trace_file_check, step_trace_file_check
from training.utils.result import Result
from utils.generate_html import generate_body
from training.utils.log import AD_INFO, AD_ERROR, AD_WARN, ad_log, ad_print_and_log
from op_bandwidth_analysis_ import op_bandwidth_analysis
from op_communication_time_analysis import op_communication_time_analysis, get_op_communication_time_analysis_result
from hccl_analysis_utils import get_communication_op_name_mapping, get_step_trace_info, parse_data, \
    get_rdma_communication_info, update_transit_size_record, update_record_dict, determine_rdma, HcclConfig
from hccl_visualization_utils import get_communication_matrix_info
from hccl_data_visualization import HcclVisualization


class HcclAnalysisTool:
    def __init__(self, cluster_rank_size, profile_dir, visualization=False):
        self.cluster_rank_size = cluster_rank_size
        self.hccl_profile_dir = profile_dir
        self.step_trace_dir = profile_dir
        self.op_name_mapping = dict()
        self.rank_id_list = set()
        self.step_timestamp = dict()
        self.html_body = []
        self.result = Result()
        self.visualization = visualization

    def run(self, analysis_op_name, op_mapping_flag=False, step_num=None, iteration_num=None):
        op_info_record = dict()
        valid_step_num = set()
        op_info_record[analysis_op_name] = {}
        html_info = {}

        if op_mapping_flag:
            if self.parse_hccl_op_name() == Constant.DATA_PARSE_ERROR:
                return Constant.HCCL_ANALYSIS_ERROR
            hccl_op_name = self.op_name_mapping.get(analysis_op_name)
        else:
            hccl_op_name = analysis_op_name
            self.rank_id_list = \
                set([int(entry.name.split("_")[-1]) for entry in os.scandir(self.hccl_profile_dir) if entry.is_dir() and
                     Constant.HCCL_DIR_FORMAT in entry.name])
        if not hccl_op_name:
            ad_print_and_log(AD_ERROR, f"There is not hccl info for communicaiton operator {analysis_op_name}")
            return Constant.HCCL_ANALYSIS_ERROR
        if not self.cluster_rank_size:
            self.cluster_rank_size = len(self.rank_id_list)
        elif self.cluster_rank_size != len(self.rank_id_list):
            ad_print_and_log(AD_ERROR, f"The input rank size is {self.cluster_rank_size},"
                                       f"but got {len(self.rank_id_list)} step_trace files. Missing profiling data!")
            return Constant.HCCL_ANALYSIS_ERROR

        HcclConfig.init_config(self.cluster_rank_size, Constant.RANK_NUM_PER_SERVER, Constant.RANK_NUM_PER_OS)

        ad_log(AD_INFO, f"Analysis hccl data for {len(self.rank_id_list)} devices")
        get_op_info = self.record_op_info(op_info_record, analysis_op_name, valid_step_num, hccl_op_name)
        if get_op_info == Constant.DATA_PARSE_ERROR:
            return Constant.HCCL_ANALYSIS_ERROR
        ad_log(AD_INFO, f"Operator {analysis_op_name} information recorded from {hccl_op_name} successfully, ")
        ad_log(AD_INFO, f"valid step_num: {valid_step_num}")
        if step_num is None:
            step_num = min(valid_step_num)
        op_info = op_info_record.get(analysis_op_name).get(step_num)
        if op_info is None:
            ad_print_and_log(AD_ERROR, f"There is not step_num like {step_num}, please check")
            return Constant.HCCL_ANALYSIS_ERROR
        iteration_num = self.op_performance_analysis(op_info, html_info, analysis_op_name, iteration_num)
        if iteration_num == Constant.DATA_PARSE_ERROR:
            return Constant.HCCL_ANALYSIS_ERROR
        ad_log(AD_INFO, f"Operator {analysis_op_name} performance analysis successfully")
        visual_save_path = os.path.realpath(self.hccl_profile_dir)
        visual_save_path = os.path.join(visual_save_path, "recommendation", "visualization", f"{analysis_op_name}")

        generate_body(analysis_op_name, html_info)
        self.html_body.append(html_info)
        if self.visualization:
            self.visualize_communication_pattern(hccl_op_name, iteration_num, visual_save_path)
        return Constant.HCCL_ANALYSIS_OK

    def parse_hccl_op_name(self):
        step_trace_files = glob.glob(os.path.join(self.step_trace_dir, Constant.STEP_TRACE_FILE))
        if len(step_trace_files) == 0:
            ad_print_and_log(AD_ERROR,
                             f"There is not {Constant.STEP_TRACE_FILE} file in {os.path.realpath(self.step_trace_dir)}")
            return Constant.DATA_PARSE_ERROR
        for step_trace_file in step_trace_files:
            if step_trace_file_check(step_trace_file) == Constant.DATA_PARSE_ERROR:
                return Constant.DATA_PARSE_ERROR
            cur_rank_id = os.path.basename(step_trace_file).split("_")[3]
            step_trace_info = get_step_trace_info(step_trace_file)
            self.step_timestamp[cur_rank_id] = step_trace_info[1]
        self.get_all_op_name_mapping(step_trace_files)
        return Constant.DATA_PARSE_OK

    def get_all_op_name_mapping(self, step_trace_files):
        for step_trace_file in step_trace_files:
            test_step_trace = step_trace_file
            cur_rank_id = test_step_trace.split("/")[-1].split("_")[-3]
            hccl_file_dir = os.path.join(self.hccl_profile_dir, Constant.HCCL_DIR_FORMAT + cur_rank_id)
            if not os.path.exists(hccl_file_dir):
                ad_print_and_log(AD_ERROR, f"hccl_file_dir: {hccl_file_dir} not exist")
                continue
            self.rank_id_list.add(int(cur_rank_id))
            op_name_mapping = get_communication_op_name_mapping(hccl_file_dir, step_trace_file)
            self.update_op_name_mapping(op_name_mapping)

    def update_op_name_mapping(self, op_name_mapping):
        for hccl_type, hccl_mapping in op_name_mapping.items():
            for cur_mapping in hccl_mapping:
                if cur_mapping[1] not in self.op_name_mapping.keys():
                    self.op_name_mapping[cur_mapping[1]] = cur_mapping[0]

    def record_op_info(self, op_info_record, analysis_op_name, valid_step_num, hccl_op_name):
        for cur_rank_id in self.rank_id_list:
            hccl_op_dir = f"{os.path.join(self.hccl_profile_dir, Constant.HCCL_DIR_FORMAT)}{cur_rank_id}/{hccl_op_name}"

            # The communication operator may not exist on the current card
            if not os.path.exists(hccl_op_dir):
                continue

            hccl_analysis_file_format = "{}/iter*.trace".format(hccl_op_dir)
            hccl_trace_files = glob.glob(hccl_analysis_file_format)

            if not hccl_trace_files:
                ad_print_and_log(AD_ERROR, f"No trace file in {hccl_op_dir}")
                return Constant.DATA_PARSE_ERROR

            op_info_result = list(map(self.parse_op_trace, hccl_trace_files))
            for op_info in op_info_result:
                if op_info == Constant.DATA_PARSE_ERROR:
                    return Constant.DATA_PARSE_ERROR
                valid_step_num.add(op_info[0])
                if op_info[0] not in op_info_record[analysis_op_name].keys():
                    op_info_record[analysis_op_name][op_info[0]] = dict()
                if cur_rank_id not in op_info_record[analysis_op_name][op_info[0]].keys():
                    op_info_record[analysis_op_name][op_info[0]][cur_rank_id] = dict()
                op_info_record[analysis_op_name][op_info[0]][cur_rank_id][op_info[1]] = op_info[2:]
        return Constant.DATA_PARSE_OK

    def op_performance_analysis(self, op_info, html_info, analysis_op_name, iteration_num=None):
        self.result.class_type = Constant.CLASS_TYPE.get(Constant.MODEL)
        self.result.error_code = Constant.ERROR_CODE.get(Constant.SUCCESS)
        self.result.summary = "Communication operator bottleneck analysis result as follows, " \
                              "See hccl_analysis_result.html for details"

        sorted_op_info = sorted(op_info.items(), key=lambda x: x[0])
        if iteration_num is None:
            iter_list = list(sorted_op_info[0][1].keys())
            iteration_num = min(iter_list)
        ad_log(AD_INFO, f"analysis iteration_num: {iteration_num}")

        iteration_num, op_time_analysis_result = op_communication_time_analysis(sorted_op_info, iteration_num)
        if iteration_num == Constant.DATA_PARSE_ERROR:
            return iteration_num
        op_hccl_analysis_extent_result, op_time_analysis_html_result = \
            get_op_communication_time_analysis_result(op_time_analysis_result, analysis_op_name)
        bottleneck_rank_transit_size, op_bandwidth_detail_extent_result, op_bandwidth_analysis_extent_result = \
            op_bandwidth_analysis(sorted_op_info, iteration_num, analysis_op_name)

        op_hccl_analysis_extent_result.value.append(op_bandwidth_analysis_extent_result.value[0])

        html_info['time_analysis_result'] = op_time_analysis_html_result
        html_info['bandwidth_analysis_result'] = op_bandwidth_analysis_extent_result.value
        html_info['bandwidth_details'] = op_bandwidth_detail_extent_result.get('value')
        html_info['detail_table_name'] = op_bandwidth_detail_extent_result.get('extend_title')

        self.result.extend_result.append(op_hccl_analysis_extent_result)
        # op_time_analysis_result bottleneck_rank_transit_size 可能出问题 需要检查
        visual_save_path = os.path.join(self.hccl_profile_dir, "recommendation", "visualization", f"{analysis_op_name}")
        if self.visualization and op_time_analysis_result:
            HcclVisualization.draw_communication_time_distribution(op_time_analysis_result, visual_save_path)
            HcclVisualization.draw_communication_size_distribution(bottleneck_rank_transit_size, visual_save_path)
        return iteration_num

    def parse_op_trace(self, file_path):
        try:
            with open(file_path, "r") as src_file:
                hccl_trace = json.load(src_file)
        except Exception as e:
            ad_print_and_log(AD_ERROR, f"Failed to load json file {file_path}, error: {e}")
            return Constant.HCCL_ANALYSIS_ERROR
        if not iter_trace_file_check(hccl_trace):
            ad_print_and_log(AD_ERROR, f"The {file_path} is invalid, please check. ")
            return Constant.HCCL_ANALYSIS_ERROR
        rank_id = hccl_trace.get(Constant.DEVICE_ID)
        iteration = hccl_trace.get(Constant.ITERATION)
        trace_events = hccl_trace.get(Constant.TRACEEVENTS)
        op_info = self.get_op_info(trace_events, rank_id)
        op_info.insert(1, iteration)
        return op_info

    def get_op_info(self, trace_event, rank_id):
        stream_thread_ids = set()
        for event in trace_event:
            stream_thread_ids.add(event.get(Constant.TID))
        main_stream_thread_id = max(stream_thread_ids)
        main_stream_events = [event for event in trace_event if event.get(Constant.TID) == main_stream_thread_id]
        op_timestamp = main_stream_events[0].get(Constant.TS, 0)

        step_id = self.calculate_step_by_timestamp(op_timestamp, rank_id)
        op_transit_size_record, op_transit_time_record = dict(), dict()
        elapse_time, transit_time, wait_time, sdma_time, rdma_time, wait_time_before_transit = 0, 0, 0, 0, 0, 0
        wait_flag = True
        for idx, event in enumerate(main_stream_events):
            # event = main_stream_events[idx]
            elapse_time += parse_data(event.get(Constant.DUR)) / Constant.US_TO_MS
            event_args = event.get(Constant.ARGS)
            transport_type, task_type = event_args.get(Constant.TRANSPORT_TYPE), event_args.get(Constant.TASK_TYPE)
            src_rank, dst_rank = event_args.get(Constant.SRC_RANK), event_args.get(Constant.DST_RANK)
            if transport_type == Constant.SDMA and task_type in Constant.SDMA_TRANSIT_ITEMS:
                wait_flag = False
                cur_transit_time = parse_data(event.get(Constant.DUR)) / Constant.US_TO_MS
                transit_time += cur_transit_time
                sdma_time += cur_transit_time
                sdma_transit_size = parse_data(event_args.get(Constant.SIZE))
                _, src_rank_os_id = HcclConfig.get_server_and_os_id(src_rank)
                _, dst_rank_os_id = HcclConfig.get_server_and_os_id(dst_rank)
                if src_rank_os_id == dst_rank_os_id and src_rank_os_id is not None and dst_rank_os_id is not None:
                    transport_type = Constant.HCCS
                if src_rank_os_id != dst_rank_os_id and src_rank_os_id is not None and dst_rank_os_id is not None:
                    transport_type = Constant.PCIE
                update_transit_size_record(op_transit_size_record, transport_type, sdma_transit_size)
                update_record_dict(op_transit_time_record, transport_type, cur_transit_time)
            if transport_type == Constant.RDMA and determine_rdma(main_stream_events, idx):
                wait_flag = False
                rdma_transit_result = get_rdma_communication_info(main_stream_events, idx)
                transit_time += rdma_transit_result[0]
                rdma_transit_size = rdma_transit_result[1]
                wait_time -= rdma_transit_result[2]
                rdma_time += rdma_transit_result[0]
                update_transit_size_record(op_transit_size_record, transport_type, rdma_transit_size)
                update_record_dict(op_transit_time_record, transport_type, rdma_transit_result[0])
            if task_type == Constant.NOTIFY_WAIT:
                if wait_flag:
                    wait_time_before_transit += parse_data(event.get(Constant.DUR)) / Constant.US_TO_MS
                wait_time += parse_data(event.get(Constant.DUR)) / Constant.US_TO_MS
        return [step_id, elapse_time, transit_time, [wait_time_before_transit, wait_time],
                op_transit_size_record, op_transit_time_record]

    def calculate_step_by_timestamp(self, timestamp, rank_id):
        """Calculate the step according to the timestamp"""
        step_timestamps_info = self.step_timestamp.get(rank_id)
        step_num = Constant.DEFAULT_STEP_NUM
        if step_timestamps_info is None or len(step_timestamps_info) == 0 or timestamp < step_timestamps_info[0][1]:
            return step_num
        if step_timestamps_info[len(step_timestamps_info) - 1][2] < timestamp:
            step_num = int(step_timestamps_info[len(step_timestamps_info) - 1][0])
        else:
            for item in step_timestamps_info:
                if item[1] <= timestamp <= item[2]:
                    step_num = int(item[0])
        return step_num

    @staticmethod
    def get_communication_matrix_from_record(communication_info_record, cluster_max_rank, matrix_type, factor=1):
        matrix_data = np.zeros((cluster_max_rank, cluster_max_rank), dtype=float)
        visualization_exception = False
        for link_rank, link_rank_info in communication_info_record.items():
            src_rank = int(link_rank.split("-")[0])
            dst_rank = int(link_rank.split("-")[1])
            transit_time = link_rank_info.get(Constant.TRANSIT_TIME, 0)
            transit_size = link_rank_info.get(Constant.TRANSIT_SIZE, 0)
            cur_bandwidth = 0
            if transit_time > 0:
                cur_bandwidth = transit_size / Constant.B_TO_G / (transit_time / Constant.MS_TO_S)
            if matrix_type == Constant.BANDWIDTH:
                data_value = float(format(cur_bandwidth, ".2f"))
            elif matrix_type == Constant.TRANSPORT_TYPE_:
                data_value = Constant.TRANSPORT_TYPE_DICT.get(link_rank_info.get(matrix_type, "None"), 0)
            elif matrix_type == Constant.BANDWIDTH_UTILIZATION:
                if transit_time > 0:
                    link_bw = Constant.TRANSPORT_TYPE_BW_DICT.get(link_rank_info.get(Constant.TRANSPORT_TYPE_))
                    data_value = float(format(cur_bandwidth / link_bw, ".2f"))
                else:
                    data_value = 0
            else:
                data_value = format(link_rank_info.get(matrix_type, 0) / factor, ".2f")
            if src_rank >= cluster_max_rank or dst_rank >= cluster_max_rank:
                visualization_exception = True
                continue
            matrix_data[src_rank][dst_rank] = data_value
        return matrix_data, visualization_exception

    def visualize_communication_pattern(self, hccl_op_name, iteration_num, visual_save_path):
        link_info_record = dict()
        for cur_rank_id in self.rank_id_list:
            link_analysis_file_format = "{}/iter{}.*".format(
                os.path.join(self.hccl_profile_dir, Constant.HCCL_DIR_FORMAT + str(cur_rank_id), hccl_op_name),
                iteration_num
            )
            link_trace_files = glob.glob(link_analysis_file_format)
            get_communication_matrix_info(link_trace_files, link_info_record, iteration_num)
        transport_type_data, type_exception = \
            self.get_communication_matrix_from_record(link_info_record, self.cluster_rank_size, "transport_type")
        HcclVisualization.draw_heatmap(
            transport_type_data, "Link Transport Type", self.cluster_rank_size, visual_save_path
        )

        matrix_memory_data, size_exception = \
            self.get_communication_matrix_from_record(link_info_record, self.cluster_rank_size,
                                                      "transit_size", factor=1024**2)
        HcclVisualization.draw_heatmap(
            matrix_memory_data, "Data Transmission Size(MB)", self.cluster_rank_size, visual_save_path)

        matrix_bw_data, bandwidth_exception = \
            self.get_communication_matrix_from_record(link_info_record, self.cluster_rank_size, "bandwidth", factor=1)
        HcclVisualization.draw_heatmap(
            matrix_bw_data, "Data Transmission Bandwidth(GB/s)", self.cluster_rank_size, visual_save_path)

        bw_utilization_data, util_exception = \
            self.get_communication_matrix_from_record(link_info_record, self.cluster_rank_size,
                                                      "bandwidth_utilization", factor=1)
        HcclVisualization.draw_heatmap(
            bw_utilization_data, "Bandwidth Utilization", self.cluster_rank_size, visual_save_path
        )

        matrix_transit_time, time_exception = \
            self.get_communication_matrix_from_record(link_info_record, self.cluster_rank_size,
                                                      "transit_time", factor=1)
        HcclVisualization.draw_heatmap(
            matrix_transit_time, "Data Transmission Time(ms)", self.cluster_rank_size, visual_save_path
        )

        exception = type_exception or size_exception or bandwidth_exception or util_exception or time_exception
        if exception:
            ad_print_and_log(AD_WARN, "An exception occurs during visualization generation,"
                                      "The possible cause is that the entered rank_size is incorrect or"
                                      "the step_trace file is missing")
