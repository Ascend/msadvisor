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


import csv
from constant import Constant
from training.utils.log import AD_ERROR, ad_print_and_log


def iter_trace_file_check(hccl_trace):
    """校验hccl iter*.trace文件"""
    valid_trace_flag = \
        check_data_type(hccl_trace, Constant.DEVICE_ID, str, is_digit=True) and \
        check_data_type(hccl_trace, Constant.ITERATION, int) and \
        check_data_type(hccl_trace, Constant.TRACEEVENTS, list)
    is_trace_events_valid = trace_events_check(hccl_trace.get(Constant.TRACEEVENTS))
    return valid_trace_flag and is_trace_events_valid


def trace_events_check(trace_events):
    """Check trace events in iter.trace file"""
    is_trace_events_valid = True
    for event in trace_events:
        valid_trace_event_flag = \
            check_data_type(event, Constant.TID, int) and \
            check_data_type(event, Constant.PID, str, is_digit=True) and \
            check_data_type(event, Constant.TS, float) and event.get(Constant.TS) > 0 and \
            check_data_type(event, Constant.DUR, float) and event.get(Constant.TS) > 0 and\
            check_data_type(event, Constant.PH, str) and \
            check_data_type(event, Constant.NAME, str) and \
            check_data_type(event, Constant.ARGS, dict)
        is_event_args_valid = events_args_check(event.get(Constant.ARGS))
        if not valid_trace_event_flag or not is_event_args_valid:
            is_trace_events_valid = False
            break
    return is_trace_events_valid


def events_args_check(event_args):
    """Check args in trace event"""
    valid_args_flag = \
        check_data_type(event_args, Constant.NOTIFY_ID, int) and \
        check_data_type(event_args, Constant.DUR_EST, float) and \
        check_data_type(event_args, Constant.STAGE, (int, str)) and \
        check_data_type(event_args, Constant.STEP, (int, str)) and \
        (event_args.get(Constant.BANDWIDTH) == Constant.NULL or
         check_data_type(event_args, Constant.BANDWIDTH, float)) and \
        check_data_type(event_args, Constant.STREAM_ID, int) and \
        check_data_type(event_args, Constant.TASK_ID, int) and \
        check_data_type(event_args, Constant.TASK_TYPE, str) and \
        check_data_type(event_args, Constant.SRC_RANK, int) and \
        check_data_type(event_args, Constant.DST_RANK, int) and \
        check_data_type(event_args, Constant.TRANSPORT_TYPE, str) and \
        (event_args.get(Constant.SIZE) is None or check_data_type(event_args, Constant.SIZE, int))
    return valid_args_flag


def step_trace_file_check(step_trace_file_path):
    with open(step_trace_file_path, "r") as src_file:
        csv_reader = csv.reader(src_file)
        header = next(csv_reader)
        communication_op_names = header[9:]
        op_name_in_step_trace = [communication_op_names[idx] for idx in range(0, len(communication_op_names), 3)]
        for op_name in op_name_in_step_trace:
            try:
                communication_op_name = op_name.split("_")[3:][-1].split("/")[-1].split("-")[0]
            except IndexError:
                ad_print_and_log(AD_ERROR, f"Failed parse communication op name from {step_trace_file_path}, "
                                           f"The step trace file format is incorrect, please check")
                return Constant.DATA_PARSE_ERROR

            if communication_op_name not in Constant.COMMUNICATIONS_OPS:
                ad_print_and_log(AD_ERROR, f"{communication_op_name} is a invalid communication operator name. "
                                           f"The step trace file format is incorrect, please check")
                return Constant.DATA_PARSE_ERROR
        for csv_row in csv_reader:
            if len(csv_row) != len(header):
                return Constant.DATA_PARSE_ERROR

            is_data_valid = isinstance(csv_row[1], float) and isinstance(csv_row[2], float)
            if not is_data_valid:
                return Constant.DATA_PARSE_ERROR
    return Constant.DATA_PARSE_OK


def check_data_type(data, key, value_type, is_digit=False):
    if isinstance(data.get(key), value_type) and True if not is_digit else data.get(key).isdigit():
        return True

    if value_type == str:
        ad_print_and_log(AD_ERROR, f"The value of {key} expected type: {value_type} and is_digit: {is_digit},"
                                   f" but get {data.get(key)}. please check profiling data")
    else:
        ad_print_and_log(AD_ERROR, f"The value of {key} expected type: {value_type}, "
                                   f"but get {data.get(key)} type: {type(data.get(key))}. "
                                   f"please check profiling data")
    return False


def check_rank_and_step(rank_size, step_num):
    if rank_size is None or (isinstance(rank_size, str) and not rank_size.isdigit()):
        return False
    a = rank_size is not None and (not isinstance(rank_size, str) or rank_size.isdigit())
    return True

