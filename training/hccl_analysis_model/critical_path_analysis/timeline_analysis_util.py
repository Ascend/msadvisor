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

from utils.constant import Constant
from utils.log import ad_log, ad_print_and_log, AD_INFO, AD_ERROR


TIMELINE_TASK_TYPE_PID = {
    Constant.AICORE: '0',
    Constant.AICPU: 9000,
    Constant.HOSTCPU: 11000,
    Constant.COMMUNICATION: 10000
}

SPECIFIC_ITEM_PID_TID = {"step": {'pid': 0, 'tid': 100000},
                         "scope name": {'pid': 0, 'tid': 100001},
                         "merged computation op": {'pid': 12000, 'tid': 7999},
                         "pure communication op": {'pid': 12000, 'tid': 8000},
                         "merged communication op": {'pid': 12000, 'tid': 8001},
                         "free time": {'pid': 12000, 'tid': 8002}}
DEFAULT_STEP_NUM = 2


def get_timeline_info(timeline_file):
    rank_id = timeline_file.split("/")[-1].split("_")[-1].split(".")[0]
    device_id = str(int(rank_id) % 8)
    with open(timeline_file) as f:
        timeline_data = json.load(f)
    timeline_task_type_pid = TIMELINE_TASK_TYPE_PID
    timeline_task_type_pid[Constant.AICORE] = device_id
    specific_item_pid_tid = SPECIFIC_ITEM_PID_TID
    specific_item_pid_tid['step'] = {'pid': device_id, 'tid': 100000}
    task_type_dict, task_types = get_task_type_dict(timeline_data, timeline_task_type_pid)
    pid_tid_dict, step_name = get_pid_tid(timeline_data, specific_item_pid_tid)
    step_info = get_step_time_info(timeline_data, pid_tid_dict, step_name)
    return task_type_dict, task_types, pid_tid_dict, step_info, timeline_data


def get_task_type_dict(timeline_data, task_types_pid):
    task_type_dict = {}
    task_types = {}
    for event in timeline_data:
        if event.get(Constant.NAME) == "process_labels" and event.get(Constant.ARGS) is not None:
            for key, task_type_pid in task_types_pid.items():
                if task_type_pid == event.get(Constant.PID):
                    task_types[key] = event.get(Constant.ARGS).get('labels')
                    task_type_dict[event.get(Constant.ARGS).get('labels')] = int(event.get(Constant.PID))
    return task_type_dict, task_types


def get_pid_tid(timeline_data, specific_item_pid_tid):
    pid_tid_dict = dict()
    step_name = None
    for event in timeline_data:
        if event.get(Constant.ARGS) is None or event.get(Constant.NAME) != 'thread_name':
            continue

        for key, item_pid_tid in specific_item_pid_tid.items():
            if event.get(Constant.PID) == item_pid_tid.get(Constant.PID) and event.get(
                    Constant.TID) == item_pid_tid.get(Constant.TID):
                cur_type = event.get(Constant.ARGS).get(Constant.NAME)
                pid_tid_dict[cur_type] = \
                    {Constant.PID: int(event.get(Constant.PID)), Constant.TID: event.get(Constant.TID)}

                if cur_type == 'Steps' or cur_type == 'Step':
                    step_name = cur_type
    return pid_tid_dict, step_name


def get_event_start_end_time(event):
    event_start_time = float(event[Constant.TS])
    event_end_time = float(event[Constant.TS]) + float(event[Constant.DUR])
    return event_start_time, event_end_time


def get_step_time_info(timeline_data, pid_tid_dict, step_name):
    step_pid = pid_tid_dict.get(step_name).get(Constant.PID)
    step_tid = pid_tid_dict.get(step_name).get(Constant.TID)
    step_event_list = get_event_by_pid_tid(timeline_data, step_pid, step_tid)
    sorted_step_event_list = sorted(step_event_list, key=lambda s: int(s[Constant.NAME]), reverse=False)
    step_info_record = dict()
    for step_event in sorted_step_event_list:
        step_id = int(step_event.get(Constant.NAME))
        start_timestamp = float(step_event.get(Constant.TS))
        end_timestamp = start_timestamp + float(step_event.get(Constant.DUR))
        step_info_record[step_id] = {"start_timestamp": start_timestamp,
                                     "dur_time": float(step_event.get(Constant.DUR)),
                                     "end_timestamp": end_timestamp}
    return step_info_record


def get_event_by_pid_tid(timeline_data, pid, tid):
    event_list = []
    for event in timeline_data:
        if event.get(Constant.ARGS) is not None:
            continue
        if int(event.get(Constant.PID)) == pid and event.get(Constant.TID) == tid:
            event_list.append(event)
    return event_list


def get_critical_timeline(profiling_dir, step_num):
    """get critical timeline which takes the longest e2e time in specified or second step"""
    timeline_format = "ascend_timeline_display_*.json"
    if step_num is None:
        step_num = DEFAULT_STEP_NUM
    ascend_timeline_files = glob.glob(os.path.join(profiling_dir, timeline_format))
    if len(ascend_timeline_files) == 0:
        ad_print_and_log(AD_ERROR, f"There is not {timeline_format} file in "
                                   f"{os.path.realpath(profiling_dir)}")
        return Constant.DATA_PARSE_ERROR
    max_time = 0
    analysis_timeline = None
    for ascend_timeline in ascend_timeline_files:
        try:
            _, _, _, step_infos, _ = get_timeline_info(ascend_timeline)
        except Exception as e:
            ad_print_and_log(AD_ERROR, f"The {ascend_timeline} is invalid. analysis error: {e}")
            return Constant.DATA_PARSE_ERROR
        step_info = step_infos.get(step_num)
        if not step_info:
            ad_print_and_log(AD_ERROR, f"Got step info from ascend timeline failed, cur step_num: {step_num}")
            return Constant.DATA_PARSE_ERROR
        if step_info.get("dur_time") > max_time:
            max_time = step_info.get("dur_time")
            analysis_timeline = ascend_timeline
    rank_id = analysis_timeline.split("/")[-1].split("_")[-1].split(".")[0]
    ad_log(AD_INFO, f"step {step_num} of rank {rank_id} takes the longest E2E time")
    return analysis_timeline
