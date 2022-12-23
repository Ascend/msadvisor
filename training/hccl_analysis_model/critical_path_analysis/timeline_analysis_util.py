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
from training.utils.log import ad_log, ad_print_and_log, AD_INFO, AD_ERROR


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
    rank_id = os.path.basename(timeline_file).split("_")[-1].split(".")[0]
    device_id = str(int(rank_id) % 8)
    try:
        with open(timeline_file) as f:
            timeline_data = json.load(f)
    except Exception as e:
        ad_print_and_log(AD_ERROR, f"Failed to load json file {timeline_file}, error: {e}")
        return Constant.DATA_PARSE_ERROR
    timeline_task_type_pid = TIMELINE_TASK_TYPE_PID
    timeline_task_type_pid[Constant.AICORE] = device_id
    specific_item_pid_tid = SPECIFIC_ITEM_PID_TID
    specific_item_pid_tid['step'] = {'pid': device_id, 'tid': 100000}
    task_type_dict = get_task_type_dict(timeline_data, timeline_task_type_pid)
    identifier = parse_identifier(timeline_data, specific_item_pid_tid)
    if task_type_dict == Constant.DATA_PARSE_ERROR or identifier == Constant.DATA_PARSE_ERROR:
        return Constant.DATA_PARSE_ERROR
    step_info = get_step_time_info(timeline_data, identifier["pid_tid"], identifier["step_name"])
    if step_info == Constant.DATA_PARSE_ERROR:
        return Constant.DATA_PARSE_ERROR
    timeline_info = {
        "task_type_pid": task_type_dict["task_type_pid"],
        "task_types": task_type_dict["task_types"],
        "pid_tid_dict": identifier["pid_tid"],
        "step_info": step_info,
        "timeline_data": timeline_data
    }
    return timeline_info


def get_task_type_dict(timeline_data, tasks_pid):
    task_type_pid = {}
    task_types = {}
    for event in timeline_data:
        if event.get(Constant.NAME) == "process_labels" and event.get(Constant.ARGS) is not None:
            if not event.get(Constant.ARGS).get('labels') or not event.get(Constant.PID):
                ad_print_and_log(AD_ERROR, "process_labels in timeline data is invalid!")
                return Constant.DATA_PARSE_ERROR
            for key, task_pid in tasks_pid.items():
                if task_pid == event.get(Constant.PID):
                    task_types[key] = event.get(Constant.ARGS).get('labels')
                    task_type_pid[event.get(Constant.ARGS).get('labels')] = int(event.get(Constant.PID))
    task_type_dict = {
        "task_type_pid": task_type_pid,
        "task_types": task_types
    }
    return task_type_dict


def parse_identifier(timeline_data, specific_item_pid_tid):
    """Get pid and tid of events. In addition, get step_name"""
    pid_tid_dict = dict()
    step_name = None
    for event in timeline_data:
        if event.get(Constant.ARGS) is None or event.get(Constant.NAME) != 'thread_name':
            continue

        if not event.get(Constant.ARGS).get(Constant.NAME):
            return Constant.DATA_PARSE_ERROR
        for key, item_pid_tid in specific_item_pid_tid.items():
            if event.get(Constant.PID) == item_pid_tid.get(Constant.PID) and event.get(
                    Constant.TID) == item_pid_tid.get(Constant.TID):
                cur_type = event.get(Constant.ARGS).get(Constant.NAME)
                pid_tid_dict[cur_type] = \
                    {Constant.PID: int(event.get(Constant.PID)), Constant.TID: event.get(Constant.TID)}

                if cur_type == 'Steps' or cur_type == 'Step':
                    step_name = cur_type
    identifier = {
        "pid_tid": pid_tid_dict,
        "step_name": step_name
    }
    return identifier


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
        data_invalid = \
            not step_event.get(Constant.NAME) or not step_event.get(Constant.TS) or not step_event.get(Constant.DUR)
        if data_invalid:
            ad_print_and_log(AD_ERROR, "Incomplete step data in timeline data!")
            return Constant.DATA_PARSE_ERROR
        step_id = int(step_event.get(Constant.NAME))
        start_timestamp, end_timestamp = get_event_start_end_time(step_event)
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
        timeline_info = get_timeline_info(ascend_timeline)
        if timeline_info == Constant.DATA_PARSE_ERROR:
            ad_print_and_log(AD_ERROR, f"The {ascend_timeline} is invalid, please check!")
            return Constant.DATA_PARSE_ERROR
        step_info = timeline_info["step_info"].get(step_num)
        if not step_info:
            ad_print_and_log(AD_ERROR, f"Got step info from ascend timeline failed, cur step_num: {step_num}")
            return Constant.DATA_PARSE_ERROR
        if step_info.get("dur_time") > max_time:
            max_time = step_info.get("dur_time")
            analysis_timeline = ascend_timeline
    rank_id = analysis_timeline.split("/")[-1].split("_")[-1].split(".")[0]
    ad_log(AD_INFO, f"step {step_num} of rank {rank_id} takes the longest E2E time")
    return analysis_timeline
