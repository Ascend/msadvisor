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
import time

from utils.constant import Constant
from .timeline_analysis_util import get_event_by_pid_tid
from training.utils.log import ad_log, ad_print_and_log, AD_INFO, AD_ERROR

# time interval(ms)
TIME_INTERVAL = 1


class CriticalPathAnalysis:
    @staticmethod
    def get_critical_path(event_list):
        get_critical_path_start_time = time.time()

        sorted_event_by_start_time = sorted(event_list, key=lambda s: float(s[Constant.TS]), reverse=False)
        sorted_event_by_end_time = sorted(event_list, key=lambda s: float(s[Constant.TS]) + float(s[Constant.DUR]),
                                          reverse=True)
        first_event = sorted_event_by_start_time[0]
        last_event = sorted_event_by_end_time[0]
        critical_path_event_list = []
        cur_event = last_event
        ad_log(AD_INFO, f"sorted event_list cost time: {time.time() - get_critical_path_start_time}")

        event_dict = CriticalPathAnalysis.get_event_dict_by_pid_tid(event_list)
        critical_path_analysis_start_time = time.time()
        while cur_event[Constant.TS] != first_event[Constant.TS]:
            critical_path_event_list.append(cur_event)
            pre_event = CriticalPathAnalysis.get_pre_event_in_same_stream(cur_event, event_dict)
            if pre_event is None:
                pre_event = CriticalPathAnalysis.get_pre_event_in_different_stream(sorted_event_by_end_time, cur_event)
            pre_event_end_time = float(pre_event[Constant.TS]) + float(pre_event[Constant.DUR])
            if float(cur_event[Constant.TS]) - pre_event_end_time <= TIME_INTERVAL:
                cur_event = pre_event
            else:
                pre_event = CriticalPathAnalysis.get_pre_event_in_different_stream(sorted_event_by_end_time, cur_event)
                cur_event = pre_event
            if cur_event is None:
                break
            if cur_event[Constant.TS] == first_event[Constant.TS]:
                critical_path_event_list.append(cur_event)
        sorted_critical_path = sorted(critical_path_event_list, key=lambda s: float(s[Constant.TS]), reverse=False)
        ad_log(AD_INFO, f"critical_path_analysis cost time: {time.time() - critical_path_analysis_start_time}")
        return sorted_critical_path

    @ staticmethod
    def get_event_dict_by_pid_tid(timeline_data):
        event_dict = {}
        for event in timeline_data:
            if event.get(Constant.ARGS) is not None:
                continue
            identify = f"{event.get(Constant.PID)}_{event.get(Constant.TID)}"
            if identify not in event_dict.keys():
                event_dict[identify] = [event]
            else:
                event_dict[identify].append(event)
        return event_dict

    @staticmethod
    def get_pre_event_in_same_stream(cur_event, event_dict):
        pid = int(cur_event.get(Constant.PID))
        tid = cur_event.get(Constant.TID)
        cur_identify = f"{pid}_{tid}"
        cur_stream_event = event_dict.get(cur_identify)
        sorted_cur_stream_event = sorted(cur_stream_event, key=lambda s: float(s[Constant.TS]), reverse=True)
        cur_idx = len(sorted_cur_stream_event)
        for idx, event in enumerate(sorted_cur_stream_event):
            if event[Constant.TS] == cur_event[Constant.TS]:
                cur_idx = idx
                break
        pre_idx = cur_idx + 1
        return sorted_cur_stream_event[pre_idx] if pre_idx < len(sorted_cur_stream_event) else None

    @staticmethod
    def get_pre_event_in_different_stream(event_list, cur_event):
        pre_idx = None
        for idx, event in enumerate(event_list):
            if float(event[Constant.TS]) + float(event[Constant.DUR]) <= cur_event[Constant.TS] and \
                    event[Constant.TS] != cur_event[Constant.TS]:
                pre_idx = idx
                break
        return event_list[pre_idx] if pre_idx else None
