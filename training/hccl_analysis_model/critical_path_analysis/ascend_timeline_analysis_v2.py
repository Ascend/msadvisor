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
import time

from utils.constant import Constant
from utils.log import ad_log, ad_print_and_log, AD_INFO, AD_ERROR
from .ascend_timeline_visualization import AscendTimeAnalysisVisualization
from .critical_path_analysis import CriticalPathAnalysis
from .timeline_analysis_util import get_critical_timeline, get_timeline_info


class AscendTimelineAnalysis:
    def run(self, timeline_file, save_path, visualization=False, step_num=2):
        try:
            task_type_dict, task_types, pid_tid_dict, step_info, timeline_data = get_timeline_info(timeline_file)
        except Exception as e:
            ad_print_and_log(AD_ERROR, f"The {timeline_file} is invalid. error: {e}")
            return Constant.DATA_PARSE_ERROR
        analysis_step = step_info[step_num]
        step_event_list = AscendTimelineAnalysis.get_op_event_by_timestamp(timeline_data,
                                                                           analysis_step["start_timestamp"],
                                                                           analysis_step["end_timestamp"],
                                                                           pid_tid_dict)
        categorized_step_event_list = self.get_event_category(step_event_list, task_type_dict)
        aicore_num = self.get_task_num(categorized_step_event_list, task_types.get(Constant.AICORE))
        communication_num = self.get_task_num(categorized_step_event_list, task_types.get(Constant.COMMUNICATION))
        aicpu_num = self.get_task_num(categorized_step_event_list, task_types.get(Constant.AICPU))

        ad_log(AD_INFO, f"There are {aicore_num} aicore op, {aicpu_num} aicpu op, "
                        f"{communication_num} communication op before critical path   analysis")

        critical_path_event = CriticalPathAnalysis.get_critical_path(categorized_step_event_list)
        event_execution_type_analysis_data = \
            self.event_execution_type_analysis(critical_path_event, categorized_step_event_list)

        op_type_analysis = self.event_type_analysis(event_execution_type_analysis_data, task_types)

        top_communication_op = {}
        for idx, communication_op in enumerate(op_type_analysis[Constant.COMMUNICATION]["topk_op"]):
            ad_log(AD_INFO, f"Top {idx} communicat  ion op: {communication_op[Constant.NAME]}, "
                            f"serial_time: {communication_op['serial_time']}, op dur time: {communication_op['dur']}")
            top_communication_op[f"top_{idx}"] = communication_op[Constant.NAME]

        if visualization:
            visual_save_path = os.path.join(save_path, "recommendation", "critical_path_analysis_visualization")
            if not os.path.exists(visual_save_path):
                os.makedirs(visual_save_path)
            ad_print_and_log(AD_INFO, f"Ascend timeline analysis visualization saved in {visual_save_path}")
            self.event_execution_type_visualization(event_execution_type_analysis_data, visual_save_path)
            self.event_type_analysis_visualization(op_type_analysis, visual_save_path)
        return top_communication_op

    @staticmethod
    def get_op_event_by_timestamp(timeline_data, start_timestamp, end_timestamp, pid_tid_dict):
        filtered_event_list = []
        for event in timeline_data:
            if event.get(Constant.ARGS) is not None or "scope_level" in event.keys():
                continue
            if AscendTimelineAnalysis.filtered_specific_event(event, pid_tid_dict):
                continue
            event_start_timestamp = float(event.get(Constant.TS))
            event_end_timestamp = event_start_timestamp + float(event.get(Constant.DUR))
            if event_start_timestamp >= start_timestamp and event_end_timestamp <= end_timestamp:
                filtered_event_list.append(event)
        return filtered_event_list

    @staticmethod
    def filtered_specific_event(event, pid_tid_dict):
        if event.get(Constant.PID) is None or event.get(Constant.TID) is None:
            return True
        cur_pid = int(event.get(Constant.PID))
        cur_tid = event.get(Constant.TID)
        for v in pid_tid_dict.values():
            if cur_pid == v[Constant.PID] and cur_tid == v[Constant.TID]:
                return True
        return False

    @staticmethod
    def get_event_category(event_list, task_type_dict):
        filtered_event_list = []
        for event in event_list:
            for task_type, pid in task_type_dict.items():
                if int(event.get(Constant.PID)) == pid:
                    event[Constant.TASK_TYPE] = task_type
                    filtered_event_list.append(event)
        return filtered_event_list

    @staticmethod
    def get_task_num(event_list, task_type):
        count = 0
        for event in event_list:
            if event.get(Constant.TASK_TYPE) == task_type:
                count += 1
        return count

    @staticmethod
    def event_type_analysis(critical_path, task_type, top_type="serial_time", top_k=3):
        aicore_op_list = []
        aicpu_op_list = []
        hostcpu_op_list = []
        communication_op_list = []
        for event in critical_path:
            if event[Constant.TASK_TYPE] == task_type[Constant.AICORE]:
                aicore_op_list.append(event)
            if event[Constant.TASK_TYPE] == task_type[Constant.AICPU]:
                aicpu_op_list.append(event)
            if event[Constant.TASK_TYPE] == task_type[Constant.HOSTCPU]:
                hostcpu_op_list.append(event)
            if event[Constant.TASK_TYPE] == task_type[Constant.COMMUNICATION]:
                communication_op_list.append(event)

        sorted_aicore_op_list, aicore_op_num, aicore_op_time = \
            AscendTimelineAnalysis.get_op_num_and_time(aicore_op_list, top_type)
        sorted_aicpu_op_list, aicpu_op_num, aicpu_op_time = \
            AscendTimelineAnalysis.get_op_num_and_time(aicpu_op_list, top_type)
        sorted_hostcpu_op_list, hostcpu_op_num, hostcpu_op_time = \
            AscendTimelineAnalysis.get_op_num_and_time(hostcpu_op_list, top_type)
        sorted_communication_op_list, communication_op_num, communication_op_time = \
            AscendTimelineAnalysis.get_op_num_and_time(communication_op_list, top_type)
        ad_log(AD_INFO, f"There are {aicore_op_num} aicore op, {aicpu_op_num} aicpu op, "
                        f"{communication_op_num} communication op after critical path analysis")

        analysis_result = {
            Constant.AICORE: {"op_num": aicore_op_num,
                              "op_time": aicore_op_time,
                              "topk_op": sorted_aicore_op_list[0:top_k]},
            Constant.AICPU: {"op_num": aicpu_op_num,
                             "op_time": aicpu_op_time,
                             "topk_op": sorted_aicpu_op_list[0:top_k]},
            Constant.HOSTCPU: {"op_num": hostcpu_op_num,
                               "op_time": hostcpu_op_time,
                               "topk_op": sorted_hostcpu_op_list[0:top_k]},
            Constant.COMMUNICATION: {"op_num": communication_op_num,
                                     "op_time": communication_op_time,
                                     "topk_op": sorted_communication_op_list[0:top_k]}}
        return analysis_result

    @staticmethod
    def get_op_num_and_time(op_list, topk_type):
        sorted_op_list = sorted(op_list, key=lambda s: float(s[topk_type]), reverse=True)
        op_num = len(sorted_op_list)
        op_time = sum([float(op[Constant.DUR]) for op in sorted_op_list])
        return sorted_op_list, op_num, op_time

    @staticmethod
    def event_type_analysis_visualization(op_type_analysis, save_path):
        AscendTimeAnalysisVisualization.task_type_visualization(op_type_analysis, save_path)
        for k, v in op_type_analysis.items():
            if v["op_num"] > 0:
                AscendTimeAnalysisVisualization.hotspot_task_visualization(v["topk_op"], k, save_path)

    def event_execution_type_analysis(self, critical_path, event_list):
        """"get execution type of event in critical path """
        execution_type_analysis_result = []
        for event in critical_path:
            intersection_event_list = self.get_time_intersection_event(event, event_list)
            serial_time, parallel_time = self.get_event_serial_parallel_time(event, intersection_event_list)
            event["serial_time"] = serial_time
            event["parallel_time"] = parallel_time
            execution_type_analysis_result.append(event)
            # self.get_event_parallel_time({"ts": 627839249.7501373, Constant.DUR: 100}, intersection_event_list)
        return execution_type_analysis_result

    def get_time_intersection_event(self, cur_event, event_list):
        interval_event_list = []
        cur_ts, cur_te = self.get_event_start_end_time(cur_event)
        for event in event_list:
            event_ts, event_te = self.get_event_start_end_time(event)
            if event != cur_event and event_ts < cur_te and event_te > cur_ts and event_ts < event_te:
                interval_event_list.append(event)
        return interval_event_list

    def get_event_serial_parallel_time(self, cur_event, intersection_event_list):
        sorted_intersection_event = sorted(intersection_event_list, key=lambda s: float(s[Constant.TS]), reverse=False)
        individual_time_list = []
        while len(sorted_intersection_event) > 0:
            individual_event = sorted_intersection_event[0]
            sorted_intersection_event.remove(individual_event)
            individual_event_ts, individual_event_te = self.get_event_start_end_time(individual_event)
            removed_event_list = []
            for event in sorted_intersection_event:
                event_ts, event_te = self.get_event_start_end_time(event)
                if event_te <= individual_event_te and event != individual_event and event not in removed_event_list:
                    removed_event_list.append(event)
                if event_ts < individual_event_te < event_te:
                    individual_event_te = event_te
                    removed_event_list.append(event)
            for event in removed_event_list:
                sorted_intersection_event.remove(event)
            individual_time_list.append([individual_event_ts, individual_event_te])
        parallel_time = 0
        cur_ts, cur_te = self.get_event_start_end_time(cur_event)
        for time_record in individual_time_list:
            ts = cur_ts if time_record[0] <= cur_ts else time_record[0]
            te = cur_te if time_record[1] >= cur_te else time_record[1]
            parallel_time += te - ts
        serial_time = float(cur_event[Constant.DUR]) - parallel_time
        return serial_time, parallel_time

    @staticmethod
    def get_event_start_end_time(event):
        event_start_time = float(event[Constant.TS])
        event_end_time = float(event[Constant.TS]) + float(event[Constant.DUR])
        return event_start_time, event_end_time

    @staticmethod
    def event_execution_type_visualization(execution_type_data, save_path):
        step_time = float(execution_type_data[-1][Constant.TS] + execution_type_data[-1][Constant.DUR] -
                          execution_type_data[0][Constant.TS])
        serial_time = 0
        parallel_time = 0
        for event in execution_type_data:
            serial_time += event["serial_time"]
            parallel_time += event["parallel_time"]
        idle_time = step_time - serial_time - parallel_time

        execution_type_time_data = {"step_time": step_time, "idle_time": idle_time,
                                    "serial_time": serial_time, "parallel_time": parallel_time}
        AscendTimeAnalysisVisualization.execution_time_visualization(execution_type_time_data, save_path)


def run_critical_path_analysis(profiling_dir, step_num=None):
    timeline_analysit_time = time.time()
    input_trace_file = get_critical_timeline(profiling_dir, step_num)
    # 需要针对寻找关键timeline失败的场景进行判断
    if input_trace_file == Constant.DATA_PARSE_ERROR:
        ad_print_and_log(AD_ERROR, "muti_timeline_analysis failed, please check profiling data")
        return Constant.DATA_PARSE_ERROR
    timeline_analysis = AscendTimelineAnalysis()
    top_op = timeline_analysis .run(input_trace_file, profiling_dir)
    ad_log(AD_INFO, f"timeline_analysis time: {time.time() - timeline_analysit_time}")
    return top_op
