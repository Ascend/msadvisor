from utils.constant import Constant
from .timeline_analysis_util import get_event_by_pid_tid

TIME_INTERVAL = 1


class CriticalPathAnalysis:

    @staticmethod
    def get_critical_path(event_list):
        sorted_event_by_start_time = sorted(event_list, key=lambda s: float(s[Constant.TS]), reverse=False)
        sorted_event_by_end_time = sorted(event_list, key=lambda s: float(s[Constant.TS]) + float(s[Constant.DUR]),
                                          reverse=True)
        first_event = sorted_event_by_start_time[0]
        last_event = sorted_event_by_end_time[0]
        critical_path_event_list = []
        cur_event = last_event
        while cur_event[Constant.TS] != first_event[Constant.TS]:
            critical_path_event_list.append(cur_event)
            pre_event = CriticalPathAnalysis.get_pre_event_in_same_stream(event_list, cur_event)
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
        return sorted_critical_path

    @staticmethod
    def get_pre_event_in_same_stream(event_list, cur_event):
        pid = int(cur_event.get(Constant.PID))
        tid = cur_event.get(Constant.TID)
        cur_stream_event = get_event_by_pid_tid(event_list, pid, tid)
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
        return event_list[pre_idx] if pre_idx is not None else None
