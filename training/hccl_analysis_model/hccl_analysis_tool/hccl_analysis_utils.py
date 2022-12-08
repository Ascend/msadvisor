import csv
import os

from utils.constant import Constant


class HcclConfig:
    network_topology_mapping = dict()

    @classmethod
    def init_config(cls, rank_size, rank_num_per_server, rank_num_per_os):
        server_num = int(rank_size / rank_num_per_server) + 1
        os_num = int(rank_num_per_server / rank_num_per_os) + 1
        rank_list = list(range(rank_size))
        server_list = list(range(1, server_num + 1, 1))
        os_list = list(range(1, os_num + 1, 1))
        network_topology = dict()
        for server_idx in server_list:
            server_id = "server_{}".format(server_idx)
            rank_list_in_server = rank_list[(server_idx - 1) * rank_num_per_server:server_idx * rank_num_per_server]
            network_topology[server_id] = dict()
            network_topology[server_id]["rank_list"] = rank_list_in_server
            for os_idx in os_list:
                os_id = "os_{}".format(os_idx)
                rank_list_in_os = rank_list[(server_idx - 1) * rank_num_per_server + (os_idx - 1) * rank_num_per_os:
                                            (server_idx - 1) * rank_num_per_server + os_idx * rank_num_per_os]
                network_topology[server_id][os_id] = rank_list_in_os
        for rank_id in rank_list:
            cls.remap_network_topology(rank_id, os_list, network_topology)

    @classmethod
    def remap_network_topology(cls, rank_id, os_list, network_topology):
        cls.network_topology_mapping[rank_id] = dict()
        server_id = None
        os_id = None
        for server_key, server_value in network_topology.items():
            if rank_id in server_value.get("rank_list"):
                server_id = server_key.split("_")[-1]
            if server_id is None:
                continue
            for os_num in os_list:
                os_key = "os_{}".format(os_num)
                if rank_id in server_value.get(os_key):
                    os_id = os_num
                    break
            if server_id is not None and os_id  is not None:
                break
        cls.network_topology_mapping[rank_id]["server_id"] = server_id
        cls.network_topology_mapping[rank_id]["os_id"] = os_id

    @classmethod
    def get_server_and_os_id(cls, rank_id):
        server_id = cls.network_topology_mapping.get(rank_id, dict()).get("server_id")
        os_id = cls.network_topology_mapping.get(rank_id, dict()).get("os_id")
        return server_id, os_id


def update_transit_size_record(transit_record, transport_type, transit_size):
    if transport_type not in transit_record.keys():
        transit_record[transport_type] = dict()
    if transit_size not in transit_record[transport_type].keys():
        transit_record[transport_type][transit_size] = 1
    else:
        transit_record[transport_type][transit_size] += 1


def determine_rdma(trace_events, cur_index):
    """judge is_rdma"""
    if cur_index > len(trace_events) - Constant.RDMA_TRANSIT_OP_NUM:
        return False
    second_task_type = trace_events[cur_index + 1].get(Constant.ARGS).get(Constant.TASK_TYPE)
    third_task_type = trace_events[cur_index + 2].get(Constant.ARGS).get(Constant.TASK_TYPE)
    return second_task_type == Constant.RDMA_SEND and third_task_type == Constant.NOTIFY_WAIT


def get_rdma_communication_info(trace_events, cur_index):
    transit_size = parse_data(trace_events[cur_index].get(Constant.ARGS).get(Constant.SIZE))
    transit_time = 0
    rdma_fake_wait_time = 0
    rdma_fake_wait_time_info = dict()
    for index in range(cur_index, cur_index + Constant.RDMA_TRANSIT_OP_NUM, 1):
        transit_time += parse_data(trace_events[index].get(Constant.DUR)) / Constant.US_TO_MS
        task_type = trace_events[index].get(Constant.ARGS).get(Constant.TASK_TYPE)
        if task_type == Constant.NOTIFY_WAIT:
            src_rank = trace_events[index].get(Constant.ARGS).get(Constant.SRC_RANK)
            dst_rank = trace_events[index].get(Constant.ARGS).get(Constant.DST_RANK)
            wait_time = parse_data(trace_events[index].get(Constant.DUR)) / Constant.US_TO_MS
            rdma_fake_wait_time += wait_time
            if int(src_rank) == int("0xffffffff", 16) or int(dst_rank) == int("0xffffffff", 16):
                continue
            wait_info_key = "{}-{}".format(src_rank, dst_rank)
            update_record_dict(rdma_fake_wait_time_info, wait_info_key, wait_time)
    return [transit_time, transit_size, rdma_fake_wait_time, rdma_fake_wait_time_info]


def parse_data(input_data):
    """str to int"""
    if input_data is None:
        return 0
    # if isinstance(input_data, str):
    #     return int(input_data)
    # else:
    #     return input_data
    return int(input_data) if isinstance(input_data, str) else input_data


def update_record_dict(record_dict, key, value):
    if key not in record_dict.keys():
        record_dict[key] = value
    else:
        record_dict[key] += value


def get_communication_op_name_mapping(hccl_dir_path, step_trace_file):
    op_name_in_hccl = [entry.name for entry in os.scandir(hccl_dir_path) if entry.is_dir()]
    op_name_in_hccl_set = set({op_name.split("_")[0] for op_name in op_name_in_hccl})
    op_name_in_hccl_dict = dict()
    for item in op_name_in_hccl_set:
        op_name_in_hccl_dict[item] = sorted([op_name for op_name in op_name_in_hccl if op_name.split("_")[0] == item],
                                            key=lambda x: int(x.split("_")[1]))
    step_trace_info = get_step_trace_info(step_trace_file)
    op_name_in_step_trace = [step_trace_info[0][i] for i in range(0, len(step_trace_info[0]), 3)]
    op_name_in_step_trace_set = set(
        {op_name.split("_")[3:][-1].split("/")[-1].split("-")[0] for op_name in op_name_in_step_trace})
    op_name_in_step_trace_dict = dict()
    for item in op_name_in_step_trace_set:
        op_name_in_step_trace_dict[item] = [op_name.split("_")[3:][-1].split("/")[-1]
                                            for op_name in op_name_in_step_trace
                                            if op_name.split("_")[3:][-1].split("/")[-1].split("-")[0] == item]
    communication_op_mapping = dict()
    for hccl_key, hccl_value in op_name_in_hccl_dict.items():
        for step_trace_key, step_trace_value in op_name_in_step_trace_dict.items():
            if hccl_key.lower() == step_trace_key.lower():
                communication_op_mapping[hccl_key] = list(zip(hccl_value, step_trace_value))
    return communication_op_mapping


def get_step_trace_info(step_trace_file_path):
    with open(step_trace_file_path, "r") as src_file:
        csv_reader = csv.reader(src_file)
        communication_op_name = next(csv_reader)[9:]
        step_timestamp_info = [[info[0], float(info[1]) / 100, float(info[2]) / 100] for info in csv_reader
                               if info[0].isdigit()]
    return [communication_op_name, step_timestamp_info]






























