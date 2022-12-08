
from utils.result import ExtendResult
from utils.constant import Constant


def op_bandwidth_analysis(op_bandwidth_info, iteration_num, analysis_op_name):
    rank_list = []
    transit_time_list = []
    for rank_id, op_info in op_bandwidth_info:
        rank_list.append(rank_id)
        cur_op_info = op_info.get(iteration_num)
        transit_time_list.append(float(format(cur_op_info[1], ".2f")))
    max_transit_time = max(transit_time_list)
    min_bandwidth_rank = rank_list[transit_time_list.index(max_transit_time)]
    min_bandwidth_rank_info = op_bandwidth_info[transit_time_list.index(max_transit_time)]
    transit_size_info = min_bandwidth_rank_info[1].get(iteration_num)[3]
    transit_time_info = min_bandwidth_rank_info[1].get(iteration_num)[4]
    sdma_transit_size, sdma_bandwidth = 0, 0
    sdma_total_transit_time = 0
    hccs_transit_size, hccs_bandwidth = \
        get_transit_size_and_bandwidth(transit_size_info, transit_time_info, Constant.HCCS)
    sdma_transit_size += hccs_transit_size
    pcie_transit_size, pcie_bandwidth = \
        get_transit_size_and_bandwidth(transit_size_info, transit_time_info, Constant.PCIE)
    sdma_transit_size += pcie_transit_size
    rdma_transit_size, rdma_bandwidth = \
        get_transit_size_and_bandwidth(transit_size_info, transit_time_info, Constant.RDMA)
    if transit_size_info.get(Constant.SDMA) is not None:
        for k, v in transit_size_info.get(Constant.SDMA).items():
            sdma_transit_size += k * v
    if sdma_transit_size > 0:
        sdma_total_transit_time = transit_time_info.get(Constant.HCCS, 0) + \
                                  transit_time_info.get(Constant.PCIE, 0) + \
                                  transit_time_info.get(Constant.SDMA, 0)
        sdma_bandwidth = sdma_transit_size / Constant.B_TO_G / (sdma_total_transit_time / Constant.US_TO_MS)

    hccs_utilization = float(format(hccs_bandwidth / Constant.LINK_BANDWIDTH[Constant.HCCS], ".2f"))
    pcie_utilization = float(format(pcie_bandwidth / Constant.LINK_BANDWIDTH[Constant.PCIE], ".2f"))
    rdma_utilization = float(format(rdma_bandwidth / Constant.LINK_BANDWIDTH[Constant.RDMA], ".2f"))

    hccs_large_packet_flag = \
        hccs_transit_size > 0 and hccs_utilization < Constant.BANDWIDTH_THRESHOLD and \
        op_message_size_analysis(transit_size_info, Constant.HCCS)
    pcie_large_packet_flag = \
        pcie_transit_size > 0 and pcie_utilization < Constant.BANDWIDTH_THRESHOLD and \
        op_message_size_analysis(transit_size_info, Constant.PCIE)
    rdma_large_packet_flag = \
        rdma_transit_size > 0 and rdma_utilization < Constant.BANDWIDTH_THRESHOLD and \
        op_message_size_analysis(transit_size_info, Constant.RDMA)

    sdma_info = [sdma_transit_size, sdma_total_transit_time, sdma_bandwidth]
    hccs_info = [hccs_transit_size, transit_time_info.get(Constant.HCCS, 0), hccs_bandwidth, hccs_utilization,
                 hccs_large_packet_flag]
    pcie_info = [pcie_transit_size, transit_time_info.get(Constant.PCIE, 0), pcie_bandwidth, pcie_utilization,
                 pcie_large_packet_flag]
    rdma_info = [rdma_transit_size, transit_time_info.get(Constant.RDMA, 0), rdma_bandwidth, rdma_utilization,
                 rdma_large_packet_flag]
    op_bandwidth_detail_extend_result = \
        get_op_bandwidth_detail_result(min_bandwidth_rank, sdma_info, hccs_info, pcie_info, rdma_info, analysis_op_name)
    op_bandwidth_analysis_extend_result = \
        get_op_bandwidth_bottleneck_info(sdma_info, hccs_info, pcie_info, rdma_info, analysis_op_name)
    return transit_size_info, op_bandwidth_detail_extend_result, op_bandwidth_analysis_extend_result


def get_op_bandwidth_detail_result(rank_id, sdma_info, hccs_info, pcie_info, rdma_info, analysis_op_name):
    op_bandwidth_detail_extend_result = ExtendResult()
    op_bandwidth_detail_extend_result.type = Constant.EXTEND_TYPE[Constant.TABLE]
    op_bandwidth_detail_extend_result.extend_title = \
        f"Communication OP {analysis_op_name} Bandwidth Detail (rank: {rank_id})"
    op_bandwidth_detail_extend_result.data_type.append(Constant.EXTEND_DATA_TYPE[Constant.STR])
    op_bandwidth_detail_extend_result.data_type.append(Constant.EXTEND_DATA_TYPE[Constant.DOUBLE])
    op_bandwidth_detail_extend_result.data_type.append(Constant.EXTEND_DATA_TYPE[Constant.DOUBLE])
    op_bandwidth_detail_extend_result.data_type.append(Constant.EXTEND_DATA_TYPE[Constant.DOUBLE])
    op_bandwidth_detail_extend_result.data_type.append(Constant.EXTEND_DATA_TYPE[Constant.STR])

    op_bandwidth_detail_extend_result.key.append("Communication Link")
    op_bandwidth_detail_extend_result.key.append("TransitSize(MB)")
    op_bandwidth_detail_extend_result.key.append("TransitTime(ms)")
    op_bandwidth_detail_extend_result.key.append("Bandwidth(GB/s)")
    op_bandwidth_detail_extend_result.key.append("BandwidthUtilization")

    sdma_value = get_communication_link_value(Constant.SDMA, sdma_info)
    op_bandwidth_detail_extend_result.value.append(sdma_value)
    hccs_value = get_communication_link_value(Constant.HCCS, hccs_info)
    op_bandwidth_detail_extend_result.value.append(hccs_value)
    pcie_value = get_communication_link_value(Constant.PCIE, pcie_info)
    op_bandwidth_detail_extend_result.value.append(pcie_value)
    rdma_value = get_communication_link_value(Constant.RDMA, rdma_info)
    op_bandwidth_detail_extend_result.value.append(rdma_value)
    return op_bandwidth_detail_extend_result


def get_communication_link_value(transport_type, info):
    value = [
        transport_type,
        "{:.2f}".format(info[0] / (1024 ** 2)),
        "{:.2f}".format(info[1]),
        "{:.2f}".format(info[2])
    ]
    if transport_type == Constant.SDMA:
        value.append("None(None/None)")
    else:
        bandwidth = Constant.LINK_BANDWIDTH[transport_type]
        value.append("{}({}/{})".format(info[3], info[2], bandwidth))
    return value


def get_op_bandwidth_bottleneck_info(sdma_info, hccs_info, pcie_info, rdma_info, analysis_op_name):
    op_bandwidth_analysis_extend_result = ExtendResult()
    op_bandwidth_analysis_extend_result.type = Constant.EXTEND_TYPE[Constant.LIST]
    op_bandwidth_analysis_extend_result.data_type.append(Constant.EXTEND_DATA_TYPE[Constant.STR])
    op_bandwidth_analysis_extend_result.extend_title = \
        f"Communication operator {analysis_op_name} bandwidth analysis result:"
    if sdma_info[0] > rdma_info[0]:
        op_bandwidth_analysis_extend_result.value.append("SDMA Communication is the Dominated Bottleneck")
    else:
        op_bandwidth_analysis_extend_result.value.append("RDMA Communication is the Dominated Bottleneck")
    hccs_analysis_result = get_op_bandwidth_analysis_result(Constant.HCCS, hccs_info)
    if hccs_analysis_result:
        op_bandwidth_analysis_extend_result.value.append(hccs_analysis_result)
    pcie_analysis_result = get_op_bandwidth_analysis_result(Constant.PCIE, pcie_info)
    if pcie_analysis_result:
        op_bandwidth_analysis_extend_result.value.append(pcie_analysis_result)
    rdma_analysis_result = get_op_bandwidth_analysis_result(Constant.RDMA, rdma_info)
    if rdma_analysis_result:
        op_bandwidth_analysis_extend_result.value.append(rdma_analysis_result)
    return op_bandwidth_analysis_extend_result


def get_op_bandwidth_analysis_result(transport_type, info):
    value = None
    if info[0] <= 0:
        return value
    if info[3] < Constant.BANDWIDTH_THRESHOLD:
        if info[4]:
            if transport_type == Constant.PCIE:
                value = f"{transport_type} Bandwidth between P2P is inefficiency, the utilization is {info[3]}, " \
                        f"please check pcie bandwidth contention"

            if transport_type == Constant.HCCS:
                value = f"{transport_type} Bandwidth is inefficiency, the utilization is {info[3]}, " \
                        f"please check hccs config"

            if transport_type == Constant.RDMA:
                value = f"{transport_type} Bandwidth is inefficiency, the utilization is {info[3]}, " \
                        f"please check switch config"

        else:
            if transport_type == Constant.PCIE:
                value = f"{transport_type} Bandwidth between P2P is inefficiency, the utilization is {info[3]}, " \
                        f"Cause the packet is too small"
            else:
                value = f"{transport_type} Bandwidth is inefficiency, the utilization is {info[3]}, " \
                        f"Cause the packet is too small"
    else:
        value = f"{transport_type} Bandwidth is fully utilized"
    return value


def op_message_size_analysis(message_size_info, message_type):
    message_size = message_size_info.get(message_type)
    packet_num = 0
    large_packet_num = 0
    message_size_threshold = Constant.MESSAGE_SIZE_THRESHOLD[message_type]
    for k, v in message_size.items():
        cur_message_size = k / 1024 / 1024
        if cur_message_size >= message_size_threshold:
            large_packet_num += v
        packet_num += v
    large_packet_ratio = large_packet_num / packet_num
    if large_packet_ratio >= Constant.LARGE_MESSAGE_RATE:
        return True
    else:
        return False


def get_transit_size_and_bandwidth(transit_size_info, transit_time_info, transport_type):
    transit_size = 0
    bandwidth = 0
    if transit_size_info.get(transport_type) is not None:
        for data_size, num in transit_size_info.get(transport_type).items():
            transit_size += data_size * num
        bandwidth = float(format(
            transit_size / Constant.B_TO_G / (transit_time_info.get(transport_type) / Constant.US_TO_MS), ".2f")
        )
    return transit_size, bandwidth
