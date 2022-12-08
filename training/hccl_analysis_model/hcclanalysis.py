import json
import os
import time
from utils import log
from hccl_analysis_tool.hccl_analysis_tool_v2 import HcclAnalysisTool
from critical_path_analysis.ascend_timeline_analysis_v2 import run_critical_path_analysis
from utils.constant import Constant
from utils.generate_html import generate_html

INVALID_RESULT = ""


def evaluate(datapath, parameter):
    """
    interface function called by msadvisor
    Args:
        datapath: string data_path
        parameter: input parameter
    Returns:
        json string of result info
        result must be ad_result
    """
    log.ad_log(log.AD_INFO, f"hccl analysis Model start runing, log and recommendation will be saved in {datapath}")
    parameters = json.loads(parameter)
    cluster_rank = parameters.get("rank_size")
    cluster_rank = int(cluster_rank) if cluster_rank else None
    step_num = parameters.get("step_num")
    step_num = int(step_num) if step_num else None

    start = time.time()
    top_com_op = run_critical_path_analysis(datapath, step_num)
    if top_com_op == Constant.DATA_PARSE_ERROR:
        return INVALID_RESULT
    hccl_start_time = time.time()
    hcclanalysistool = HcclAnalysisTool(cluster_rank, datapath, True)
    for key, comm_op in top_com_op.items():
        if hcclanalysistool.run(comm_op, True, step_num) == Constant.HCCL_ANALYSIS_ERROR:
            log.ad_print_and_log(log.AD_ERROR, "The Hccl Analysis Model runs Failed")
            return INVALID_RESULT
    log.ad_log(log.AD_INFO, f"communication operator analysis time: {time.time() - hccl_start_time}")

    save_path = os.path.join(os.path.realpath(datapath), "recommendation", "visualization")
    generate_html(body=hcclanalysistool.html_body, save_path=save_path)
    log.ad_print_and_log(log.AD_INFO, f"visualization and hccl_analysis_result.html saved in {save_path}")
    log.ad_log(log.AD_INFO, f"Overall run time: {time.time() - start}")
    return hcclanalysistool.result.generate()

