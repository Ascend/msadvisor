from fileinput import filename
import os
import sys
import json
import numpy as np
from typing import Dict, List, Any

from op_test_frame.common import data_generator
from op_test_frame.utils import shape_utils
from op_test_frame.common.ascend_tbe_op import AscendOpKernel
from op_test_frame.common.ascend_tbe_op import AscendOpKernelRunner

class OpInfo(object):
    def __init__(self, bin_path, json_path) -> None:
        self.bin_path = bin_path
        self.json_path = json_path

class OpRunner(object):
    def __init__(self, op_info: OpInfo, out_path) :
        self.op_info = op_info
        self.out_path = out_path

    @staticmethod
    def _get_simulator_mode(run_cfg):
        if not run_cfg or not isinstance(run_cfg, dict):
            return "ca"
        return run_cfg.get("simulator_mode", "ca")

    @staticmethod
    def _get_simulator_lib_path(run_cfg):
        if not run_cfg or not isinstance(run_cfg, dict):
            return None
        return run_cfg.get("simulator_lib_path", None)
    
    @staticmethod
    def _is_dynamic_operator(json_path):
        op_conf = json.load(open(json_path, 'r'))
        op_para_size = op_conf.get('opParaSize')
        if isinstance(op_para_size, str):
            if not op_para_size.isdigit():
                return False
            op_para_size = int(op_para_size)
        return op_para_size > 0

    @staticmethod
    def _gen_input_params():
        params = [{"shape": (1, 10000, 10000), "dtype": "float16", "param_type": "input", "value_range": [-2, 2]},
            {"shape": (1, 10000, 10000), "dtype": "float16", "param_type": "output", "value_range": [-2, 2]},]
        return params

    @staticmethod
    def _get_param_type(one_param):
        if not one_param:
            return None
        if isinstance(one_param, (tuple, list)):
            if not one_param or not isinstance(one_param[0], dict):
                return None
            return one_param[0].get("param_type", None)
        if isinstance(one_param, dict):
            return one_param.get("param_type")
        return None

    @staticmethod
    def _get_input_outputs(param_list: List):
        def _add_to_params(params: List, one_param):
            if isinstance(one_param, list):
                for sub_param in one_param:
                    params.append(sub_param)
            else:
                params.append(one_param)

        input_list = []
        output_list = []
        for arg in param_list:
            param_type = OpRunner._get_param_type(arg)
            if param_type == "input":
                _add_to_params(input_list, arg)
            if param_type == "output":
                _add_to_params(output_list, arg)
        return input_list, output_list

    @staticmethod
    def _gen_input_data(param_info):
        if "value" in param_info.keys():
            return
        distribution = param_info.get("distribution", "uniform")
        value_range = param_info.get("value_range", [-2, 2])
        # if dynamic shape use run_shape, if static shape use shape
        shape = param_info.get("run_shape")
        if shape is None:
            shape = param_info.get("shape")
        dtype = param_info.get("dtype")
        data = data_generator.gen_data(data_shape=shape,
                                        min_value=value_range[0],
                                        max_value=value_range[1],
                                        dtype=dtype,
                                        distribution=distribution)
        param_info["value"] = data

    def run_kernel(self, run_soc_version: str, run_cfg: Dict[str, Any] = None):
        """
        执行算子仿真
        :param run_soc_version: 运行的aicore版本
        :param run_cfg: 仿真运行配置
        :return: 运行成功返回True，否则返回False
        """
        bin_path = self.op_info.bin_path
        json_path = self.op_info.json_path
        params = self._gen_input_params()
        input_info_list, output_info_list = self._get_input_outputs(params)
        input_data_list = []
        for input_info in input_info_list:
            self._gen_input_data(input_info)
            input_data_list.append(input_info.get("value"))
        op_kernel = AscendOpKernel(bin_path, json_path)
        op_kernel.set_input_info(input_info_list)
        op_kernel.set_output_info(output_info_list)
        simulator_mode = self._get_simulator_mode(run_cfg)
        simulator_dump_path = self.out_path
        with AscendOpKernelRunner(simulator_mode=simulator_mode,
                                  soc_version=run_soc_version,
                                  simulator_lib_path=self._get_simulator_lib_path(run_cfg),
                                  simulator_dump_path=simulator_dump_path) as runner:
            if not self._is_dynamic_operator(json_path):
                output_data_list = runner.run(op_kernel, inputs=input_data_list)
            else:
                print("[error] Not support dynamic operator.")
                return False

            if not isinstance(output_data_list, (tuple, list)):
                output_data_list = [output_data_list, ]
            for output_data in output_data_list:
                if output_data:
                    output_data.sync_from_device()
            if len(output_data_list) == 0:
                print('[error] Operator simulation run failed, bin_path={}.'.format(bin_path))
                return False
        return True

def scan_operator_bin_list(data_path: str) -> List[OpInfo]:
    """
    扫描算子列表
    :param data_path: 算子.o和.json所在路径
    :return: 返回获取的算子信息
    """
    operators_list = []
    file_list = os.listdir(data_path)
    for file_name in file_list:
        file_path = data_path + os.path.sep + file_name
        if not os.path.isfile(file_path) and not file_path.endswith('.o'):
            continue
        bin_path = file_path
        json_path = file_path[0:-2] + '.json'
        if not os.path.isfile(json_path):
            continue
        op_conf = json.load(open(json_path, 'r'))
        kernel_name = op_conf.get('kernelName')
        if kernel_name is None:
            continue
        operators_list.append(OpInfo(bin_path, json_path))
    return operators_list

if __name__ == '__main__':
    data_path = sys.argv[1]
    soc_version = sys.argv[2] # 白名单校验
    simulation_path = sys.argv[3]
    op_infos = scan_operator_bin_list(data_path)
    if len(op_infos) == 0:
        print('[error] There is no operator bin file or json file, please check.')
        exit(1)
    run_cfg = { 'simulator_lib_path' : simulation_path }
    for op_info in op_infos:
        op_name = os.path.basename(op_info.bin_path)[0:-2]
        dump_path = data_path + os.path.sep + 'dump'
        if not os.path.exists(dump_path):
            os.mkdir(dump_path)
        op_dump_path = dump_path + os.path.sep + op_name
        if not os.path.exists(op_dump_path):
            os.mkdir(op_dump_path)
        # execute simulator
        print('[info] Start to execute simulation, operator name: {}'.format(op_name))
        op_runner = OpRunner(op_info, op_dump_path)
        op_runner.run_kernel(soc_version, run_cfg)
        print('[info] Execute simulation finished.')
    exit(0)
