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
from jinja2 import Environment, FileSystemLoader
from constant import Constant
from training.utils.log import AD_INFO, AD_ERROR, ad_print_and_log


def generate_html(body, save_path):
    env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
    template = env.get_template('hccl_analysis_template.html')
    if not os.path.exists(save_path):
        ad_print_and_log(AD_INFO, f"{save_path} not exits, Automatically make this dir")
        try:
            os.makedirs(save_path)
        except OSError as e:
            ad_print_and_log(AD_ERROR, f"Failed to make dir {save_path}, error: {e}")
            return Constant.HCCL_ANALYSIS_ERROR
    html_save_path = os.path.join(save_path, 'hccl_analysis_result.html')
    with open(html_save_path, 'w+', encoding="utf-8") as fout:
        html_content = template.render(body=body)
        fout.write(html_content)
    return Constant.HCCL_ANALYSIS_OK


def generate_body(analysis_op_name, html_info):
    html_info['visualization'] = {
        'bandwidth_utilization': os.path.join(analysis_op_name, 'Bandwidth Utilization.jpg'),
        'communication_time_analysis': os.path.join(analysis_op_name, 'Communication Time Analysis.jpg'),
        'data_transmission_bandwidth': os.path.join(analysis_op_name, 'Data Transmission Bandwidth.jpg'),
        'data_transmission_size': os.path.join(analysis_op_name, 'Data Transmission Size.jpg'),
        'data_transmission_time': os.path.join(analysis_op_name, 'Data Transmission Time.jpg'),
        'link_transport_type': os.path.join(analysis_op_name, 'Link Transport Type.jpg'),
        'packet_size': os.path.join(analysis_op_name, 'Packet Size Distribution.jpg')
    }
    html_info['op_name'] = analysis_op_name
