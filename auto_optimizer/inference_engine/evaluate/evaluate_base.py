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

from abc import abstractmethod


class EvaluateBase(object):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, loop, batch_size, cfg, in_queue, out_queue):
        """
        loop: 循环次数，根据数据集大小、batch_size及worker计算得到loop次数
        batch_size：数据batch信息
        cfg: 配置文件，参考auto_optimizer\configs\cv\classification\example.py
        in_queue: 输入数据队列
        out_queue： 输出数据队列
        数据队列建议存放数据格式：[[batch_file_name], [[batch_data_0], [batch_data_1]]]
        batch_file_name：表示多batch时，对应数据集的文件名，用于精度评测
        batch_data_n：表示第n个输入or输出，batch_data_n包含batch组数据
        """
        pass
