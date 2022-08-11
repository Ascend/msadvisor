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

import multiprocessing
# import numpy as np

from abc import ABC

# from PIL import Image

from ..pre_process_base import PreProcessBase
from ...data_process_factory import PreProcessFactory


class ImageNetPreProcess(PreProcessBase, ABC):
    def __init__(self):
        print("test init")
        pass

    def __call__(self, *args, **kwargs):
        print("pre_process")
        for key, value in kwargs.items():
            print(key, value)
        return True




PreProcessFactory.add_pre_process("ImageNet", ImageNetPreProcess())
