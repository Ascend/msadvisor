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
from abc import ABC
from dataclasses import dataclass

import numpy as np
from PIL import Image
from ..pre_process_base import PreProcessBase
from ...data_process_factory import PreProcessFactory


@dataclass
class ImageParam:
    mean: list
    std: list
    center_crop: int
    resize: int
    dataset_path: str
    dtype: str


class ImageNetPreProcess(PreProcessBase, ABC):
    def __call__(self, index, batch_size, worker, cfg, in_queue, out_queue):
        """
        和基类的参数顺序和个数需要一致
        """
        print("pre_process start")
        try:
            image_param = ImageNetPreProcess._get_params(cfg)

            files = os.listdir(image_param.dataset_path)
            files.sort()

            data = []
            file_paths = []
            file_len = len(files)
            for i in range(index, file_len, worker):
                img, file_path = ImageNetPreProcess.image_process(i, files, image_param)
                data.append(img)
                file_paths.append(file_path)

                if len(data) == batch_size:
                    out_queue.put([file_paths, np.stack(data)])
                    file_paths.clear()
                    data.clear()

            while len(data) and len(data) < batch_size:
                file_paths.append(file_paths[0])
                data.append(data[0])  # 数据集尾部补齐
                out_queue.put([file_paths, np.stack(data)])
        except Exception as err:
            print("pre_process failed error={}".format(err))

        print("pre_process end")

    @staticmethod
    def image_process(i, files, image_param):
        # RGBA to RGB
        tmp_path = os.path.join(image_param.dataset_path, files[i])
        image = Image.open(tmp_path).convert('RGB')
        image = ImageNetPreProcess.resize(image, image_param.resize)
        image = ImageNetPreProcess.center_crop(image, image_param.center_crop)
        if image_param.dtype == "fp32":
            img = np.array(image, dtype=np.float32)
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            img = img / 255.    # ToTensor: div 255
            img -= np.array(image_param.mean, dtype=np.float32)[:, None, None]
            img /= np.array(image_param.std, dtype=np.float32)[:, None, None]
        elif image_param.dtype == "int8":
            img = np.array(image, dtype=np.int8)
        else:
            raise RuntimeError("dtype is not support")

        return img, tmp_path

    @staticmethod
    def center_crop(img, output_size):
        if isinstance(output_size, int):
            output_size = (int(output_size), int(output_size))
        image_width, image_height = img.size
        crop_height, crop_width = output_size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))

    @staticmethod
    def resize(img, size, interpolation=Image.Resampling.BILINEAR):
        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                o_w = size
                o_h = int(size * h / w)
                return img.resize((o_w, o_h), interpolation)
            else:
                o_h = size
                o_w = int(size * w / h)
                return img.resize((o_w, o_h), interpolation)
        else:
            return img.resize(size[::-1], interpolation)

    @staticmethod
    def _get_params(cfg):
        try:
            mean = cfg["mean"]
            std = cfg["std"]
            center_crop = cfg["center_crop"]
            resize = cfg["resize"]
            dataset_path = cfg["dataset_path"]
            dtype = cfg["dtype"]
            real_path = os.path.realpath(dataset_path)
            return ImageParam(mean, std, center_crop, resize, real_path, dtype)
        except Exception as err:
            raise RuntimeError("get params failed error={}".format(err))


PreProcessFactory.add_pre_process("ImageNet", ImageNetPreProcess())
