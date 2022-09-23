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


model = dict(
    name='Resnet50',
    type='onnx',
    dataset='ImageNet',
    batch_size=1,
    engine=dict(
        pre_process=dict(
            type='ImageNet',
            worker=1,
            dataset_path='./dataset/test_img/',
            resize=256,
            center_crop=[224, 224],
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            dtype='fp32',
        ),
        inference=dict(
            type='acl',
            model='./om/squeezenet1_1.om',
            device_id=0,
        ),
        model_convert=dict(
            type='atc',
            framework='5',
            model='./onnx/squeezenet1_1.onnx',
            output='./om/squeezenet1_1',
            input_format='NCHW',
            input_shape='image:1,3,224,224',
            soc_version='Ascend310P3',
        ),
        post_process=dict(
            type='classification',
        ),
        evaluate=dict(
            type='classification',
            ground_truth='./dataset/label.txt',
            topk=[1, 5]
        ),
    )
)
