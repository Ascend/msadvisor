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
    type='Resnet50',
    batch_size=4,
    engine=dict(
        pre_process=dict(
            type='ImageNet',
            resize=256,
            centercrop=256,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        inference=dict(
            type='acl',
        ),
        model_convert=dict(
            type='atc',
        ),
        post_process=dict(
            type='classification',
        ),
        evaluate=dict(
            type='classification',
        ),
    )
)
