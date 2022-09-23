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

import subprocess

from auto_optimizer.common.config import Config
from .compiler import Compiler

class OmCompiler(Compiler):

    def __init__(self, cfg):
        self.atc_cmd = [
                'atc',
                '--framework={}'.format(cfg['framework']),
                '--model={}'.format(cfg['model']),
                '--output={}'.format(cfg['output']),
                '--input_shape={}'.format(cfg['input_shape']),
                '--soc_version={}'.format(cfg['soc_version']),
        ]
        if cfg.get('custom_cfg', None):
            for key, value in self.cfg['custom_cfg'].items:
                self.atc_cmd.append('--{}={}'.format(key, value))

    def build_model(self):
        subprocess.run(self.atc_cmd, shell=False)
