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
import importlib
import platform

class Register():
    def __init__(self, path_name: str):
        self.path_name = path_name


    def _handle_errors(self, errors):
        if not errors:
            return

        for name, err in errors:
            print("Module {} import failed: {}".format(name, err))
        print("Please check modules")


    def _path_to_module_format(self, path: str):
        if path.endswith(".py"):
            if platform.system().lower() == 'windows':
                return path.replace("\\", ".").rstrip(".py").replace(".", "", 1)
            else:
                return path.replace("/", ".").rstrip(".py").replace("/", "", 1)

        return ""


    def _add_modules(self, modules: list):
        pwd_dir = os.getcwd()

        for root, dirs, files in os.walk(pwd_dir, topdown=False):
            if root.endswith(self.path_name):
                modules += [
                    self._path_to_module_format(os.path.join(root.split(pwd_dir)[1], file)) for file in files
                ]

    def import_modules(self):
        modules = []
        self._add_modules(modules)

        errors = []
        for module in modules:
            try:
                importlib.import_module(module)
            except ImportError as error:
                errors.append((module, error))

        self._handle_errors(errors)