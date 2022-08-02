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


class Register:
    def __init__(self, path_name: str):
        """
        path_name: 待注册文件夹的绝对路径
        """
        try:
            real_path = os.path.realpath(path_name)
            self.path_name = real_path
        except Exception as err:
            raise RuntimeError("Invalid file error={}".format(err))

    @staticmethod
    def _handle_errors(errors):
        if not errors:
            return

        for name, err in errors:
            raise RuntimeError("Module {} import failed: {}".format(name, err))

    @staticmethod
    def _path_to_module_format(path: str):
        """
        路径转换，把文件相对路径转换成python的import路径
        """
        if path.endswith(".py"):
            if platform.system().lower() == 'linux':
                return path.replace("/", ".").replace(".", "", 1)[:-3]
            else:
                return path.replace("\\", ".").replace(".", "", 1)[:-3]

        return ""

    def _add_modules(self, modules: list):
        pwd_dir = os.getcwd()

        for root, dirs, files in os.walk(self.path_name, topdown=False):
            modules += [
                self._path_to_module_format(os.path.join(root.split(pwd_dir)[1], file)) for file in files
            ]

    def import_modules(self):
        modules = []
        try:
            self._add_modules(modules)
        except Exception as error:
            print("add_modules failed, {}".format(error))

        errors = []
        for module in modules:
            if not module:
                continue

            try:
                importlib.import_module(module)
            except ImportError as error:
                errors.append((module, error))

        self._handle_errors(errors)
