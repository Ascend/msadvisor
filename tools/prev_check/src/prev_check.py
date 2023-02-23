# Copyright 2023 Huawei Technologies Co., Ltd
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
import re
import getpass
import platform

from enum import Enum, unique
from typing import Dict

# cann version : torch+torch_npu version
CANN_AND_TORCH_VERSION_MAP = {
    '5.0.2': ['1.5.0+ascend.post2'],
    '5.0.3': ['1.5.0+ascend.post3'],
    '5.0.4': ['1.5.0+ascend.post4'],
    '5.0.5': ['1.5.0+ascend.post4'],
    '5.1.RC1': ['1.5.0+ascend.post5', '1.8.1+ascend.rc1'],
    '5.1.RC2': ['1.5.0+ascend.post6', '1.8.1+ascend.rc2', '1.8.0a0'],
    '6.0.RC1': ['1.5.0+ascend.post7', '1.8.1+ascend.rc3', '1.8.0a0', '1.11.0a0'],
    '6.0.0': ['1.5.0+ascend.post8', '1.8.1+ascend', '1.8.0a0', '1.11.0a0'],
    '6.0.1': ['1.5.0+ascend.post8', '1.8.1+ascend', '1.8.0a0', '1.11.0a0']
}
# cann version : torch_npu version
CANN_AND_TORCH_NPU_VERSION_MAP = {
    '5.1.RC1': ['1.8.1rc1'],
    '5.1.RC2': ['1.8.1rc2'],
    '6.0.RC1': ['1.8.1rc3', '1.11.0rc1'],
    '6.0.0': ['1.8.1', '1.11.0rc2'],
    '6.0.1': ['1.8.1', '1.11.0rc2']
}
# torch_npu version : torch version
TORCH_NPU_AND_TORCH_VERSION_MAP = {
    '1.8.1rc1': ['1.8.1+ascend.rc1'],
    '1.8.1rc2': ['1.8.1+ascend.rc2', '1.8.0a0'],
    '1.8.1rc3': ['1.8.1+ascend.rc3', '1.8.0a0'],
    '1.8.1': ['1.8.0a0'],
    '1.11.0rc1': ['1.11.0a0'],
    '1.11.0rc2': ['1.11.0a0']
}
# cann version: tuscancy version
CANN_AND_TUSCANCY_VERSION_MAP = {
    '5.0.2': ['1.78'],
    '5.0.3': ['1.79'],
    '5.0.4': ['1.80'],
    '5.0.5': ['1.80'],
    '5.1.RC1': ['1.81'],
    '5.1.RC2': ['1.82'],
    '6.0.RC1': ['1.83'],
    '6.0.0': ['1.84'],
    '6.0.1': ['1.84']
}
# cann version: driver version
CANN_AND_DRIVER_VERSION_MAP = {
    '5.0.2': ['1.78', '1.79', '1.80'],
    '5.0.3': ['1.78', '1.79', '1.80'],
    '5.0.4': ['1.78', '1.79', '1.80', '1.81'],
    '5.0.5': ['1.78', '1.79', '1.80', '1.81'],
    '5.1.RC1': ['1.80', '1.81', '1.82', '1.83', '1.84'],
    '5.1.RC2': ['1.81', '1.82', '1.83', '1.84'],
    '6.0.RC1': ['1.81', '1.82', '1.83', '1.84'],
    '6.0.0': ['1.81', '1.82', '1.83', '1.84', '1.85'],
    '6.0.1': ['1.81', '1.82', '1.83', '1.84', '1.85']
}


# common functions
def split_(msg: str, split: str = '='):
    if msg.count(split) == 0:
        return '', ''
    k, v = msg.split(split, 1)
    k = k.strip().rstrip()
    v = v.strip().rstrip()
    return k, v


def read_param(filepath: str, key: str, split: str = '=') -> str:
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        lines = f.readlines()
        for line in lines:
            k, v = split_(line, split)
            if k == key:
                return v if v != '' else None
    return None


def read_params(filepath: str, prefix: str = '', split: str = '=') -> Dict[str, list]:
    '''read all param from file if param has the prefix
    '''
    if not os.path.exists(filepath):
        return {}
    res: Dict[str, list] = {}
    with open(filepath) as f:
        lines = f.readlines()
        for line in lines:
            k, v = split_(line, split)
            if not k.startswith(prefix):
                continue
            if len(v) == 0:
                continue
            if k not in res:
                res[k] = []
            res[k].append(v)
    return res


def removeprefix(msg: str, prefix: str) -> str:
    '''remove prefix from msg
    '''
    if not msg.startswith(prefix):
        return msg
    return msg[len(prefix):]


@unique
class DataType(Enum):
    VERSION = 0,
    ENVIRON = 1


class Loader(object):
    def __init__(self) -> None:
        pass

    def load(self, data_type: DataType) -> bool:
        return True


class TorchLoader(Loader):
    def __init__(self) -> None:
        super().__init__()
        self._torch_version: str = None
        self._torch_npu_version: str = None

    @property
    def torch_version(self) -> Dict[str, str]:
        return self._torch_version

    @property
    def torch_npu_version(self) -> Dict[str, str]:
        return self._torch_npu_version

    def load(self, data_type: DataType) -> bool:
        if data_type is DataType.VERSION:
            return self._query_torch_version() and \
                self._query_torch_npu_version()
        else:
            print(f'[error] invalid data type {data_type}.')
            return False

    def _query_torch_npu_version(self) -> bool:
        try:
            import torch_npu
            self._torch_npu_version = torch_npu.version.__version__
        except ImportError as e:
            # if torch version is 1.5, no need to install torch_npu
            if not self._torch_version.startswith('1.5'):
                print(f'[error] {e}.')
                return False
            return True
        except AttributeError as e:
            print(f'[error] {e}.')
        if self._torch_npu_version is not None:
            print(f'[info] torch_npu version: {self._torch_npu_version}')
            return True
        return False

    def _query_torch_version(self) -> bool:
        try:
            import torch
            self._torch_version = torch.version.__version__
        except ImportError as e:
            print(f'[error] {e}.')
        except AttributeError as e:
            print(f'[error] {e}.')
        if self._torch_version is not None:
            print(f'[info] torch version: {self._torch_version}')
            return True
        return False


class CannLoader(Loader):
    def __init__(self) -> None:
        super().__init__()
        self._version: str = None
        self._environ: Dict[str, list] = None

        self._install_path: str = None
        self._install_path_queried: bool = False

    @property
    def version(self) -> str:
        return self._version

    @property
    def environ(self) -> Dict[str, list]:
        return self._environ

    def load(self, data_type: DataType) -> bool:
        if data_type is DataType.VERSION:
            return self._query_version()
        elif data_type is DataType.ENVIRON:
            return self._query_environ()
        else:
            print(f'[error] invalid data type {data_type}.')
            return False

    def _query_install_path(self) -> bool:
        if self._install_path_queried:
            return self._install_path is not None
        self._install_path_queried = True

        install_info_path = 'Ascend/ascend_cann_install.info'
        if getpass.getuser() == 'root':
            path_ = os.path.join('/etc', install_info_path)
        else:
            path_ = os.path.join(os.environ['HOME'], install_info_path)
        if not os.path.exists(path_):
            print(f'[error] cann package has not bean installed.')
            return False
        self._install_path = read_param(path_, key='Install_Path')
        if self._install_path is None:
            print(f'[error] no Install_Path param in {path_}.')
            return False
        if not os.path.exists(os.path.join(self._install_path, 'ascend-toolkit')):
            print('[error] cann package has not bean installed.')
            return False
        return True

    def _query_version(self) -> bool:
        if self._install_path is None:
            if not self._query_install_path():
                return False
        path_ = os.path.join(self._install_path, \
            f'ascend-toolkit/latest/{platform.machine()}-linux/ascend_toolkit_install.info')
        if not os.path.exists(path_):
            print(f'[error] {path_} not exist, please update cann package.')
            return False
        version = read_param(path_, key='version')
        if version is None:
            print(f'[error] no version param in {path_}, please check.')
            return False
        if version not in CANN_AND_TUSCANCY_VERSION_MAP:
            print(f'[error] cann version {version} not support.')
            return False
        print(f'[info] cann version: {version}')
        self._version = version
        return True

    def _query_environ(self) -> bool:
        set_prefix_ = 'export'

        if self._install_path is None:
            if not self._query_install_path():
                return False
        path_ = os.path.join(self._install_path, 'ascend-toolkit/set_env.sh')
        if not os.path.exists(path_):
            print(f'[error] set_env.sh not exist, please check.')
            return False
        params: Dict[str, list] = read_params(path_, prefix=set_prefix_)
        if len(params) == 0:
            print(f'[error] set_env.sh content is invalid, please check.')
            return False
        environ: Dict[str, list] = {}
        for key, value in params.items():
            key = removeprefix(key, set_prefix_).strip()
            values = [v for item in value for v in item.split(':') if len(v) != 0]
            environ[key] = [item for item in values if f'${key}' != item and f'${{{key}}}' != item]
        for key in environ.keys():
            for k, v in environ.items():
                if k == key:
                    continue
                # replace shell variable
                for i, item in enumerate(v):
                    if key not in item:
                        continue
                    environ[k][i] = item.replace(f'${key}', environ[key][0])
                    environ[k][i] = item.replace(f'${{{key}}}', environ[key][0])
        self._environ = {k: v for k, v in environ.items() if len(v) != 0}
        if len(self._environ) == 0:
            print(f'[error] set_env.sh content is invalid, please check.')
            return False
        return True


class DriverLoader(Loader):
    def __init__(self) -> None:
        super().__init__()
        self._version: str = None

        self._install_path: str = None
        self._default_install_path = '/usr/local/Ascend'

    @property
    def version(self) -> str:
        return self._version

    def load(self, data_type: DataType) -> bool:
        if data_type is DataType.VERSION:
            return self._query_version()
        else:
            print(f'[error] invalid data type {data_type}.')
            return False

    def _query_install_path(self) -> bool:
        path_ = '/etc/ascend_install.info'

        if os.path.exists(path_):
            self._install_path = \
                read_param(path_, key='Driver_Install_Path_Param')
        if self._install_path is None:
            print(f'[info] use default driver install path: {self._default_install_path}.')
            self._install_path = self._default_install_path
        if not os.path.exists(os.path.join(self._install_path, 'driver')):
            print(f'[error] driver has not bean installed.')
            return False
        return True

    def _query_version_fun0(self, version_path: str) -> bool:
        '''support the version after 1.81
        '''
        version = read_param(version_path, key='package_version')
        if version is None:
            # only support external version
            return False
        version = version.upper()
        if version not in CANN_AND_TUSCANCY_VERSION_MAP:
            print(f'[error] package_version {version} not support.')
            return False
        version = CANN_AND_TUSCANCY_VERSION_MAP[version][0]
        self._version = version
        return True

    def _query_version_fun1(self, version_path: str) -> bool:
        version = read_param(version_path, key='Innerversion')
        if version is None:
            # only support external version
            return False
        match = re.search('^V[0-9]{3}R[0-9]{3}C[0-9]{2}', version)
        if match is None:
            print(f'[error] Innerversion is invalid in {version_path}.')
            return False
        version = match.group()[-2:]
        version = f'1.{version}' # C80 -> 1.80
        for values in CANN_AND_DRIVER_VERSION_MAP.values():
            for value in values:
                if version.startswith(value):
                    self._version = version
                    return True
        print(f'[error] Innerversion {version} not support.')
        return False

    def _query_version_fun2(self, version_path: str) -> bool:
        '''only support inner package version
        '''
        version = read_param(version_path, key='Version')
        if version is None:
            return False
        for values in CANN_AND_DRIVER_VERSION_MAP.values():
            for value in values:
                if version.startswith(value):
                    self._version = version
                    return True
        print(f'[error] Version {version} not support.')
        return False

    def _query_version(self) -> bool:
        if not self._query_install_path():
            return False

        path_ = os.path.join(self._install_path, 'driver/version.info')
        if not os.path.exists(path_):
            print(f'[error] {path_} not exist, get driver version failed.')
            return False

        if self._query_version_fun0(path_) or \
            self._query_version_fun1(path_) or \
            self._query_version_fun2(path_):
            print(f'[info] driver version: {self._version}')
            return True

        print(f'[error] get driver version failed from {path_}.')
        return False


class Checker(object):
    ''' check the version matching status and environment variables '''
    def __init__(self) -> None:
        self._torch_loader: Loader = TorchLoader()
        self._cann_loader: Loader = CannLoader()
        self._driver_loader: Loader = DriverLoader()

    def check_version(self) -> bool:
        if not self._torch_loader.load(DataType.VERSION) or \
            not self._cann_loader.load(DataType.VERSION) or \
            not self._driver_loader.load(DataType.VERSION):
            return False

        torch_npu_version: str = self._torch_loader.torch_npu_version
        torch_version: str = self._torch_loader.torch_version
        cann_version: str = self._cann_loader.version
        driver_version: str = self._driver_loader.version

        if torch_npu_version is None and torch_version.startswith('1.5'):
            if not torch_version.startswith('1.5'):
                print(f'[error] torch_npu has not bean installed.')
                return False
            return self._check_torch_and_cann_version(torch_version, cann_version) and \
                self._check_cann_and_driver_version(cann_version, driver_version)
        return self._check_torch_npu_and_torch_version(torch_npu_version, torch_version) and \
            self._check_torch_npu_and_cann_version(torch_npu_version, cann_version) and \
            self._check_torch_and_cann_version(torch_version, cann_version) and \
            self._check_cann_and_driver_version(cann_version, driver_version)

    def check_environ(self) -> bool:
        if not self._cann_loader.load(DataType.ENVIRON):
            return False

        loss_environ: str = ''
        environ: Dict[str, list] = self._cann_loader.environ
        for name, value in environ.items():
            if name not in os.environ:
                loss_environ += f'{name}={value}\n'
                continue
            for v in value:
                if v not in os.environ[name]:
                    loss_environ += f'{name}={v}\n'
        loss_environ = loss_environ.rstrip()
        if len(loss_environ) != 0:
            print(f'[error] some environs has not bean setted:')
            print(f'{loss_environ}')
            print('please configure these environment vriables, or ' \
                'execute the command \'source set_env.sh\'')
            return False
        print('[info] environment variable check succeed.')
        return True

    def _get_keys_by_match_value(self, msg: str, dict_: Dict[str, list]) -> list:
        results = []
        max_match_ = ''
        for key, values in dict_.items():
            match_ = ''
            for v in values:
                if msg.startswith(v) and \
                    len(match_) < len(v):
                        match_ = v
            if match_ == '':
                continue
            if len(max_match_) < len(match_):
                max_match_ = match_
                results = [key]
            elif len(max_match_) == len(match_):
                results.append(key)
        return results

    def _get_values_by_match_key(self, msg: str, dict_: Dict[str, list]) -> list:
        max_key_ = ''
        for key in dict_.keys():
            if msg.startswith(key) and \
                len(max_key_) < len(key):
                max_key_ = key
        if max_key_ == '':
            return []
        return dict_[max_key_]

    def _check_torch_npu_and_torch_version(self, torch_npu_version: str, torch_version: str) -> bool:
        return self._check_version(
            ('torch_npu', torch_npu_version),
            ('torch', torch_version),
            TORCH_NPU_AND_TORCH_VERSION_MAP
        )

    def _check_torch_npu_and_cann_version(self, torch_npu_version: str, cann_version: str) -> bool:
        return self._check_version(
            ('cann', cann_version),
            ('torch_npu', torch_npu_version),
            CANN_AND_TORCH_NPU_VERSION_MAP
        )

    def _check_torch_and_cann_version(self, torch_version: str, cann_version: str) -> bool:
        return self._check_version(
            ('cann', cann_version),
            ('torch', torch_version),
            CANN_AND_TORCH_VERSION_MAP
        )

    def _check_cann_and_driver_version(self, cann_version: str, driver_version: str) -> bool:
        return self._check_version(
            ('cann', cann_version),
            ('driver', driver_version),
            CANN_AND_DRIVER_VERSION_MAP
        )

    def _check_version(self, dict_key_: tuple, dict_val_: tuple, dict_: Dict[str, list]) -> bool:
        ''' common version check function
        dict_key_: dict key property (name, value)
        dict_val_: dict value property (name, value)
        '''
        support_k_versions = self._get_keys_by_match_value(dict_val_[1], dict_)
        if len(support_k_versions) == 0:
            support_v_versions = self._get_values_by_match_key(dict_key_[1], dict_)
            if len(support_v_versions) == 0:
                print(f'[error] {dict_key_[0]} and {dict_val_[0]} version not support, please update version match table.')
            else:
                print(f'[error] {dict_key_[0]} version {dict_key_[1]} not match {dict_val_[0]} version {dict_val_[1]}, ' \
                    f'{dict_val_[0]} version {dict_val_[1]} not support, please update version match table, or ' \
                    f'update {dict_val_[0]} version to: {support_v_versions}')
            return False
        for item in support_k_versions:
            if dict_key_[1].startswith(item):
                print(f'[info] {dict_key_[0]} and {dict_val_[0]} version check succeed.')
                return True
        support_v_versions = self._get_values_by_match_key(dict_key_[1], dict_)
        if len(support_v_versions) == 0:
            print(f'[error] {dict_key_[0]} version {dict_key_[1]} not match {dict_val_[0]} version {dict_val_[1]}, ' \
                f'{dict_key_[0]} version {dict_key_[1]} not support, please update version match table, or ' \
                f'update {dict_key_[0]} version to: {support_k_versions}.')
        else:
            print(f'[error] {dict_key_[0]} version {dict_key_[1]} not match {dict_val_[0]} version {dict_val_[1]}, ' \
                f'please update {dict_val_[0]} version to: {support_v_versions}, or ' \
                f'update {dict_key_[0]} version to: {support_k_versions}.')
        return False
