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
import unittest
import mock

from src.prev_check import Checker
from src.prev_check import CannLoader
from src.prev_check import TorchLoader
from src.prev_check import DriverLoader


class TestChecker(unittest.TestCase):

    def setUp(self) -> None:
        self.checker = Checker()

    def test_check_torch_npu_and_torch_version(self):
        params = [
            ('1.8.1rc1', '1.8.1+ascend.rc1', True),
            ('1.8.1rc2', '1.8.1+ascend.rc2', True),
            ('1.8.1rc3', '1.8.1+ascend.rc3', True),
            ('1.8.1.xx', '1.8.0a0', True),
            ('1.11.0rc1', '1.11.0a0', True),
            ('1.11.0rc2', '1.11.0a0', True),
            ('1.8.1.xx', '1.8.1+ascend.rc3', False)
        ]
        for torch_npu_version, torch_version, result in params:
            self.assertEqual(result, self.checker._check_torch_npu_and_torch_version(torch_npu_version, torch_version))

    def test_check_torch_npu_and_cann_version(self):
        params = [
            ('1.8.1rc1', '5.1.RC1', True),
            ('1.8.1rc2', '5.1.RC2', True),
            ('1.8.1rc3', '5.1.RC3', False),
            ('1.8.1rc3', '6.0.RC1', True),
            ('1.8.1.xx', '6.0.0', True),
            ('1.8.1.xx', '6.0.1', True),
        ]
        for torch_npu_version, cann_version, result in params:
            self.assertEqual(result, self.checker._check_torch_npu_and_cann_version(torch_npu_version, cann_version))

    def test_check_torch_and_cann_version(self):
        params = [
            ('1.8.0a0', '5.1.RC1', False),
            ('1.8.0a0', '5.1.RC2', True),
            ('1.8.0a0', '6.0.RC1', True),
            ('1.8.0a0', '5.0.RC3', False),
            ('1.8.0a0', '6.0.0', True),
            ('1.8.0a0', '6.0.1', True),
            ('1.5.0+ascend.post4', '5.0.4', True),
            ('1.5.0+ascend.post4', '5.0.5', True),
            ('1.5.0+ascend.post1', '5.0.2', False),
        ]
        for torch_version, cann_version, result in params:
            self.assertEqual(result, self.checker._check_torch_and_cann_version(torch_version, cann_version))

    def test_check_cann_and_driver_version(self):
        params = [
            ('6.0.0', '1.80', False),
            ('6.0.0', '1.81', True),
            ('6.0.0', '1.82', True),
            ('6.0.0', '1.83', True),
            ('6.0.0', '1.84', True),
            ('6.0.0', '1.85', True),
            ('6.0.0', '1.86', False),
        ]
        for cann_version, driver_version, result in params:
            self.assertEqual(result, self.checker._check_cann_and_driver_version(cann_version, driver_version))

    def test_check_version(self):
        params = [
            ('1.8.1.xx', '1.8.0a0', '6.0.0', '1.84', True),
            ('1.8.1.xx', '1.8.1+ascend.rc3', '6.0.0', '1.84', False),
            ('1.8.1rc3', '1.8.0a0', '6.0.0', '1.84', False),
            ('1.8.1.xx', '1.8.0a0', '6.0.RC1', '1.84', False),
            ('1.8.1.xx', '1.8.0a0', '6.0.0', '1.80', False),
            ('1.8.1.xx', '1.8.xa0', '6.0.0', '1.84', False),
            ('1.7.1.xx', '1.8.0a0', '6.0.0', '1.84', False),
            ('1.8.1.xx', '1.8.0a0', '6.x.0', '1.84', False),
            ('1.8.1.xx', '1.8.0a0', '6.0.0', '1.8x', False)
        ]
        TorchLoader.load = mock.Mock(return_value = True)
        CannLoader.load = mock.Mock(return_value = True)
        DriverLoader.load = mock.Mock(return_value = True)
        for param0, param1, param2, param3, param4 in params:
            TorchLoader.torch_npu_version = mock.PropertyMock(return_value = param0)
            TorchLoader.torch_version = mock.PropertyMock(return_value = param1)
            CannLoader.version = mock.PropertyMock(return_value = param2)
            DriverLoader.version = mock.PropertyMock(return_value = param3)

            self.assertEqual(param4, self.checker.check_version())

    def test_check_environ(self):
        CannLoader.load = mock.Mock(return_value = True)
        CannLoader.environ = mock.PropertyMock(return_value = {'xx': ['yy', 'zz']})

        os.environ.setdefault('xx', 'yy')
        self.assertFalse(self.checker.check_environ())

        os.environ.update({'xx': 'yy:zz'})
        self.assertTrue(self.checker.check_environ())

        os.environ.update({'xx': 'zz'})
        self.assertFalse(self.checker.check_environ())

        os.environ.update({'xx': 'tt'})
        self.assertFalse(self.checker.check_environ())

if __name__ == "__main__":
    unittest.main()
