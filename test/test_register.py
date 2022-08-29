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
import sys
import unittest

from auto_optimizer.common import Register


class TestRegister(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_register(self):
        register = Register(os.path.join(os.getcwd(), "auto_optimizer", "pattern", "knowledges"))
        ret = register.import_modules()
        self.assertEqual(None, ret)

    def test_register_invalid_path(self):
        register = Register(os.path.join(os.getcwd(), "hello"))
        ret = register.import_modules()
        self.assertEqual(None, ret)


def test_suite():
    suite = unittest.TestSuite()

    suite.addTest(TestRegister("test_register"))
    suite.addTest(TestRegister("test_register_invalid_path"))

    return suite


if __name__ == "__main__":
    unittest.main(defaultTest=test_suite)
