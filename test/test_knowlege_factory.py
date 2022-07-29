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

from src.register import Register
from src.pattern import KnowlegeFactory


class TestKnowlege(unittest.TestCase):
    def setUp(self) -> None:
        sys.path.append("..")
        os.chdir("..")
        register = Register("knowleges")
        register.import_modules()

    def tearDown(self) -> None:
        pass

    def test_knowlege_pattern(self):
        for name, knowlege in KnowlegeFactory.get_knowlege_pool().items():
            knowlege.pattern()

    def test_knowlege_apply(self):
        for name, knowlege in KnowlegeFactory.get_knowlege_pool().items():
            ret = knowlege.apply(None)
            self.assertEqual(True, ret)


if __name__ == "__main__":
    unittest.main(verbosity=2)
