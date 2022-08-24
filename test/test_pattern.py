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

import unittest

from auto_optimizer.pattern.pattern import MATCH_PATTERN
from auto_optimizer.pattern.pattern import Pattern


class TestPattern(unittest.TestCase):
    
    def test_add_node_func_0(self):
        pattern = Pattern()
        pattern.add_node('Conv_0', ['Conv'], None)
        pattern.add_node('Conv_1', ['Conv'], None)

        self.assertEqual(len(pattern.node_dict), 2)
        self.assertTrue(pattern.node_dict.get('Conv_0') is not None)
        self.assertTrue(pattern.node_dict.get('Conv_1') is not None)

    def test_add_node_func_1(self):
        pattern = Pattern()

        try:
            pattern.add_node('Conv', ['Conv'], None)
            pattern.add_node('Conv', ['Conv'], None)
        except RuntimeError as e:
            pass

        self.assertEqual(len(pattern.node_dict), 1)
        self.assertTrue(pattern.node_dict.get('Conv') is not None)

    def test_node_cann_match_more_func(self):
        pattern = Pattern() \
            .add_node('Conv', ['Conv'], None) \
            .add_node('Relu', ['Relu'], None) \
            .add_edge('Conv', 'Relu') \
            .set_input('Conv') \
            .set_output('Relu') \
            .set_node_loop('Conv', MATCH_PATTERN.MATCH_ONECE_OR_MORE) \
            .set_node_loop('Relu', MATCH_PATTERN.MATCH_ZERO_OR_MORE)

        self.assertTrue(pattern.node_cann_match_more('Conv'))
        self.assertTrue(pattern.node_cann_match_more('Relu'))

    def test_node_cann_match_zero_func(self):
        pattern = Pattern() \
            .add_node('Conv', ['Conv'], None) \
            .add_node('Relu', ['Relu'], None) \
            .add_edge('Conv', 'Relu') \
            .set_input('Conv') \
            .set_output('Relu') \
            .set_node_loop('Conv', MATCH_PATTERN.MATCH_ONECE_OR_MORE) \
            .set_node_loop('Relu', MATCH_PATTERN.MATCH_ZERO_OR_MORE)

        self.assertFalse(pattern.node_cann_match_zero('Conv'))
        self.assertTrue(pattern.node_cann_match_zero('Relu'))

    def test_cann_match_more_func(self):
        pattern = Pattern() \
            .add_node('Conv', ['Conv'], None) \
            .add_node('Relu', ['Relu'], None) \
            .add_edge('Conv', 'Relu') \
            .set_input('Conv') \
            .set_output('Relu') \
            .set_loop(MATCH_PATTERN.MATCH_ONECE_OR_MORE)

        self.assertTrue(pattern.can_match_more())


if __name__ == "__main__":
    unittest.main()
