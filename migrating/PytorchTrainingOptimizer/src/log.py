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
import msadvisor as m

AD_DEBUG = 0
AD_INFO = 1
AD_WARN = 2
AD_ERROR = 3
AD_NULL = 4


def ad_print(log_level, log_info):
    file_name = str(os.path.basename(sys._getframe(1).f_code.co_filename))
    file_line = str(sys._getframe(1).f_lineno)
    if log_level == AD_DEBUG or log_level == AD_INFO or log_level == AD_WARN or log_level == AD_ERROR or log_level == AD_NULL:
        full_log_info = '[' + file_name + ':' + file_line + '] ' + log_info
        m.utils.print(log_level, full_log_info)
    else:
        log_info_illegal = '[' + file_name + ':' + file_line + '] ' + 'The log level is illegal, please confirm the ' \
                                                                      'input! '
        m.utils.log(AD_WARN, log_info_illegal)


def ad_print_and_log(log_level, log_info):
    file_name = str(os.path.basename(sys._getframe(1).f_code.co_filename))
    file_line = str(sys._getframe(1).f_lineno)
    if log_level == AD_DEBUG or log_level == AD_INFO or log_level == AD_WARN or log_level == AD_ERROR or log_level == AD_NULL:
        full_log_info = '[' + file_name + ':' + file_line + '] ' + log_info
        m.utils.print_and_log(log_level, full_log_info)
    else:
        log_info_illegal = '[' + file_name + ':' + file_line + '] ' + 'The log level is illegal, please confirm the ' \
                                                                      'input! '
        m.utils.log(AD_WARN, log_info_illegal)


def ad_log(log_level, log_info):
    file_name = str(os.path.basename(sys._getframe(1).f_code.co_filename))
    file_line = str(sys._getframe(1).f_lineno)
    if log_level == AD_DEBUG or log_level == AD_INFO or log_level == AD_WARN or log_level == AD_ERROR or log_level == AD_NULL:
        full_log_info = '[' + file_name + ':' + file_line + '] ' + log_info
        m.utils.log(log_level, full_log_info)
    else:
        log_info_illegal = '[' + file_name + ':' + file_line + '] ' + 'The log level is illegal, please confirm the ' \
                                                                      'input! '
        m.utils.log(AD_WARN, log_info_illegal)
