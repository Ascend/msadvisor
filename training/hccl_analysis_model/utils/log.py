import sys
import os

import msadvisor as m


AD_DEBUG = 0
AD_INFO = 1
AD_WARN = 2
AD_ERROR = 3
AD_NULL = 4


def ad_print(log_level, log_info):
        file_name = str(os.path.basename(sys._getframe(1).f_code.co_filename))
        file_line = str(sys._getframe(1).f_lineno)
        log_level_legal = log_level == AD_DEBUG or log_level == AD_INFO or \
                          log_level == AD_WARN or log_level == AD_ERROR or \
                          log_level == AD_NULL
        if log_level_legal:
            full_log_info = '[' + file_name + ':' + file_line + ']' + log_info
            m.utils.print(log_level, full_log_info)
        else:
            log_info_illegal = '[' + file_name + ':' + file_line + ']' + \
                               'The log level is illegal, please confirm the input'
            m.utils.log(AD_WARN, log_info_illegal)


def ad_print_and_log(log_level, log_info):
    file_name = str(os.path.basename(sys._getframe(1).f_code.co_filename))
    file_line = str(sys._getframe(1).f_lineno)
    log_level_legal = log_level == AD_DEBUG or log_level == AD_INFO or \
                      log_level == AD_WARN or log_level == AD_ERROR or \
                      log_level == AD_NULL
    if log_level_legal:
        full_log_info = '[' + file_name + ':' + file_line + ']' + log_info
        m.utils.print_and_log(log_level, full_log_info)
    else:
        log_info_illegal = '[' + file_name + ':' + file_line + ']' + \
                           'The log level is illegal, please confirm the input'
        m.utils.log(AD_WARN, log_info_illegal)


def ad_log(log_level, log_info):
    file_name = str(os.path.basename(sys._getframe(1).f_code.co_filename))
    file_line = str(sys._getframe(1).f_lineno)
    log_level_legal = log_level == AD_DEBUG or log_level == AD_INFO or \
                      log_level == AD_WARN or log_level == AD_ERROR or \
                      log_level == AD_NULL
    if log_level_legal:
        full_log_info = '[' + file_name + ':' + file_line + ']' + log_info
        m.utils.log(log_level, full_log_info)
    else:
        log_info_illegal = '[' + file_name + ':' + file_line + ']' + \
                           'The log level is illegal, please confirm the input'
        m.utils.log(AD_WARN, log_info_illegal)
