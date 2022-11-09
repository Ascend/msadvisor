import unittest
from test_param_type import TestParamType
from test_memory_management import TestMemoryManagement
from test_async_call import TestAsyncCall
from test_zero_copy import TestZeroCopy
from test_context_management import TestContextManagement
from test_v1_api_transfer import TestV1ApiTransfer
from HTMLTestRunner import HTMLTestRunner


if __name__ == '__main__':
    testcases = \
        [ unittest.TestLoader().loadTestsFromTestCase(TestParamType)
        , unittest.TestLoader().loadTestsFromTestCase(TestMemoryManagement)
        , unittest.TestLoader().loadTestsFromTestCase(TestAsyncCall)
        , unittest.TestLoader().loadTestsFromTestCase(TestZeroCopy)
        , unittest.TestLoader().loadTestsFromTestCase(TestContextManagement)
        , unittest.TestLoader().loadTestsFromTestCase(TestV1ApiTransfer)
        ]
    suite = unittest.TestSuite(testcases)
    runner = HTMLTestRunner(log=True, verbosity=2, output='report', title='Api Optimization Test Report', report_name='report',
                            open_in_browser=False)
    runner.run(suite)