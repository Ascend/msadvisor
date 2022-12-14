import os
import unittest


if __name__ == '__main__':
    dirname = os.path.dirname(os.path.realpath(__file__))
    suite = unittest.defaultTestLoader.discover(dirname)

    print('Total Case: ', suite.countTestCases())

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

