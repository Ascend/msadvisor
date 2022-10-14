import unittest
import os
import python


class TestPythonParsers(unittest.TestCase):
    def test_remove_comments(self):
        python_parser_path = os.path.abspath(python.__file__)
        with open(python_parser_path) as f:
            lines = f.readlines()

        origin_source = ''.join(lines)
        output_source = python.remove_comments(origin_source)
        self.assertIsNotNone(output_source)

        python_parser_dir, python_parser_file = os.path.split(python_parser_path)
        output_file = os.path.splitext(python_parser_file)[0] + '_output.py'
        output_path = os.path.join(python_parser_dir, output_file)
        with open(output_path, 'w') as f:
            f.write(output_source)

        # load output source as module, test self-enumerate
        module = __import__('python_output')
        self.assertEqual(output_source, module.remove_comments(origin_source))


if __name__ == '__main__':
    unittest.main()