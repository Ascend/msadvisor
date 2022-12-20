import json
import os
import pytest
import sys
sys.path.append("./../")
from model import evaluate


class TestClass:
    @classmethod
    def setup_class(cls):
        """
        class level setup_class
        """
        cls.init(TestClass)

    @classmethod
    def teardown_class(cls):
        print('\n ---class level teardown_class')

    def init(self):
        # data的路径，可能是公用
        self.data_path = "./data"
        self.PROF_list = self.Search_PROF_file_name(self.data_path)

    # 搜索所有以‘PROF_’开头的文件夹的绝对路径
    def Search_PROF_file_name(abs_path):
        list = []
        pre_data_path = abs_path + '/profiling/'
        # 切换工作目录到abspath指定目录,也就是当前工作目录
        # os.chdir(pre_data_path)
        # 列出本目录下的文件
        L = os.listdir(pre_data_path)
        for v in L:
            if os.path.isdir(pre_data_path + v) and ('PROF_' in v):  # 将本目录下的符合条件的文件夹名字输出
                PROF_data_path = v  # 格式：xxx/xxx/data/profiling/PROF_xxxx
                list.append(PROF_data_path)
        return list

    def change(self, path, pathtest):
        f = open(path, "r", encoding='utf-8')
        data = f.read()
        ft = open(pathtest, "r", encoding='utf-8')
        datat = ft.read()
        f = open(path, "w", encoding='utf-8')
        ft = open(pathtest, "w", encoding='utf-8')
        f.truncate()
        ft.truncate()
        f.write(datat)
        ft.write(data)
        f.close()
        ft.close()

    def test_normal_well_optimited(self):
        """
        场景一:算子已经被很好地优化了
        """
        for i in range(5):
            src = './data/profiling/' + self.PROF_list[i] + '/device_0/summary/op_summary_0_1_1.csv'
            dst = './data/profiling/' + self.PROF_list[i] + '/device_0/summary/test1_op_summary_0_1_1.csv'
            self.change(src, dst)
        ret = evaluate(self.data_path, '{"device_id": 0, "Topk": 5}')
        print('test_case_1：没有算子搬运瓶颈时，结果为:\n' + str(ret))
        print()
        ret = json.loads(ret)
        assert ret['errorCode'] == '1'
        for i in range(5):
            src = './data/profiling/' + self.PROF_list[i] + '/device_0/summary/op_summary_0_1_1.csv'
            dst = './data/profiling/' + self.PROF_list[i] + '/device_0/summary/test1_op_summary_0_1_1.csv'
            self.change(src, dst)

    def test_abnormal_no_profiling(self):
        """
        场景二:用户未放入Profiling数据文件
        """
        os.rename('./data/profiling', "./data/profiling1")
        os.mkdir('./data/profiling')
        ret = evaluate(self.data_path, '{"device_id": 0, "Topk": 5}')
        print('test_case_2：未输入Profiling数据文件时，结果为:\n' + str(ret))
        print()
        ret = json.loads(ret)
        assert ret["errorCode"] == '1'
        os.rmdir('./data/profiling')
        os.rename('./data/profiling1', "./data/profiling")

    def test_abnormal_lack_profiling(self):
        """
        场景三:缺少某项指标采集的Profiling文件
        """
        os.rename('./data/profiling/PROF_000001_20221111164735818_AENQMAPQBDAIEKCA', "./data/profiling/1")
        ret = evaluate(self.data_path, '{"device_id": 0, "Topk": 5}')
        print('test_case_3：缺少某项指标采集的Profiling文件时，结果为:\n' + str(ret))
        print()
        ret = json.loads(ret)
        assert ret['errorCode'] == '1'
        os.rename('./data/profiling/1', "./data/profiling/PROF_000001_20221111164735818_AENQMAPQBDAIEKCA")

    def test_normal_bottleneck_pathway(self):
        """
        场景四：实现场景--识别数据访问瓶颈通路
        """
        for i in range(5):
            src = './data/profiling/' + self.PROF_list[i] + '/device_0/summary/op_summary_0_1_1.csv'
            dst = './data/profiling/' + self.PROF_list[i] + '/device_0/summary/test4_op_summary_0_1_1.csv'
            self.change(src, dst)
        ret = evaluate(self.data_path, '{"device_id": 0, "Topk": 5}')
        print('test_case_4：实现场景--识别数据访问瓶颈通路，结果为:\n' + str(ret))
        print()
        ret = json.loads(ret)
        assert ret['errorCode'] == '0'
        for i in range(5):
            src = './data/profiling/' + self.PROF_list[i] + '/device_0/summary/op_summary_0_1_1.csv'
            dst = './data/profiling/' + self.PROF_list[i] + '/device_0/summary/test4_op_summary_0_1_1.csv'
            self.change(src, dst)

    def test_normal_repeat_mig_and_bottleneck_pipeline(self):
        """
        场景五：实现场景--识别重复搬运
        场景六：实现场景—识别流水内部发生不合理的阻塞
        同一用例，实现两个场景
        """
        for i in range(5):
            src = './data/profiling/' + self.PROF_list[i] + '/device_0/summary/op_summary_0_1_1.csv'
            dst = './data/profiling/' + self.PROF_list[i] + '/device_0/summary/test5_op_summary_0_1_1.csv'
            self.change(src, dst)
        ret = evaluate(self.data_path, '{"device_id": 0, "Topk": 5}')
        print('test_case_5：实现场景--识别重复搬运 和 实现场景--识别流水内部发生不合理的阻塞，结果为:\n' + str(ret))
        print()
        ret = json.loads(ret)
        assert ret['errorCode'] == '0'
        for i in range(5):
            src = './data/profiling/' + self.PROF_list[i] + '/device_0/summary/op_summary_0_1_1.csv'
            dst = './data/profiling/' + self.PROF_list[i] + '/device_0/summary/test5_op_summary_0_1_1.csv'
            self.change(src, dst)

    def test_normal_migration_granularity(self):
        """
        # 场景七：实现场景—识别数据搬运粒度
        """
        for i in range(5):
            src = './data/profiling/' + self.PROF_list[i] + '/device_0/summary/op_summary_0_1_1.csv'
            dst = './data/profiling/' + self.PROF_list[i] + '/device_0/summary/test7_op_summary_0_1_1.csv'
            self.change(src, dst)
        ret = evaluate(self.data_path, '{"device_id": 0, "Topk": 5}')
        print('test_case_6：实现场景—识别数据搬运粒度，结果为:\n' + str(ret))
        print()
        ret = json.loads(ret)
        assert ret['errorCode'] == '0'
        for i in range(5):
            src = './data/profiling/' + self.PROF_list[i] + '/device_0/summary/op_summary_0_1_1.csv'
            dst = './data/profiling/' + self.PROF_list[i] + '/device_0/summary/test7_op_summary_0_1_1.csv'
            self.change(src, dst)


if __name__ == '__main__':
    pytest.main(['test.py', '-vs'])
