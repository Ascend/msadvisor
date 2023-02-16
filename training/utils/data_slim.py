# Container data reduction

import os
import glob

try:
    import moxing as mox
except:
    print("There is not moxing")

PROFILER_DATA_DICT = {
    "HCCL": [
        "PROF_*",
        "hccl_info_*",
        "ascend_timeline_display_*.json",
        "step_trace_raw_*_detail_time.csv"
    ]
}


# data 可选项有 ALL、HCCL
def collect_data(src_url, dst_url, data="ALL"):
    """Collect all data to OBS or filter redundant data"""
    if data == "All":
        mox.file.copy_parallel(src_url, dst_url)
    else:
        src_url = os.path.join(src_url, "profiler")
        dst_url = os.path.join(dst_url, "profiler")

        data_list = PROFILER_DATA_DICT.get(data)
        for name in data_list:
            file_names = glob.glob(os.path.join(src_url, name))
            for file_name in file_names:
                if os.path.isdir(file_name):
                    dir_basename = os.path.basename(file_name)
                    if "PROF_" in dir_basename:
                        slim_pro_data(file_name, dst_url)
                    else:
                        dst_path = os.path.join(dst_url, dir_basename)
                        mox.file.copy_parallel(file_name, dst_path)
                elif os.path.isfile(file_name):
                    dir_basename = os.path.basename(file_name)
                    mox.file.copy(file_name, os.path.join(dst_url, dir_basename))
                else:
                    continue


def slim_pro_data(pro_dir, dst_url):
    """filter redundant data in PROF_*"""
    pro_dir_name = os.path.basename(pro_dir)
    for root, dir_name, files in os.walk(pro_dir):
        root_name = os.path.basename(root)
        if 'device_' in root_name:
            for file in files:
                srt_path = os.path.join(root, file)
                dst_path = os.path.join(dst_url, pro_dir_name, root_name, file)
                mox.file.copy(srt_path, dst_path)

            for dir_ in dir_name:
                data_files = os.listdir(os.path.join(root, dir_))
                for file in data_files:
                    if 'HCCL.' not in file and 'hwts.data.join' not in file:
                        srt_path = os.path.join(root, dir_, file)
                        dst_path = os.path.join(dst_url, pro_dir_name, root_name, dir_, file)
                        mox.file.copy(srt_path, dst_path)