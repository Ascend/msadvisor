#!/bin/bash
SHELL_DIR=$(readlink -f $(cd "$(dirname "$0")" || exit;pwd))
CURR_DIR=$(pwd)

ERROR=error
WARN=warning
INFO=info

SUPPORT_DATA_TYPES=('profiling', 'om', 'cce')

function print_log() {
    local content=`echo "$@" | cut -d" " -f2-`
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[$1] [${cur_date}]: $content"
}

# 转换模型，同时生成cce代码
function convert_model2om_with_cce() {
    local _default_op_debug_level="--op_debug_level=4"

    local _op_debug_level_param=$(echo "$model_convert_cmd" | grep -o "\-\-op_debug_level=[0-9]*")
    if [ ! -z ${_op_debug_level_param} ]; then
        model_convert_cmd=${model_convert_cmd/"$_op_debug_level_param"/"$_default_op_debug_level"}
        $model_convert_cmd
    else
        $model_convert_cmd $_default_op_debug_level
    fi
    if [ $? -ne 0 ]; then
        print_log $ERROR "convert model failed."
        exit 1
    fi
}

# 直接转换模型，不考虑是否生成cce代码
function convert_model2om() {
    local _need_cce="$1"

    # 备份当前目录下的kernel_meta数据
    if [ $(ls kernel_meta* 2>/dev/null | wc -l) -ne 0 ]; then
        tar -zcvf bak_kernel_meta.tar.gz kernel_meta* >/dev/null
        rm -rf kernel_meta*
    fi

    if [ $_need_cce -eq 1 ]; then
        convert_model2om_with_cce
    else
        $model_convert_cmd
        if [ $? -ne 0 ]; then
            print_log $ERROR "convert model failed."
            exit 1
        fi
    fi

    # 将om和cce数据软连接到project目录
    if [ -d "$output_path/project" ]; then
        rm -rf "$output_path/project"
    fi
    mkdir -p "$output_path/project"

    local _output_param=$(echo "$model_convert_cmd" | grep -o "\-\-output=[0-9a-zA-Z\/_-]*")
    local _model_path=$(readlink -f $(dirname $(echo "$_output_param" | cut -d"=" -f2-)))
    local _model_name=$(basename $(echo "$_output_param" | cut -d"=" -f2-))
    ln -s "${_model_path}/${_model_name}.om" "$output_path/project/${_model_name}.om"

    if [ $_need_cce -eq 1 -o -d "$CURR_DIR"/kernel_meta*/kernel_meta ]; then
        ln -s "$CURR_DIR"/kernel_meta*/kernel_meta "$output_path/project/kernel_meta"
    fi

    # 还原kernel_meta数据
    if [ -f bak_kernel_meta.tar.gz ]; then
        tar -zxvf bak_kernel_meta.tar.gz
        rm -f bak_kernel_meta.tar.gz
    fi
}

# 采集profiling数据，覆盖原有的profiling数据
function collect_profiling() {
    local metrics="$@"

    local _atc_path=$(which atc)
    if [ -z "$_atc_path" ]; then
        print_log $WARN "atc not exist."
    fi
    local _prof_dir=$(readlink -f "$(dirname $_atc_path)/../../tools/profiler/")
    if [ ! -d "$_prof_dir" ]; then
        print_log $ERROR "profiler not exist."
        exit 1
    fi

    local _msprof="$_prof_dir/bin/msprof"
    if [ ! -f "$_msprof" ]; then
        print_log $ERROR "msprof not exist."
        exit 1
    fi

    if [ -d "${output_path}/profiling" ]; then
        rm -rf "${output_path}/profiling"
    fi
    for metric in ${metrics[@]}; do
        $_msprof --output="${output_path}"/profiling --application="${app_cmd}" --aicpu=on --ai-core=on --aic-mode=task-based --ascendcl=on --runtime-api=on --aic-metrics=$metric
        if [ $? -ne 0 ]; then
            print_log $ERROR "msprof failed."
            exit 1
        fi
    done
}

function check_profiling_data() {
    local _prof_metrics="$@"

    # check profiling
    local _prof_path="${output_path}/profiling"
    local _prof_dirs=$(ls "$_prof_path")
        if [ -z "${_prof_dirs}" ]; then
            print_log $ERROR "profiling data not fullly."
            return 1
        fi
        for metric in ${_prof_metrics[@]}; do
            local _match_flag=0
            for dir in ${_prof_dirs[@]}; do
                [ ! -f "$_prof_path/$dir"/device_*/sample.json ] && continue
                cat "$_prof_path/$dir"/device_*/sample.json | grep -o "\"ai_core_metrics\":\"$metric\""
                [ $? -eq 0 ] && _match_flag=1 && break
            done
            if [ $_match_flag -eq 0 ]; then
                print_log $ERROR "profiling data not exist for metric=$metric."
                return 1
            fi
        done
    return 0
}

function check_model() {
    local _need_cce=$1

    # check .om and .cce
        model_path=$(ls "${output_path}"/project/*.om 2>/dev/null)
        if [ -z "$model_path" ]; then
            print_log $ERROR "model not exist, please specify model convert command."
            exit 1
        fi
    if [ ${_need_cce} -eq 1 ]; then
        cce_dir=$(ls "${output_path}/project/kernel_meta" 2>/dev/null)
        if [ -z "$cce_dir" ]; then
            print_log $ERROR "cce directory not exist."
            exit 1
        fi
    fi
}

function collect_inference_data() {
    if [ -z "$model_convert_cmd" ]; then
        # check .om and .cce
        check_model 1
	[ $? -ne 0 ] && exit 1
    else
        convert_model2om 1
    fi

    local _prof_metrics=(PipeUtilization Memory MemoryUB MemoryL0)
    if [ -z "$app_cmd" ]; then
        # check profiling
        check_profiling_data ${_prof_metrics[@]}
        [ $? -ne 0 ] && exit 1
    else
        collect_profiling ${_prof_metrics[@]}
    fi
}

function collect_train_data() {
    local _prof_metrics=(PipeUtilization)
    if [ -z "$app_cmd" ]; then
        # check profiling
        check_profiling_data ${_prof_metrics[@]}
        [ $? -ne 0 ] && exit 1
    else
        collect_profiling ${_prof_metrics[@]}
    fi
}

function collect_operator_data() {
    print_log $INFO "Not support simulation dump data collect."
}

function parse_model_json() {
    local _model_json_paths="$@"

    python3 - "$_model_json_paths" <<-END
import json
import os
import sys

SUPPORT_DATA_TYPES = ['profiling', 'om', 'cce']

def check_data_type(data_types: str):
    """
    check data type is valid
    """
    if data_types is None:
        return False
    data_type_list = data_types.split(',')
    for data_type in data_type_list:
        match_flag = False
        for support_type in SUPPORT_DATA_TYPES:
            if data_type == support_type:
                match_flag = True
                break
        if not match_flag:
            return False
    return True

def parse_json(jsonstr: str):
    """
    parse model json file to get data_type
    """
    jsons = jsonstr.split(',')
    if len(jsons) == 0:
        print('[error] No model json.')
        return None
    data_types = set()
    for json_path in jsons:
        if not os.path.isfile(json_path):
            print('[error] {} file not exist.'.format(json_path))
            return None
        data = ''
        with open(json_path, 'r') as f:
            data = json.load(f)
        if (len(data) is 0) or (len(data.get('model_list')) is 0):
            # no model_list
            print("[error] No model_list in model json.")
            return None
        for model in data['model_list']:
            if not check_data_type(model.get('data_type')):
                continue
            data_types = data_types.union(model['data_type'].split(','))
    result = ''
    for data_type in data_types:
        if len(result) is 0:
            result = data_type
            continue
        result += ' ' + data_type
    return result

if __name__ == "__main__":
    """
    parse model json, and get requirement data type
    """
    data_types = parse_json(sys.argv[1])
    if (data_types is None) or (len(data_types) is 0):
        print("[error] No requirement data type.")
    else:
        print("Data type: {}".format(data_types))

END
}

function collect_echosystem_data() {
    # parse all *.json, read data_type
    local _parser_py="$SHELL_DIR/parser.py"
    local _parse_result=$(parse_model_json "$conf_path")
    local _correct_result=$(echo "$_parse_result" | grep -o "Data type: *")
    if [ -z "$_correct_result" ]; then
        print_log $ERROR "$_parse_result"
        exit 1
    fi
    local _data_types=$(echo "$_parse_result" | cut -d" " -f3-)
    if [ -z "$_data_types" ]; then
        print_log $ERROR "No requirement data type."
        exit 1
    fi

    # support data type: 'profiling', 'om', 'cce'
    declare -A _type_map
    for data_type in ${SUPPORT_DATA_TYPES[@]}; do
        _type_map[$data_type]=0
    done
    for data_type in ${_data_types[@]}; do
        _type_map[$data_type]=1
    done
    if [ ${_type_map['cce']} == 1 ]; then
        if [ -z "$model_convert_cmd" ]; then
            # check .om and .cce
            check_model 1
            [ $? -ne 0 ] && exit 1	
        else
            convert_model2om 1
        fi
    elif [ ${_type_map['om']} == 1 ]; then
        if [ -z "$model_convert_cmd" ]; then
            # check .om and .cce
            check_model 0
            [ $? -ne 0 ] && exit 1
        else
            convert_model2om 0
        fi
    fi
    if [ ${_type_map['profiling']} == 1 ]; then
        local _prof_metrics=(PipeUtilization Memory MemoryUB MemoryL0)
        if [ -z "$app_cmd" ]; then
            # check profiling
            check_profiling_data ${_prof_metrics[@]}
            [ $? -ne 0 ] && exit 1
        else
	    collect_profiling ${_prof_metrics[@]}
        fi
    fi
}

function help_info() {
    echo 'Usage:
    bash msadvisor.sh --type=0 --model_convert="atc --framework=5 --model=model/deit_bs8.onnx --output=model/deit_bs8 --input_format=NCHW --input_shape=image:8,3,224,224 --soc_version=Ascend310 --op_debug_level=4" --application="model/benchmark.aarch64 -batch_size=8 -device_id=0 -om_path=/home/zjj/advisor/msadvisor/data/model/deit_bs8.om -round=50" --data=./data_out

Options:
    --type                   <Optional> scene type, support 0:inference; 1:train; 2:operator; 3:echosystem, default 0:inference
    --model_convert          <Optional> model(include caffe/mindspore/tensorflow/onnx) convert command
    --application            <Optional> Specify application path, considering the risk of privilege escalation, please pay attention to
                                        the group of the application and confirm whether it is the same as the user currently
    --conf                   <Optional> Specify model config path, only support .json format
    --data                              Specify data output path
'
}

function run_advisor() {
    if [ -z $(which msadvisor) ]; then
        print_log $ERROR "msadvisor command not exist, please source set_env.sh firstly."
        exit 1
    fi

    if [ ! -z $conf_path ]; then
        msadvisor -c $conf_path -d "$output_path"
    else
        if [ $scene_type -eq 0 -o $scene_type -eq 1 ]; then
            msadvisor -c model -d "$output_path"
        else
            msadvisor -c op -d "$output_path"
        fi
    fi
}

function collect_data() {
    if [ -z "$output_path" ]; then
        print_log $ERROR "output path is empty, please specify output path."
	help_info
	exit 1
    fi
    if [ ! -d "$output_path" ]; then
        print_log $INFO "output path not exist, auto create."
	mkdir -p $output_path
	[ $? -ne 0 ] && exit 1
    fi
    # support scene type: 0: inference; 1: train; 2: operator; 3: echosystem
    if [ $scene_type -eq 0 ]; then
        collect_inference_data && return
    fi

    if [ $scene_type -eq 1 ]; then
        collect_train_data && return
    fi

    if [ $scene_type -eq 2 ]; then
        collect_operator_data && return
    fi

    if [ $scene_type -eq 3 ]; then
        collect_echosystem_data && return
    fi
}

function main() {
    collect_data
    run_advisor
}

declare -A support_scene_type=([0]='inference', [1]='train', [2]='operator', [3]='echosystem')
# 场景类型
scene_type=0
# 执行指令
model_convert_cmd=""
app_cmd=""
# 路径
output_path=""
conf_path=""

while true; do
    case "$1" in
    --type=*)
        scene_type=$(echo "$1" | cut -d"=" -f2)
        if [ -z ${support_scene_type[$scene_type]} ]; then
            print_log $ERROR "scene type is invalid."
            help_info
            exit 1
        fi
        shift
        ;;
    --model_convert=*)
        model_convert_cmd=$(echo "$1" | cut -d"=" -f2-)
        bin=$(echo "$model_convert_cmd" | cut -d" " -f1)
        if [ -z $(which "$bin") -a ! -f "$bin" ]; then
            print_log $ERROR "$bin not exist."
            exit 1
        fi
        shift
        ;;
    --application=*)
        app_cmd=$(echo "$1" | cut -d"=" -f2-)
        bin=$(echo "$app_cmd" | cut -d" " -f1)
        if [ -z $(which "$bin") -a ! -f "$bin" ]; then
            print_log $ERROR "$bin not exist."
            exit 1
        fi
        shift
        ;;
    --conf=*)
        conf_path=$(echo "$1" | cut -d"=" -f2)
        if [ -z "$conf_path" ]; then
            print_log $WARN "conf path is empty."
        fi
        shift
        ;;
    --data=*)
        output_path=$(echo "$1" | cut -d"=" -f2-)
        if [ ! -d "$output_path" ]; then
            mkdir -p "$output_path" 2>&1 >/dev/null
            if [ $? -ne 0 ]; then
                print_log $ERROR "create $output_path failed."
                exit 1
            fi
        fi
	output_path=$(readlink -f "$output_path")
        shift
        ;;
    --help)
        help_info
        exit 0
        ;;
    *)
        if [ ! -z "$1" ]; then
            print_log $ERROR "Unrecognized parameters: $1."
            help_info
            exit 1
        fi
        break
	;;
    esac
done

main

exit 0
