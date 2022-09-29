#!/bin/bash
SHELL_DIR=$(readlink -f $(cd "$(dirname "$0")" || exit;pwd))

ERROR=error
WARN=warning
INFO=info

SUPPORT_DATA_TYPES=('profiling', 'om', 'cce')
ADVISOR_CONF_PATH=$SHELL_DIR/advisor.conf

function print_log() {
    local content=`echo "$@" | cut -d" " -f2-`
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[$1] [${cur_date}]: $content"
}

# 转换模型
function convert_model2om() {
    local _out_path=$1
    local _origin_model_path="$model_path"
    local _origin_model_name=$(basename "$_origin_model_path")

    get_model_convert_cmd

    if [ -d "$_out_path/project" ]; then
        rm -rf "$_out_path/project"
    fi
    mkdir -p "$_out_path/project"
    if [ ! -d "$_out_path/project" ]; then
        print_log $ERROR "project path not exist, path: $_out_path/project"
	exit 1
    fi
    cd "$_out_path/project"
    rm -rf "$_out_path"/project/kernel_meta* 2>&1 >/dev/null
    local _prefix=$(echo $_origin_model_name | cut -d"." -f1)
    model_convert_cmd="$model_convert_cmd --output=$_out_path/project/$_prefix"

    $model_convert_cmd --op_debug_level=4
    if [ $? -ne 0 ]; then
        print_log $ERROR "Convert model failed, please check."
        exit 1
    fi

    # 将onnx文件路径软连接到project目录
    local _origin_relative_model_path=$(realpath --relative-to=$(pwd) "$_origin_model_path")
    ln -sf $_origin_relative_model_path $_out_path/project/$_origin_model_name

    cd -
}

# 获取模型框架类型
function get_framework() {
    local _model_path=$1
    local _model_suffix="${_model_path##*.}"

    [ $_model_suffix == "prototxt" ] && return 0
    [ $_model_suffix == "pb" ] && return 3
    [ $_model_suffix == "onnx" ] && return 5

    print_log $ERROR "Model type is not support."
    exit 1
}

# 获取芯片类型
function get_soc_version() {
    echo $(python3 -c "import acl; print(acl.get_soc_name())" 2>/dev/null)
}

# 获取模型转换指令
function get_model_convert_cmd() {
    # get framework
    get_framework $model_path
    local _framework=$?

    # get soc version
    local _soc_version=$(get_soc_version)
    if [ -z "$_soc_version" ]; then
        print_log $ERROR "Get soc version failed, please set cann environment variable."
        exit 1
    fi
    local _om_path=$(basename "$model_path" | cut -d"." -f1)

    # construct model convert command
    model_convert_cmd="atc --framework=$_framework --model=$model_path --soc_version=$_soc_version"

    if [ ! -z "$weight_path" ]; then
        model_convert_cmd="$model_convert_cmd --weight=$weight_path"
    fi
    if [ ! -z $input_shape ]; then
        model_convert_cmd="$model_convert_cmd --input_shape=$input_shape"
    fi
    if [ ! -z $input_format ]; then
        model_convert_cmd="$model_convert_cmd --input_format=$input_format"
    fi
}

# 采集profiling数据
function collect_profiling() {
    local _out_path=$1
    local metrics=$(echo "$@" | cut -d" " -f2-)

    local _cann_path=$(get_cann_path)
    if [ -z "$_cann_path" -o ! -d "$_cann_path" ]; then
        print_log $ERROR "Get cann path failed."
        exit 1
    fi
    local _prof_bin="$_cann_path/ascend-toolkit/latest/tools/profiler/bin/msprof"
    if [ ! -f "$_prof_bin" ]; then
        print_log $ERROR "Prof bin not exist."
        exit 1
    fi

    get_app_command "$_out_path"

    local _prof_cmd="$_prof_bin --output=${_out_path}/profiling"
    _prof_cmd="$_prof_cmd --aicpu=on --ai-core=on --aic-mode=task-based --ascendcl=on --runtime-api=on"

    local _soc_version=$(get_soc_version)
    if [ ! -z "$_soc_version" -a ! "$_soc_version" == 'Ascend310' ]; then
        _prof_cmd="$_prof_cmd --l2=on"
    fi

    if [ -d "${_out_path}/profiling" ]; then
        rm -rf "${_out_path}/profiling"
    fi
    mkdir "${_out_path}/profiling"
    if [ ! -d "${_out_path}/profiling" ]; then
        print_log $ERROR "profiling path not exist, path: ${_out_path}/profiling"
        exit 1
    fi
    for metric in ${metrics[@]}; do
        print_log $INFO "Start to collect $metric profiling data."
        _prof_cmd="$_prof_cmd --aic-metrics=$metric"
        $_prof_cmd --application="${app_cmd}" 2>&1 >/dev/null
        if [ $? -ne 0 ]; then
            print_log $ERROR "Collect $metric profiling data failed."
            exit 1
        fi
        print_log $INFO "Collect $metric profiling data succeed."
    done
}

# 获取模型batch
function get_model_batch_size() {
    if [ ! -z $input_shape ]; then
        local _shape=$(echo $input_shape | cut -d";" -f1 | cut -d":" -f2)
    else
        local _shape=$(python3 -c "import sys; sys.path.append('$SHELL_DIR');\
            import common.common as common;\
            print(common.get_model_shape('$model_path'))")
        if [ -z $_shape ]; then
            print_log $ERROR "Get input shape failed, please specify input shape by --input_shape="
            exit 1
        fi
    fi
    local _batch_size=$(echo $_shape | cut -d"," -f1)
    echo $_batch_size
}

# 获取app指令
function get_app_command() {
    local _out_path=$1
    local _origin_model_path=$model_path
    local _origin_model_name=$(basename "$_origin_model_path")

    # benchmark.aarch64 -batch_size=1 -om_path=./alexnet_tf_aipp.om -round=10 -device_id=0
    local _om_path="$_out_path/project"/$(echo $_origin_model_name | cut -d"." -f1)".om"
    local _batch_size=$(get_model_batch_size)
    local _benchmark_path="$SHELL_DIR/benchmark.$(arch)"
    if [ ! -f "${_benchmark_path}" ]; then
        print_log $ERROR "benchmark not exist, please check."
        exit 1
    fi
    app_cmd="$_benchmark_path -batch_size=$_batch_size -om_path=$_om_path -round=10 -device_id=0"
}

# 获取cann路径
function get_cann_path() {
    if [ $(id -u) -eq 0 ]; then
        if [ -f /etc/Ascend/ascend_cann_install.info ]; then
            cann_path=$(cat /etc/Ascend/ascend_cann_install.info | cut -d"=" -f2)
        fi
    else
        if [ -f ~/Ascend/ascend_cann_install.info ]; then
            cann_path=$(cat ~/Ascend/ascend_cann_install.info | cut -d"=" -f2)
        fi
    fi
    echo $cann_path
}

# 采集推理数据，包括om、cce、profiling
function collect_inference_data() {
    [ -z "$model_path" ] && return
    model_path=$(realpath "$model_path")

    local _origin_model_path="$model_path"
    local _origin_model_name=$(basename "$_origin_model_path")
    local _out_path="$output_path/"$(echo $_origin_model_name | cut -d"." -f1)

    convert_model2om "$_out_path"

    local _prof_metrics=(PipeUtilization Memory MemoryUB MemoryL0)
    collect_profiling "$_out_path" ${_prof_metrics[@]}
}

# 执行算子仿真
function execute_operator_simulation() {
    if [ -z $soc_version ]; then
        print_log $WARN "soc_version is not specified, then will get soc version from acl."
        soc_version=$(python3 -c "import acl; print(acl.get_soc_name())")
    fi
    cann_path=$(get_cann_path)
    if [ -z "$cann_path" -o ! -d "$cann_path" ]; then
        print_log $ERROR "Get cann path failed, please check."
        exit 1
    fi
    simulation_path="$cann_path/ascend-toolkit/latest/$(arch)-linux/simulator/"
    if [ ! -d "$simulation_path" ]; then
        print_log $ERROR "Simulation path does not exist."
        exit 1
    fi
    python3 $SHELL_DIR/operator/simulation.py $output_path $soc_version $simulation_path
    if [ $? -ne 0 ]; then
        print_log $ERROR "Operator simulation execute failed."
        exit 1
    fi
}

# 基于仿真，分析算子性能瓶颈
function analysis_operator() {
    local _model_jsons=$1
    local _temp="$output_path/temp.txt"
    local _dump_path="$output_path/dump"

    which msadvisor 2>&1 >>/dev/null
    if [ $? -ne 0 ]; then
        print_log $ERROR "msadvisor not exist, please check."
        return
    fi
    local _dump_dirs=$(ls $_dump_path)
    for dump_dir in ${_dump_dirs[@]}; do
        print_log $INFO "Start to analysis operator $dump_dir"
        timeout 300 msadvisor -c "$_model_jsons" -d "$_dump_path/$dump_dir" 2>&1 >$_temp
        if [ $? -ne 0 ]; then
            print_log $ERROR "Execute msadvisor failed, check log $_dump_path/$dump_dir/log/."
            continue
        fi
        start_line=$(cat $_temp | grep -n "\[Advisor Result\]" | cut -d":" -f1)
        end_line=$(cat $_temp | grep -n "Write result json file" | cut -d":" -f1)
        sed -n "${start_line},${end_line}p" $_temp
        print_log $INFO "Analysis operator $dump_dir succeed."
        echo ""
    done
    rm -f "$_temp"
}

function help_info() {
    echo 'Usage:
    bash msadvisor.sh --type=0 --model="/usr/project/*.onnx" --data="/usr/project/output"
    bash msadvisor.sh --type=1 --data="/usr/project/output" 

Options:
    --type                   <Optional> scene type, support 0:inference; 1:operator, default 0:inference
    --model                  <Optional> model(include caffe/mindspore/tensorflow/onnx) convert command
    --data                              Specify data output path
    --help                   <Optional> print help info

Inference Options:
    --input_shape            <Optional> model input shape, only when model is dynamic shape
    --input_format           <Optional> model input format, only when model is dynamic shape and format is not NCHW
    --weight                 <Optional> model input weight, only when model is tensorflow

Operator Options:
    --soc_version            <Optional> specify soc version, include Ascend310, Ascend310P3 and so on
'
}

function get_ecosystem_knowledge_json_paths() {
    advisor_conf=$SHELL_DIR/advisor.conf
    if [ -f ${advisor_conf} ]; then
        local _conf_title=${support_scene_type[$scene_type]}
        local _model_json_conf=$(python3 -c "import sys; sys.path.append('$SHELL_DIR');\
            import common.common as common;\
            print(common.get_advisor_conf('$advisor_conf', '$_conf_title'))")
        echo $_model_json_conf
    else
        echo ""
    fi
}

function analysis_data() {
    if [ -z $(which msadvisor) ]; then
        print_log $ERROR "msadvisor command not exist, please source set_env.sh firstly."
        exit 1
    fi

    if [ $scene_type -eq 0 ]; then
        local _model_json="model"
    else
        local _model_json="op"
    fi
    local _ecosystem_json=$(get_ecosystem_knowledge_json_paths)
    if [ ! -z $_ecosystem_json ]; then
        _model_json="$_model_json;$_ecosystem_json"
    fi
    if [ $scene_type -eq 0 ]; then
        local _model_dir=$(basename $model_path | cut -d"." -f1)
        timeout 300 msadvisor -c "$_model_json" -d "$output_path/$_model_dir"
    else
        analysis_operator $_model_json
    fi
}

function check_data_path() {
    if [ -z "$output_path" ]; then
        print_log $ERROR "Output path is empty, please specify output path."
        help_info
        exit 1
    fi
    if [ ! -d "$output_path" ]; then
        print_log $INFO "Output path not exist, auto create."
        mkdir -p $output_path
        if [ $? -ne 0 ]; then
            print_log $ERROR "Output path not exist, create failed."
            exit 1
        fi
    fi
}

function collect_data() {
    check_data_path
    output_path=$(realpath "$output_path")

    # support scene type: 0: inference; 1: operator
    if [ $scene_type -eq 0 ]; then
        collect_inference_data
        return
    fi

    if [ $scene_type -eq 1 ]; then
        execute_operator_simulation
        return
    fi
}

function main() {
    export ASCEND_SLOG_PRINT_TO_STDOUT=0

    collect_data
    analysis_data
}

declare -A support_scene_type=([0]='inference' [1]='operator')
# 场景类型
scene_type=0
# 执行指令
model_convert_cmd=""
app_cmd=""
# 路径
output_path=""
model_path=""
# 模型参数
input_shape=""
input_format=""
weight_path=""

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
    --model=*)
        model_path=$(echo "$1" | cut -d"=" -f2)
        if [ -z "$model_path" ]; then
            print_log $ERROR "model is empty."
            exit 1
        fi
        if [ ! -f "$model_path" ]; then
            print_log $ERROR "model not exist."
            exit 1
        fi
        shift
        ;;
    --weight=*)
        weight_path=$(echo "$1" | cut -d"=" -f2)
        if [ ! -f "$weight_path" ]; then
            print_log $ERROR "weight file not exist."
            exit 1
        fi
        shift
        ;;
    --input_shape=*)
        input_shape=$(echo "$1" | cut -d"=" -f2)
        if [ -z $input_shape ]; then
            print_log $ERROR "input shape is empty."
            exit 1
        fi
        shift
        ;;
    --input_format=*)
        input_format=$(echo "$1" | cut -d"=" -f2)
        if [ -z $input_format ]; then
            print_log $ERROR "input format is empty."
            exit 1
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
        shift
        ;;
    --soc_version=*)
        soc_version=$(echo "$1" | cut -d"=" -f2-)
        if [ ! -z "$soc_version" ]; then
            print_log $ERROR "soc_version is empty."
            exit 1
        fi
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

