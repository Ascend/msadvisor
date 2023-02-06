#!/bin/bash
SHELL_DIR=$(readlink -f $(cd "$(dirname "$0")" || exit;pwd))

ERROR=error
WARN=warning
INFO=info

function print_log() {
    local content=`echo "$@" | cut -d" " -f2-`
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[$1] [${cur_date}]: $content"
}

function set_msadvisor_max_run_time_config() {
    if [ -z "${ASCEND_TOOLKIT_HOME}" ]; then
        print_log $ERROR "Not set environment variables of CANN"
        exit 1
    fi
    local _max_run_time=$1
    sed -i "4c max_run_time=${_max_run_time}" ${ASCEND_TOOLKIT_HOME}/tools/msadvisor/conf/advisor.conf
}

function help_info() {
    echo 'Usage:
    bash msadvisor.sh --rank_size=24 --bucket_name="obs://path/to/profiler"    Execute advisor with profiling data download
    bash msadvisor.sh --rank_size=24 --data="path/to/profiler"                 Execute advisor with data path

Options:
    --rank_size                         Number of NPUs used for training
    --bucket_name           <Optional>  Directory path for storing profiling data on OBS
    --data                  <Optional>  When data download is not required, directly use --data Incoming profiling data path
    --step                  <Optional>  Specify step num in training
'
}


function collect_data() {
    local _out_path=$1
    local _rank=${rank_size}
    local _data_path=${bucket_name}

    if [ -z "${_data_path}" ]; then
        print_log $ERROR "bucket_name is empty"
        help_info
        exit 1
    fi

    if [ -z "${_rank}" ]; then
        print_log $ERROR "rank_size is empty"
        help_info
        exit 1
    fi

    if [ -d "${_out_path}" ]; then
        rm -rf "${_out_path}"
    fi

    python3 ./utils/obs_data_collector.py --rank_size=${rank_size} --bucket_name=${bucket_name} --download_path=${_out_path}

    if [ $? -ne 0 ]; then
        print_log $ERROR "obs data collection failed"
        exit 1
    fi
}

function hcclanalysis() {
    local _data=$1
    local _rank=${rank_size}
    local _step_num=${step_num}

    if [ -z "${_data}" ]; then
        print_log $ERROR "data path is empty"
        help_info
        exit 1
    fi

    if [ -z "${_rank}" ]; then
        print_log $ERROR "rank_size is empty"
        help_info
        exit 1
    fi

    if [ -n "${_step_num}" ]; then
      msadvisor -c "$SHELL_DIR/hccl_analysis_model/hcclanalysis.json" -d $_data -p "hcclanalysis.rank_size=$_rank;hcclanalysis.step_num=$_step_num"
    else
      msadvisor -c "$SHELL_DIR/hccl_analysis_model/hcclanalysis.json" -d $_data -p "hcclanalysis.rank_size=$_rank"
    fi
}


function main() {
    set_msadvisor_max_run_time_config 1800

    if [ -z "${data_path}" ]; then
        data_path="$(dirname $(dirname "$SHELL_DIR"))/profiler"
        collect_data "$data_path"
    fi

    hcclanalysis "$data_path"
}

while true; do
    case "$1" in 
    --rank_size=*)
        rank_size=$(echo "$1" | cut -d"=" -f2)
        if [ -z "${rank_size}" ]; then
            print_log $ERROR "rank_size is empty."
            exit 1
        fi
        shift
        ;;
    --bucket_name=*)
        bucket_name=$(echo "$1"|  cut -d"=" -f2)
        if [ -z "${bucket_name}" ]; then
            print_log $ERROR "bucket_name is empty."
            exit 1
        fi
        shift
        ;;
    --data=*)
        data_path=$(echo "$1"|  cut -d"=" -f2)
        if [ -z "${data_path}" ]; then
            print_log $ERROR "Data path is empty, please specify profiling data path."
            exit 1
        fi

        if [ ! -d "${data_path}" ]; then
            print_log $ERROR "data_path not exist."
            exit 1
        fi
        shift
        ;;
    --step=*)
        step_num=$(echo "$1"|  cut -d"=" -f2)
        expr "${step_num}" + 1 &> /dev/null
        if [ $? -ne 0 ]; then
            print_log $ERROR "step is invalid num."
            exit 1
        fi
        shift
        ;;
    --help)
        help_info
        exit 0
        ;;
    *)
        if [ -n "$1" ]; then
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
