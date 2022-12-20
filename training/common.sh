#!/bin/bash

set -e

sed -i "4c max_run_time=1800" ${ASCEND_TOOLKIT_HOME}/tools/msadvisor/conf/advisor.conf
echo "config set successfully"