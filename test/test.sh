#!/bin/bash

set -u

pwd_dir=${PWD}

# copy auto_optimizer to test file, and test
cp ${pwd_dir}/../auto_optimizer ${pwd_dir}/ -rf

coverage run -p -m unittest
coverage combine
coverage report -m --omit="test_*.py" > ${pwd_dir}/test.coverage

coverage_line=`cat ${pwd_dir}/test.coverage | grep "TOTAL" | awk '{print $4}' | awk '{print int($0)}'`

target=60
if [ ${coverage_line} -lt ${target} ]; then
    echo "coverage failed! coverage_line=${coverage_line}, Coverage does not achieve target(${target}%), Please add ut case."
    exit -1
fi

echo "coverage_line=${coverage_line}"
