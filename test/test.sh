#!/bin/bash

set -u

pwd_dir=${PWD}

# copy auto_optimizer to test file, and test
cp ${pwd_dir}/../auto_optimizer ${pwd_dir}/ -rf

coverage run -p -m unittest
coverage combine
coverage report -m --omit="test_*.py" > ${pwd_dir}/test.coverage

coverage_line=`cat ${pwd_dir}/test.coverage | grep "TOTAL" | awk '{print $4}' | awk '{print int($0)}'`

echo "coverage_line="$coverage_line

if [ ${coverage_line} -lt 60 ]; then
    echo "coverage failed"
    exit -1
fi
