#!/bin/bash

pip3 install coverage

cp ../auto_optimizer ./ -rf

coverage run -p -m unittest
coverage combine
coverage report -m