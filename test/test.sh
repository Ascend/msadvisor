#!/bin/bash


cp ../auto_optimizer ./ -rf

coverage3 run -p -m unittest
coverage3 combine
coverage3 report -m