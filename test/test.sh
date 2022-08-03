#!/bin/bash


cp ../auto_optimizer ./ -rf

coverage run -p -m unittest
coverage combine
coverage report -m