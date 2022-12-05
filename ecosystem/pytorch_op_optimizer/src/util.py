import os
import json


def load_json(path):
    with open(path, 'r', encoding='UTF-8') as task_json_file:
        json_data = json.load(task_json_file)
    return json_data
