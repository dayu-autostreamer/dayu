"""
Dayu Log Analysis Tool

Tool script to parse and analyze log exported from dayu system.

Log files can be downloaded from frontend ui (in "Result Display" module , click the "Export Log" button).

Examples:
    python tools/log_analysis.py --log file_path

"""

import sys
import json
import os
import argparse

sys.path.append('./dependency')

from core.lib.content import Task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dayu Log Analysis Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--log",
        type=str,
        required=True,
        default=None,  # 显式设置默认值
        metavar="LOG_FILE_PATH",
        help="Specify the log file path"
    )

    return parser.parse_args()


def parse_logs(record_file):
    if not os.path.exists(record_file):
        print(f"Log file {record_file} not exists")
        return None
    if not os.path.isfile(record_file):
        print(f"Log file {record_file} is not a file")
        return None
    try:
        with open(record_file) as f:
            json_data = json.load(f)
            tasks = []
            for task_data in json_data:
                task = Task.deserialize(task_data)
                tasks.append(task)
    except Exception as e:
        print(f"Failed to parse log file {record_file}, this file is not exported from dayu system.")
        return None

    return tasks

def analyze_logs(tasks):
    # add analyze functions here.
    pass


def parse_and_analyze_logs(file_path):
    tasks = parse_logs(file_path)
    if not tasks:
        return

    print('##################################################################')
    print('###################### Dayu Log Analysis Tool ####################')
    print(f'Analyze log file {os.path.basename(file_path).split(".")[0]} ..')
    print()

    analyze_logs(tasks)

    print('##################################################################')
    print()
    print()


if __name__ == '__main__':
    args = parse_args()
    parse_and_analyze_logs(args.log)

