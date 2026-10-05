#!/bin/bash
# bash process_data.sh <task_name> <data_version: isaac45|isaac51> <expert_data_num>
# reads <UniVTAC>/data/<data_version>/<task_name>, writes <UniVTAC>/data/act_tacenc/sim-<task>/<data_version>-<N>
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${SCRIPT_DIR}/process_data.py" "${1}" "${2}" "${3}"
