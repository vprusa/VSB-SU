#!/bin/bash

#python -m venv .venv
#.venv/bin/pip install pandas scikit-learn

[[ -n "${1}" ]] && TARGET_VAR="${1}" || TARGET_VAR='Gender'
[[ -n "${2}" ]] && CNT="${2}" || CNT='5'
.venv/bin/python ./main.py -i Mall_Customers.csv -t ${TARGET_VAR} -x 'Annual Income (k$)' -y 'Spending Score (1-100)' -c "${CNT}" -g "CustomerID"

