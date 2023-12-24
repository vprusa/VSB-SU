#!/bin/bash

#python -m venv .venv
#.venv/bin/pip install pandas scikit-learn

#python -m unittest -v ./tests/test.py 
[[ -n "${1}" ]] && TARGET_VAR="${1}" || TARGET_VAR='Drug'
#python ./main.py
#python ./main.py -i ./drug200.csv -t ${TARGET_VAR}
#
if [[ ${TARGET_VAR} = *"all"* ]] ; then
  for i in Age Sex BP Cholesterol Na_to_K Drug ; do
    echo "Start - Target variable: ${i}"
    .venv/bin/python ./main.py -i ./drug200.csv -t $i
    echo "done"
  done
else
  .venv/bin/python ./main.py -i Mall_Customers.csv -t ${TARGET_VAR}
fi

