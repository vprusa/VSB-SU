#!/bin/bash

#python -m venv .venv
#.venv/bin/pip install pandas scikit-learn

#python -m unittest -v ./tests/test.py 
[[ -n "${1}" ]] && TARGET_VAR="${1}" || TARGET_VAR='Gender'
#python ./main.py
#python ./main.py -i ./drug200.csv -t ${TARGET_VAR}
#
if [[ ${TARGET_VAR} = *"all"* ]] ; then
  for i in Age ; do
    echo "Start - Target variable: ${i}"
#    .venv/bin/python ./main.py -i ./Mall_Customers.csv -t $i
    .venv/bin/python ./main.py -i Mall_Customers.csv -t ${i} -x 'Annual Income (k$)' -y 'Spending Score (1-100)' -c 2 -g CustomerId
    echo "done"
  done
else
  .venv/bin/python ./main.py -i Mall_Customers.csv -t ${TARGET_VAR} -x 'Annual Income (k$)' -y 'Spending Score (1-100)' -c 3 -g CustomerId
fi

