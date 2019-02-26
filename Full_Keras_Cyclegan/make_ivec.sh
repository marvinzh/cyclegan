#!/bin/bash

condition = $1

date

if [ $# != 1 ]; then
  echo "Choose one evaluation condition('C2' or 'C5')."

python main.py

./ivec_ark2scp_$condition.sh

date
