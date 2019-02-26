#!/bin/bash

qsub -g tga-tslab -o log/$1 qsub_train.sh main_basic_cyc.py
