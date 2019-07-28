#!/usr/bin/env bash

#PYTHONPATH="/data_GPU/hassan/CHU brest/ECG_Nature_Study/ecg/:"
#export PYTHONPATH
export PYTHONIOENCODING=utf-8

python classes/train_new.py config.json -e ppg_tests

#CUDA_VISIBLE_DEVICES=0
#"conv_subsample_lengths": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],

