PYTHONPATH="/data_GPU/hassan/CHU brest/ECG_Nature_Study/ecg/:"
export PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python classes/train.py config.json -e ppg_tests

#"conv_subsample_lengths": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
