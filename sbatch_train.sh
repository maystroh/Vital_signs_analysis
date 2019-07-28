#!/bin/bash

#SBATCH -p 2080GPU
#SBATCH -J ppg_train
#SBATCH -c 7
#SBATCH --output=ppg_train_%j.out
#SBATCH --gres=gpu

srun singularity exec --bind /data_GPU:/data_GPU --nv /data_GPU/hassan/Containers/ecg_ppg_latest.simg sh train_ppg_new.sh