#!/bin/bash

#SBATCH -p 1080GPU
#SBATCH -J test_ppg
#SBATCH -c 7
#SBATCH --output=ppg_test_%j.out
#SBATCH --gres=gpu

srun singularity exec --bind /data_GPU:/data_GPU --nv /data_GPU/hassan/Containers/ecg_ppg.simg sh test_ppg.sh

