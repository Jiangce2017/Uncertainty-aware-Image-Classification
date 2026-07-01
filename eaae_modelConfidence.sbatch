#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 10:00:00
#SBATCH --gpus=v100-32:1

module purge
module load AI/pytorch_23.02-1.13.1-py3
conda activate graph

python vae_train.py