#!/bin/bash
#SBATCH -A berzelius-2025-243
#SBATCH --job-name=CFedDC_Movielens_40clients
#SBATCH --output=CFedDC_Movielens_40clients.txt

#SBATCH --reservation=1g.10gb     # Use the MIG reservation
#SBATCH --gpus=1                   # One MIG GPU in that reservation
#SBATCH -t 05:00:00                # 5 hours walltime
#SBATCH --mem=32G                  # MIG slice provides ~32 GB RAM               


# Activate env
source ~/.bashrc
conda activate personalized_fl   # or: source ~/miniconda3/bin/activate personalized_fl

srun python -W ignore main.py 