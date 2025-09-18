#!/bin/bash
#SBATCH -A berzelius-2025-243
#SBATCH --job-name=FedDcprivacy_KT_RL_lamda_min_0.4_max_0.6_GE_30_exp_0_mobilevit_kappa_08

#SBATCH --output=FedDcprivacy_KT_RL_lamda_min_0.4_max_0.6_GE_30_exp_0_mobilevit_kappa_08.txt

#SBATCH --gpus=1
#SBATCH -C thin              
#SBATCH -t 02:00:00                 


# Activate env
source ~/.bashrc
conda activate personalized_fl   # or: source ~/miniconda3/bin/activate personalized_fl

srun python -W ignore main.py --algorithm=FedDcprivacy_KT_RL --country=both --num_global_iters=30 --local_iters=5 --lambda_1=0.4 --lambda_2=0.6 --num_teams=2 --kappa=0.8 --delta=1.0 --exp_start=0 --model_name=timm_mobilevit 