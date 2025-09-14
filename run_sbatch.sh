#!/bin/bash
#SBATCH -A berzelius-2024-110
#SBATCH --job-name=july_25_apriori_FedDC_all_cluster_info
#SBATCH --output=july_25_apriori_FedDC_all_cluster_info.txt
#SBATCH --time=05:00:00  # Job will run for 5 hr 00 min
#SBATCH --gres=gpu:1  # Request 1 GPU

# Load necessary modules or activate your environment
source activate personalized_fl

python main.py --wandb --num_global_iters=30 --local_iters=5