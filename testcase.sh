python -W ignore main.py --algorithm=FedDcprivacy --country=both --num_global_iters=10 --local_iters=10 --wandb
python -W ignore main.py --algorithm=Siloed --country=both --num_global_iters=3 --local_iters=2 --wandb



interactive --gpus=1 -t 03:00:00 --account=berzelius-2023-313
conda activate personalized_fl

python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=20 --local_iters=10 --wandb --num_teams=10 --kappa=0.1 --delta=0.1 
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=20 --local_iters=10 --wandb --num_teams=10 --kappa=0.5 --delta=0.1 
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=20 --local_iters=10 --wandb --num_teams=10 --kappa=0.8 --delta=0.1 
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=20 --local_iters=10 --wandb --num_teams=10 --kappa=0.1 --delta=0.5 
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=20 --local_iters=10 --wandb --num_teams=10 --kappa=0.5 --delta=0.5 
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=20 --local_iters=10 --wandb --num_teams=10 --kappa=0.8 --delta=0.5 
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=20 --local_iters=10 --wandb --num_teams=10 --kappa=0.1 --delta=0.8 
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=20 --local_iters=10 --wandb --num_teams=10 --kappa=0.5 --delta=0.8
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=20 --local_iters=10 --wandb --num_teams=10 --kappa=0.8 --delta=0.8 
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=20 --local_iters=10 --wandb --num_teams=10 --kappa=1.0 --delta=1.0 