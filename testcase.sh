python -W ignore main.py --algorithm=FedDcprivacy --country=both --num_global_iters=10 --local_iters=10 --wandb
python -W ignore main.py --algorithm=Siloed --country=both --num_global_iters=3 --local_iters=2 --wandb



interactive --gpus=1 -t 03:00:00 --account=berzelius-2023-313
conda activate personalized_fl

python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=30 --local_iters=5  --num_teams=2 --kappa=1.0 --delta=0.1 > CFedDC_k_1_d_01.txt
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=30 --local_iters=5  --num_teams=2 --kappa=1.0 --delta=0.5 > CFedDC_k_1_d_05.txt
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=30 --local_iters=5  --num_teams=2 --kappa=1.0 --delta=0.8 > CFedDC_k_1_d_08.txt
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=30 --local_iters=5  --num_teams=2 --kappa=0.1 --delta=1.0 > CFedDC_k_01_d_1.txt
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=30 --local_iters=5  --num_teams=2 --kappa=0.5 --delta=1.0 > CFedDC_k_05_d_1.txt
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=30 --local_iters=5  --num_teams=2 --kappa=0.8 --delta=1.0 > CFedDC_k_08_d_1.txt



python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=30 --local_iters=5 --wandb --num_teams=3 --kappa=1.0 --delta=1.0 
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=30 --local_iters=5  --num_teams=4 --kappa=1.0 --delta=1.0 
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=30 --local_iters=5  --num_teams=5 --kappa=1.0 --delta=1.0
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=30 --local_iters=5  --num_teams=10 --kappa=1.0 --delta=1.0 > CFedDC_C10.txt
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=30 --local_iters=5  --num_teams=15 --kappa=1.0 --delta=1.0


python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=2 --local_iters=5  --num_teams=2 --lamda_sim_sta=0.0 --kappa=1.0 --delta=1.0 > CFedDC_l_00_k_1_d_1.txt
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=30 --local_iters=5  --num_teams=2 --lamda_sim_sta=0.2 --kappa=1.0 --delta=1.0 > CFedDC_l_02_k_1_d_1.txt
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=30 --local_iters=5  --num_teams=2 --lamda_sim_sta=0.4 --kappa=1.0 --delta=1.0 > CFedDC_l_04_k_1_d_1.txt
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=30 --local_iters=5  --num_teams=2 --lamda_sim_sta=0.6 --kappa=1.0 --delta=1.0 > CFedDC_l_06_k_1_d_1.txt
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=30 --local_iters=5  --num_teams=2 --lamda_sim_sta=0.8 --kappa=1.0 --delta=1.0 > CFedDC_l_08_k_1_d_1.txt
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=30 --local_iters=5  --num_teams=2 --lamda_sim_sta=1.0 --kappa=1.0 --delta=1.0 > CFedDC_l_1_k_1_d_1.txt

python -W ignore main.py --algorithm=Fedmem --country=both --num_global_iters=30 --local_iters=5  --num_teams=2 --lambda_1=0.5 --lambda_2=0.5 > FedMME_l1_05_l2_05.txt


python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=30 --local_iters=5  --num_teams=2 --lamda_sim_sta=0.4 --kappa=1.0 --delta=1.0 > CFedDC_l_04_k_1_d_1.txt


python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=30 --local_iters=5 --lambda_1=0.2 --lambda_2=0.8 --num_teams=2 --kappa=1.0 --delta=1.0 > CFedDC_lamda_min_0.2_max_0.8_GE_30.txt
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=30 --local_iters=5 --lambda_1=0.4 --lambda_2=0.6 --num_teams=2 --kappa=1.0 --delta=1.0 > CFedDC_lamda_min_0.4_max_0.6_GE_30.txt
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=30 --local_iters=5 --lambda_1=0.6 --lambda_2=0.4 --num_teams=2 --kappa=1.0 --delta=1.0 > CFedDC_lamda_min_0.6_max_0.4_GE_30.txt
python -W ignore main.py --algorithm=dynamic_FedDcprivacy --country=both --num_global_iters=30 --local_iters=5 --lambda_1=0.2 --lambda_2=0.8 --num_teams=2 --kappa=1.0 --delta=1.0 > CFedDC_lamda_min_0.8_max_0.2_GE_30.txt

python -W ignore main.py --algorithm=FedDcprivacy_KT_RL --country=both --num_global_iters=100 --local_iters=5 --lambda_1=0.1 --lambda_2=0.9 --num_teams=2 --kappa=1.0 --delta=1.0 > FedDcprivacy_KT_RL_lamda_min_0.1_max_0.9_GE_100.txt
python -W ignore main.py --algorithm=FedDcprivacy_KT_RL --country=both --num_global_iters=100 --local_iters=5 --lambda_1=0.2 --lambda_2=0.8 --num_teams=2 --kappa=1.0 --delta=1.0 > FedDcprivacy_KT_RL_lamda_min_0.2_max_0.8_GE_100.txt

python -W ignore main.py --algorithm=FedDcprivacy_KT_RL --country=both --num_global_iters=100 --local_iters=5 --lambda_1=0.3 --lambda_2=0.7 --num_teams=2 --kappa=1.0 --delta=1.0 > FedDcprivacy_KT_RL_lamda_min_0.3_max_0.7_GE_100.txt
python -W ignore main.py --algorithm=FedDcprivacy_KT_RL --country=both --num_global_iters=100 --local_iters=5 --lambda_1=0.4 --lambda_2=0.6 --num_teams=2 --kappa=1.0 --delta=1.0 > FedDcprivacy_KT_RL_lamda_min_0.4_max_0.6_GE_100.txt
python -W ignore main.py --algorithm=FedDcprivacy_KT_RL --country=both --num_global_iters=100 --local_iters=5 --lambda_1=0.5 --lambda_2=0.5 --num_teams=2 --kappa=1.0 --delta=1.0 > FedDcprivacy_KT_RL_lamda_min_0.5_max_0.5_GE_100.txt

python -W ignore main.py --algorithm=FedDcprivacy_KT_RL --country=both --num_global_iters=30 --local_iters=5 --lambda_1=0.3 --lambda_2=0.7 --num_teams=2 --kappa=1.0 --delta=1.0 > FedDcprivacy_KT_RL_c_2_lamda_min_0.3_max_0.7_GE_30_silhouette.txt
python -W ignore main.py --algorithm=FedDcprivacy_KT_RL --country=both --num_global_iters=30 --local_iters=5 --lambda_1=0.3 --lambda_2=0.7 --num_teams=5 --kappa=1.0 --delta=1.0 > FedDcprivacy_KT_RL_c_5_lamda_min_0.3_max_0.7_GE_30_silhouette.txt
python -W ignore main.py --algorithm=FedDcprivacy_KT_RL --country=both --num_global_iters=30 --local_iters=5 --lambda_1=0.3 --lambda_2=0.7 --num_teams=10 --kappa=1.0 --delta=1.0 > FedDcprivacy_KT_RL_c_10_lamda_min_0.3_max_0.7_GE_30_silhouette.txt
python -W ignore main.py --algorithm=FedDcprivacy_KT_RL --country=both --num_global_iters=30 --local_iters=5 --lambda_1=0.3 --lambda_2=0.7 --num_teams=15 --kappa=1.0 --delta=1.0 > FedDcprivacy_KT_RL_c_15_lamda_min_0.3_max_0.7_GE_30_silhouette.txt


python -W ignore main.py --algorithm=FedDcprivacy_KT_RL --country=both --num_global_iters=30 --local_iters=5 --lambda_1=0.3 --lambda_2=0.7 --num_teams=2 --kappa=0.1 --delta=1.0 > FedDcprivacy_KT_RL_lamda_min_0.3_max_0.7_GE_30_kappa_01_delta_1.txt
python -W ignore main.py --algorithm=FedDcprivacy_KT_RL --country=both --num_global_iters=30 --local_iters=5 --lambda_1=0.3 --lambda_2=0.7 --num_teams=2 --kappa=0.3 --delta=1.0 > FedDcprivacy_KT_RL_lamda_min_0.3_max_0.7_GE_30_kappa_03_delta_1.txt
python -W ignore main.py --algorithm=FedDcprivacy_KT_RL --country=both --num_global_iters=30 --local_iters=5 --lambda_1=0.3 --lambda_2=0.7 --num_teams=2 --kappa=0.5 --delta=1.0 > FedDcprivacy_KT_RL_lamda_min_0.3_max_0.7_GE_30_kappa_05_delta_1.txt
python -W ignore main.py --algorithm=FedDcprivacy_KT_RL --country=both --num_global_iters=30 --local_iters=5 --lambda_1=0.3 --lambda_2=0.7 --num_teams=2 --kappa=0.8 --delta=1.0 > FedDcprivacy_KT_RL_lamda_min_0.3_max_0.7_GE_30_kappa_08_delta_1.txt

python -W ignore main.py --algorithm=FedDcprivacy_KT_RL --country=both --num_global_iters=30 --local_iters=5 --lambda_1=0.3 --lambda_2=0.7 --num_teams=2 --kappa=1.0 --delta=0.1 > FedDcprivacy_KT_RL_lamda_min_0.3_max_0.7_GE_30_kappa_1_delta_01.txt
python -W ignore main.py --algorithm=FedDcprivacy_KT_RL --country=both --num_global_iters=30 --local_iters=5 --lambda_1=0.3 --lambda_2=0.7 --num_teams=2 --kappa=1.0 --delta=0.3 > FedDcprivacy_KT_RL_lamda_min_0.3_max_0.7_GE_30_kappa_1_delta_03.txt
python -W ignore main.py --algorithm=FedDcprivacy_KT_RL --country=both --num_global_iters=30 --local_iters=5 --lambda_1=0.3 --lambda_2=0.7 --num_teams=2 --kappa=1.0 --delta=0.5 > FedDcprivacy_KT_RL_lamda_min_0.3_max_0.7_GE_30_kappa_1_delta_05.txt
python -W ignore main.py --algorithm=FedDcprivacy_KT_RL --country=both --num_global_iters=30 --local_iters=5 --lambda_1=0.3 --lambda_2=0.7 --num_teams=2 --kappa=1.0 --delta=0.8 > FedDcprivacy_KT_RL_lamda_min_0.3_max_0.7_GE_30_kappa_1_delta_08.txt


python -W ignore main.py --algorithm=FedDcprivacy_KT_RL --country=both --num_global_iters=30 --local_iters=5 --lamda_sim_sta=0.0 --num_teams=2 --kappa=1.0 --delta=1.0 > FedDcprivacy_KT_RL_no_lamda_GE_30_kappa_1_delta_1.txt
