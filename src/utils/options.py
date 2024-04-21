import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default="Basemodel")
    parser.add_argument("--algorithm", type=str, default="Fedmem",
                        choices=["pFedme", "FedAvg", "Fedmem", "FeSEM", "FedProx"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.7,
                        help="Regularizer for pFedme")
    parser.add_argument("--K", type=int, default=30,
                        help="Regularizer for pFedme")
    
    parser.add_argument("--lambda_1", type=float, default=0.25, 
                        help="Regularization term lambda_1")
    parser.add_argument("--lambda_2", type=float, default=0.25, 
                        help="Regularization term lambda_2")
    parser.add_argument("--gamma", type=float, default=0.05, 
                        help="regularization term gamma for PerMFL and scale parameter for RBF kernel in Fedmem")
    parser.add_argument("--alpha", type=float, default=0.05, 
                        help="learning rate for local models in fedmem")
    parser.add_argument("--eta", type=float, default=0.01, 
                        help="personalization parameter for Fedmem")
    
    parser.add_argument("--num_global_iters", type=int, default=20)
    parser.add_argument("--local_iters", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    
    parser.add_argument("--times", type=int, default=1, 
                        help="running time")
    parser.add_argument("--exp_start", type=int, default=0,
                        help="experiment start no")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    
    parser.add_argument("--users_frac", type=float, default=1.0, 
                        help="selected fraction of users available per global round")
    parser.add_argument("--total_users", type=int, default=40, 
                        help="total participants")
    parser.add_argument("--data_silo", type=int, default=100)
   
    parser.add_argument("--num_teams", type=int, default=5,
                        help="Number of teams")
    parser.add_argument("--p_teams", type=int, default=1,
                        help="number of team selected per global round")
    parser.add_argument("--cluster", type = str, default="apriori", choices=["apriori_hsgd", "dynamic", "apriori"])
    parser.add_argument("--target", type=int, default=10, choices=[3,10], help="number of target classes")

    parser.add_argument("--fixed_user_id", type=int, default=16)
    parser.add_argument("--fix_client_every_GR", type=int, default=0, choices=[0,1])

    args = parser.parse_args()

    return args