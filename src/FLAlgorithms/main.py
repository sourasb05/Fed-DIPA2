import warnings
warnings.filterwarnings("ignore")

from server.Serveravg import FedAvg_server
from server.Serverprox import FedProx_server
from server.Servermoon import MOON_server
from TrainModels.trainmodels import *
from utils.options import args_parser
import torch
import os

torch.manual_seed(0)


def main(args):

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    current_directory = os.getcwd()
    print(current_directory)
    i = args.exp_start
    while i < args.times:
        try:
            if args.algorithm == "FedAvg":
                server = FedAvg_server(device, args,i, current_directory)
            elif args.algorithm == "FedProx": 
                server = FedProx_server(device, args, i, current_directory)
            elif args.algorithm == "MOON":
                server = MOON_server(device, args, i, current_directory)


            """elif args.algorithm == "FedDcprivacy":
                server = Server(device, args, i, current_directory)
            elif args.algorithm == "dynamic_FedDcprivacy":
                server = Server(device, args, i, current_directory)
            elif args.algorithm == "Clustered_FedDC":
                server = C_server(device, args, i, current_directory)
            elif args.algorithm == "apriori_FedDcprivacy":
                server = AprioriFedDCServer(device, args, i, current_directory)
            """
        except ValueError:
            raise ValueError("Wrong algorithm selected")

        if args.test:
            server.test()    
        else:
            server.train()
        i+=1

if __name__ == "__main__":
    args = args_parser()
    
    print("=" * 80)
    print("Summary:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("alpha       : {}".format(args.alpha))
    print("eta      : {}".format(args.eta))
    if args.algorithm == "FedProx":
        print("lambda_prox       : {}".format(args.lambda_prox))
    elif args.algorithm == "MOON":
        print("mu       : {}".format(args.mu))
        print("temperature       : {}".format(args.temperature))

    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_iters))
    
    print("=" * 80)

    
    main(args)