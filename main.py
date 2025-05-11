import warnings
warnings.filterwarnings("ignore")

from src.Fedavg.FedAvgServer import FedAvg
from src.Fedmem.FedMEMServer import Fedmem
from src.FedDCPrivacy.server import Server
# from src.dynamic_FedDCPrivacy.dynamic_server import Server
from src.Siloed.SiloedServer import Siloedserver
from src.ClusteredFedDC.clustered_server import C_server
from src.apriori_FedDCPrivacy.server import Server as AprioriFedDCServer
from src.FedProx.FedProxServer import FedProxServer
#from src.ClusteredFedRep.server import ClusteredFedRepServer

from src.TrainModels.trainmodels import *
from src.utils.options import args_parser
import torch
import os
import wandb

torch.manual_seed(0)


def main(args):

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    current_directory = os.getcwd()
    print(current_directory)
    i = args.exp_start
    while i < args.times:
        try:
            if args.algorithm == "Siloed":
                server = Siloedserver(device, args,i, current_directory)
            if args.algorithm == "FedAvg":
                server = FedAvg(device, args,i, current_directory)
            if args.algorithm == "FedProx":
                server = FedProxServer(device, args,i, current_directory)
            elif args.algorithm == "Fedmem": 
                server = Fedmem(device, args, i, current_directory)
            elif args.algorithm == "FedDC":
                server = Server(device, args, i, current_directory)
            elif args.algorithm == "dynamic_FedDcprivacy":
                server = Server(device, args, i, current_directory)
            elif args.algorithm == "Clustered_FedDC":
                server = C_server(device, args, i, current_directory)
            elif args.algorithm == "apriori_FedDcprivacy":
                server = AprioriFedDCServer(device, args, i, current_directory)
            #elif args.algorithm == "ClusteredFedRep":
            #    server = ClusteredFedRepServer(device, args, i, current_directory)

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
    print("kappa       : {}".format(args.kappa))
    
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_iters))
    
    print("=" * 80)

    
    main(args)