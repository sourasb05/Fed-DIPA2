from src.Fedavg.FedAvgServer import FedAvg
# from src.Fedmem_mm.FedMEMServer import Fedmem_mm
from src.Fedmem.FedMEMServer import Fedmem
from src.FedProx.FedProxServer import FedProx
from src.pFedme.pFedme_server import pFedme
from src.FeSEM.FeSEM_server import FeSEM
from src.TrainModels.trainmodels import *
from src.utils.options import args_parser
import torchvision.models as models
import torch
from tqdm import tqdm, trange
import os


torch.manual_seed(0)


def main(args):

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    current_directory = os.getcwd()
    print(current_directory)
    i = args.exp_start
    while i < args.times:
        if args.model_name == "Basemodel":
            model = BaseModel().to(device)
        try:    
            if args.algorithm == "FedAvg":
                server = FedAvg(device, model, args,i, current_directory)
            elif args.algorithm == "FedProx":
                server = FedProx(device, model, args,i, current_directory)
            elif args.algorithm == "pFedme":
                server = pFedme(device, model, args, i, current_directory)
            elif args.algorithm == "Fedmem": # and args.contexual == 0: 
                server = Fedmem(device, model, args, i, current_directory)
            elif args.algorithm == "FeSEM":
                server = FeSEM(device, model, args, i, current_directory)
            # elif args.algorithm == "Fedmem" and args.contexual == 1:
            #    print("in Fedmem_mm")
            #    server = Fedmem_mm(device, args, i, current_directory)
            
        except ValueError:
            raise ValueError("Wrong algorithm selected")
        server.train()
        i+=1

if __name__ == "__main__":
    args = args_parser()
    
    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("alpha       : {}".format(args.alpha))
    print("beta        : {}".format(args.beta))
    print("gamma       : {}".format(args.gamma))
    print("lambda_1       : {}".format(args.lambda_1))
    print("lambda_2       : {}".format(args.lambda_2))
    print("number of teams : {}".format(args.num_teams))
    print("Cluster type: {}".format(args.cluster))
    print("eta         : {}".format(args.eta))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of team rounds          : {}".format(args.num_team_iters))
    print("Number of local rounds       : {}".format(args.local_iters))
    print("Local Model       : {}".format(args.model_name))

    print("=" * 80)

    
    main(args)