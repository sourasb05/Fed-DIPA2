import torch
import os
import h5py
from src.Siloed.SiloedUser import Siloeduser
import numpy as np
import copy
from tqdm import trange
from tqdm import tqdm
import numpy as np
import sys
import wandb
import datetime
import json

import torch
from torchmetrics import Precision, Recall, F1Score
from src.utils.results_utils import InformativenessMetrics

class Siloedserver():
    def __init__(self,device, args, exp_no, current_directory):
                
        self.device = device
        self.local_iters = args.local_iters
        self.batch_size = args.batch_size
        self.learning_rate = args.alpha
        
        self.total_train_samples = 0
        self.exp_no = exp_no
        self.algorithm = args.algorithm
        
        self.current_directory = current_directory

        self.country = args.country
        if args.country == "japan":
            self.user_ids = args.user_ids[0]
            self.total_users = len(self.user_ids)
        elif args.country == "uk":
            self.user_ids = args.user_ids[1]
            self.total_users = len(self.user_ids)
        
        elif args.country == "both":
            self.user_ids = args.user_ids[3]
            self.total_users = len(self.user_ids)
        
        else:
            self.user_ids = args.user_ids[2]
            self.total_users = len(self.user_ids)
            print(f"self.total_users : {self.total_users}")
            
            
  
        self.users = []
        self.selected_users = []

        self.global_test_metric = []
        self.global_test_loss = []
        self.global_test_distance = []
        self.global_test_mae = []

        self.global_train_metric = []
        self.global_train_loss = []
        self.global_train_distance = []
        self.global_train_mae = []

        self.data_frac = []
        
        self.minimum_test_loss = 1000000.0

        date_and_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.wandb = wandb.init(project="DIPA2", name="Siloed_%s_%d" % (date_and_time, self.total_users), mode=None if args.wandb else "disabled")
                
        for i in trange(self.total_users, desc="Data distribution to clients"):
            user = Siloeduser(device, args, int(self.user_ids[i]), exp_no, current_directory, self.wandb)
            if user.valid: # Copy for all algorithms
                self.users.append(user)
                self.total_train_samples += user.train_samples
        
        #Create Global_model
        for user in self.users:
            self.data_frac.append(user.train_samples/self.total_train_samples)
        print(f"data available {self.data_frac}")
        self.global_model = copy.deepcopy(self.users[0].local_model)
        for param in self.global_model.parameters():
            param.data.zero_()
        

        print("Finished creating FedAvg server.")

    def __del__(self):
        self.wandb.finish()
        
    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.global_model)
    
    def eval_train(self):
        avg_loss = 0.0
        avg_distance = 0.0
        avg_mae = 0.0
        accumulator = {}
        for c in self.users:
            loss, distance, c_dict, mae = c.train_evaluation()
            avg_loss += (1/len(self.users))*loss
            avg_distance += (1/len(self.users))*distance
            avg_mae += (1/len(self.users))*mae
            if c_dict:  # Check if test_dict is not None or empty
                self.initialize_or_add(accumulator, c_dict)
        average_dict = {key: [x / len(self.users) for x in value] for key, value in accumulator.items()}

        self.wandb.log(data={ "global_train_loss" : avg_loss})

        self.global_train_metric.append(average_dict)
        self.global_train_loss.append(avg_loss)
        self.global_train_distance.append(avg_distance)
        self.global_train_mae.append(avg_mae)

                    
        print(f"siloed avg Train loss {avg_loss} avg distance {avg_distance}") 
        print(f"siloed avg Train Performance metric : {average_dict}")
        print(f"siloed avg Train global mae : {avg_mae}")


    def initialize_or_add(self, dest, src):
    
        for key, value in src.items():
            if key in dest:
                dest[key] = [x + y for x, y in zip(dest[key], value)]
            else:
                dest[key] = value.copy()  # Initialize with a copy of the first list

    
  
    def eval_test(self):
        avg_loss = 0.0
        avg_distance = 0.0
        accumulator = {}
        avg_mae = 0.0
        for c in self.users:
            loss, distance, c_dict, mae = c.test()
            avg_loss += (1/len(self.users))*loss
            avg_distance += (1/len(self.users))*distance
            avg_mae += (1/len(self.users))*mae
            if c_dict:  # Check if test_dict is not None or empty
                self.initialize_or_add(accumulator, c_dict)
        average_dict = {key: [x / len(self.users) for x in value] for key, value in accumulator.items()}

        
        self.wandb.log(data={ "global_val_loss" : avg_loss})
        self.wandb.log(data={ "global_mae" : avg_mae})
        self.global_test_metric.append(average_dict)
        self.global_test_loss.append(avg_loss)
        self.global_test_distance.append(avg_distance)
        self.global_test_mae.append(avg_mae)
            
                    
        print(f"siloed avg Test loss {avg_loss} avg distance {avg_distance}") 
        print(f"siloed avg Performance metric : {average_dict}")
        print(f"siloed avg Test global mae : {avg_mae}")


    def evaluate(self):
        self.eval_test()
        self.eval_train()
        self.save_results()
    
    def save_results(self):
        date_and_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        file = "avg_siloed_model" +  date_and_time
        
        print(file)
       
        directory_name = str(self.algorithm) + "/" +"h5" + "/siloed_model/" 
        # Check if the directory already exists
        if not os.path.exists(self.current_directory + "/results/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/results/" + directory_name)

        json_test_metric = json.dumps(self.global_test_metric)
        json_train_metric = json.dumps(self.global_train_metric)


        with h5py.File(self.current_directory + "/results/" + directory_name + "/" + '{}.h5'.format(file), 'w') as hf:
            hf.create_dataset('Local iters', data=self.local_iters)
            hf.create_dataset('Learning rate', data=self.learning_rate)
            hf.create_dataset('Batch size', data=self.batch_size)
            hf.create_dataset('global_test_metric', data=[json_test_metric.encode('utf-8')])
            hf.create_dataset('global_test_loss', data=self.global_test_loss)
            hf.create_dataset('global_test_distance', data=self.global_test_distance)
            hf.create_dataset('global_test_mae', data=self.global_test_mae)

            hf.create_dataset('global_train_metric', data=[json_train_metric.encode('utf-8')])
            hf.create_dataset('global_train_loss', data=self.global_train_loss)
            hf.create_dataset('global_train_distance', data=self.global_train_distance)
            hf.create_dataset('global_train_mae', data=self.global_train_mae)

            hf.close()


    def train(self):
        self.send_parameters()
        list_user_id = []
        for user in self.users:
            list_user_id.append(user.id)
            user.train()
        self.evaluate()
    
    def test(self):
        output_channel = {'informationType': 6, 'sharingOwner': 7, 'sharingOthers': 7}
        threshold = 0.5
        average_method = 'weighted'
        metrics = [Precision, Recall, F1Score]
        metrics_data = {}
        for metric in metrics:
            metrics_data[metric.__name__] = [metric(task="multilabel",
                                                    num_labels=output_dim,
                                                    threshold = threshold,
                                                    average=average_method,
                                                    ignore_index = output_dim - 1) \
                                                    for i, (output_name, output_dim) in enumerate(output_channel.items())]
        informativeness_scores = [[], []]

        for user in self.users:
            results = user.test_eval()

            for result in results:
                information, informativeness, sharingOwner, sharingOthers, y_preds = result
                gt = [information, sharingOwner, sharingOthers]
                output_dims = output_channel.values()
                for o, (output_dim, gt) in enumerate(zip(output_dims, gt)):
                    start_dim = o*(output_dim)
                    end_dim = o*(output_dim)+output_dim
                    for metric_name in metrics_data.keys():
                        metrics_data[metric_name][o].update(y_preds[:, start_dim:end_dim], gt)
                informativeness_scores[0].extend(informativeness.detach().cpu().numpy().tolist())
                informativeness_scores[1].extend(y_preds[:, 6].detach().cpu().numpy().tolist())
        results_data = {}
        for metric_name in metrics_data.keys():
            results_data[metric_name] = [i.compute().detach().cpu().numpy() for i in metrics_data[metric_name]]
        for i, k in enumerate(output_channel.keys()):
            for metric, values in results_data.items():
                print("%.02f " % values[i], end="")

        info_prec, info_rec, info_f1, info_cmae, info_mae = InformativenessMetrics(informativeness_scores[0], informativeness_scores[1])
        print("%.02f %.02f %.02f %.02f %.02f" % (info_prec, info_rec, info_f1, info_cmae, info_mae))
