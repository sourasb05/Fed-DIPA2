import torch
import os
import h5py
from src.Fedavg.UserFedAvg import UserAvg
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
from src.utils.results_utils import CalculateMetrics, InformativenessMetrics


class FedAvg():
    def __init__(self,device, args, exp_no, current_directory):
                
        self.device = device
        self.num_glob_iters = args.num_global_iters
        self.local_iters = args.local_iters
        self.batch_size = args.batch_size
        self.learning_rate = args.alpha
        
        self.user_ids = args.user_ids
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
            print(self.user_ids)
            self.total_users = len(self.user_ids)

        print(f"total users : {self.total_users}")
         
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
        self.wandb = wandb.init(project="DIPA2", name="FedAvg_%s_%d" % (date_and_time, self.total_users), mode=None if args.wandb else "disabled")
                
        for i in trange(self.total_users, desc="Data distribution to clients"):
            user = UserAvg(device, args, int(self.user_ids[i]), exp_no, current_directory, self.wandb)
            if user.valid: # Copy for all algorithms
                self.users.append(user)
                self.total_train_samples += user.train_samples
        
        self.total_users = len(self.users) 
        self.num_users = self.total_users * args.users_frac    #selected users
        
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

    def add_parameters(self, user, ratio):
        model = self.global_model.parameters()
        for server_param, user_param in zip(self.global_model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)

    

    def save_model(self, glob_iter):
        if glob_iter == self.num_glob_iters-1:
            model_path = self.current_directory + "/models/" + self.algorithm + "/global_model/"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            checkpoint = {'GR': glob_iter,
                        'model_state_dict': self.global_model.state_dict(),
                        'loss': self.minimum_test_loss
                        }
            torch.save(checkpoint, os.path.join(model_path, "server_checkpoint_GR" + str(glob_iter) + ".pt"))
            
        if self.global_test_loss[glob_iter] < self.minimum_test_loss:
            self.minimum_test_loss = self.global_test_loss[glob_iter]
            model_path = self.current_directory + "/models/" + self.algorithm + "/global_model/"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            checkpoint = {'GR': glob_iter,
                        'model_state_dict': self.global_model.state_dict(),
                        'loss': self.minimum_test_loss
                        }
            torch.save(checkpoint, os.path.join(model_path, "server_checkpoint" + ".pt"))

    def select_users(self, round, subset_users):

        if subset_users == len(self.users):
            return self.users
        elif  subset_users < len(self.users):
         
            np.random.seed(round)
            return np.random.choice(self.users, subset_users, replace=False) 

        else: 
            assert (self.subset_users > len(self.users))
            
    
    def initialize_or_add(self, dest, src):
    
        for key, value in src.items():
            if key in dest:
                dest[key] = [x + y for x, y in zip(dest[key], value)]
            else:
                dest[key] = value.copy()  # Initialize with a copy of the first list


    def eval_train(self, t):
        avg_loss = 0.0
        avg_distance = 0.0
        accumulator = {}
        for c in self.selected_users:
            loss, distance, c_dict, mae = c.train_evaluation(self.global_model.parameters(), t)
            avg_loss += (1/len(self.selected_users))*loss
            avg_distance += (1/len(self.selected_users))*distance
            if c_dict:  # Check if test_dict is not None or empty
                self.initialize_or_add(accumulator, c_dict)
        average_dict = {key: [x / len(self.selected_users) for x in value] for key, value in accumulator.items()}

        self.wandb.log(data={ "global_train_loss" : avg_loss})

        self.global_train_metric.append(average_dict)
        self.global_train_loss.append(avg_loss)
        self.global_train_distance.append(avg_distance)
        self.global_train_mae.append(mae)

                    
        print(f"Global round {t} avg loss {avg_loss} avg distance {avg_distance}") 
        

    def save_global_model(self, glob_iter, current_loss):
            
        model_path = self.current_directory + "/models/" + self.algorithm + "/global_model/""_GE_" + str(self.num_glob_iters) + "_LE_" + str(self.local_iters) + "/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if glob_iter == self.num_glob_iters-1:
            
            checkpoint = {'GR': glob_iter,
                        'model_state_dict': self.global_model.state_dict(),
                        'loss': self.minimum_test_loss
                        }
            torch.save(checkpoint, os.path.join(model_path, "server_checkpoint_GR" + str(glob_iter) + ".pt"))
            
        if current_loss < self.minimum_test_loss:
            self.minimum_test_loss = current_loss
            
            
            checkpoint = {'GR': glob_iter,
                        'model_state_dict': self.global_model.state_dict(),
                        'loss': self.minimum_test_loss
                        }
            torch.save(checkpoint, os.path.join(model_path, "best_server_checkpoint" + ".pt"))

  
    def evaluate_global(self, t):
        avg_mae = 0.0
        avg_cmae = 0.0
        avg_loss = 0.0
        for c in self.users:
            loss, info_prec, info_rec, info_f1, info_cmae, info_mae, _ = c.test_global_model_val(self.global_model.parameters())
            test_loss, test_info_prec, test_info_rec, test_info_f1, test_info_cmae, test_info_mae, _ = c.test_global_model_val(self.global_model.parameters())

            print(f"info_prec {info_prec}, info_rec {info_rec}, info_f1 {info_f1}, info_cmae {info_cmae}, info_mae {info_mae}")
            
            avg_mae += (1/len(self.selected_users))*info_mae
            avg_cmae += (1/len(self.selected_users))*info_cmae
            avg_loss += (1/len(self.select_users))*loss
            avg_f1 += (1/len(self.select_users))*info_f1
            
            test_avg_mae += (1/len(self.selected_users))*test_info_mae
            test_avg_cmae += (1/len(self.selected_users))*test_info_cmae
            test_avg_loss += (1/len(self.select_users))*test_loss
            test_avg_f1 += (1/len(self.select_users))*test_info_f1

            
        
        print(f"\n Global round {t} : Global val f1: {avg_f1} Global val cmae {avg_cmae} global val mae : {avg_mae} \n")
        print(f"\n Global round {t} : Global test f1: {test_avg_f1} Global test cmae {test_avg_cmae} global test mae : {test_avg_mae} \n")

        self.save_global_model(t, avg_loss)
    
    def evaluate_local(self, t):
        val_avg_mae = 0.0
        val_avg_cmae = 0.0
        test_avg_mae = 0.0
        test_avg_cmae = 0.0
        for c in self.users:
            info_prec, info_rec, info_f1, info_cmae, info_mae, _ = c.test_local_model_val()
            test_info_prec, test_info_rec, test_info_f1, test_info_cmae, test_info_mae, _ = c.test_local_model_test()
            
            # print(f"info_prec {info_prec}, info_rec {info_rec}, info_f1 {info_f1}, info_cmae {info_cmae}, info_mae {info_mae}")
            
            val_avg_mae += (1/len(self.selected_users))*info_mae
            val_avg_cmae += (1/len(self.selected_users))*info_cmae
            test_avg_mae += (1/len(self.selected_users))*test_info_mae
            test_avg_cmae += (1/len(self.selected_users))*test_info_cmae
        
        print(f"\033[92m\n Global round {t} : Local val cmae {val_avg_cmae} Local val mae : {val_avg_mae} \n\033[0m")   # Green
        print(f"\033[93m\n Global round {t} : Local Test cmae {test_avg_cmae} Local Test mae : {test_avg_mae} \n\033[0m")  # Yellow



    def train(self):
        
        for glob_iter in trange(self.num_glob_iters, desc="Global Rounds"):
            self.send_parameters()
            self.selected_users = self.select_users(glob_iter, self.num_users)
            list_user_id = []
            for user in self.selected_users:
                list_user_id.append(user.id)
            #print(f"Exp no{self.exp_no} : users selected for global iteration {glob_iter} are : {list_user_id}")

            for user in self.selected_users:
                user.train(glob_iter)  # * user.train_samples

            self.aggregate_parameters()
            self.evaluate_global(glob_iter)
            self.evaluate_local(glob_iter)
            self.save_model(glob_iter)
        self.save_results()

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


    def save_results(self):
        for user in self.users:
            val_dict = user.val_round_result_dict
            test_dict = user.test_round_result_dict
            global_val_dict = user.val_global_round_result_dict
            global_test_dict = user.test_global_round_result_dict
        

            user_id = str(user.id)
            val_json_path = f"results/client_level/FedAvg/local_val/user_{user_id}_val_round_results.json"
            test_json_path = f"results/client_level/FedAvg/local_test/user_{user_id}_test_round_results.json"
            val_global_json_path = f"results/client_level/FedAvg/global_val/user_{user_id}_val_round_results.json"
            test_global_json_path = f"results/client_level/FedAvg/global_test/user_{user_id}_test_round_results.json"

        
            # Combine resource category and val_dict into one JSON object
            val_full_output = {
                "User": user_id,
                "validation_results": val_dict
            }

            test_full_output = {
               "User": user_id,
                "validation_results": test_dict
            }

                        # Combine resource category and val_dict into one JSON object
            val_global_full_output = {
                "User": user_id,
                "validation_results": global_val_dict
            }

            test_global_full_output = {
               "User": user_id,
                "validation_results": global_test_dict
            }


            # Ensure the parent folder exists
            os.makedirs(os.path.dirname(val_json_path), exist_ok=True)
            os.makedirs(os.path.dirname(test_json_path), exist_ok=True)
            os.makedirs(os.path.dirname(val_global_json_path), exist_ok=True)
            os.makedirs(os.path.dirname(test_global_json_path), exist_ok=True)


            # Save to JSON file (overwrite if it exists)
            with open(val_json_path, 'w') as f:
                json.dump(val_full_output, f, indent=2, default=self.convert_numpy)
            # Save to JSON file (overwrite if it exists)
            with open(test_json_path, 'w') as f:
                json.dump(test_full_output, f, indent=2, default=self.convert_numpy)

            # Save to JSON file (overwrite if it exists)
            with open(val_global_json_path, 'w') as f:
                json.dump(val_global_full_output, f, indent=2, default=self.convert_numpy)
            # Save to JSON file (overwrite if it exists)
            with open(test_global_json_path, 'w') as f:
                json.dump(test_global_full_output, f, indent=2, default=self.convert_numpy)
