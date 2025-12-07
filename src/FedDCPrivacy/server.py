import torch
import os
import h5py
from src.FedDCPrivacy.user import User
import numpy as np
import copy
from tqdm import trange
from tqdm import tqdm
import numpy as np
import random
import sys
import wandb
import datetime
import json
import sys
from torchmetrics import Precision, Recall, F1Score
from src.utils.results_utils import InformativenessMetrics

class Server():
    def __init__(self,device, args, exp_no, current_directory):
                
        self.device = device
        self.num_glob_iters = args.num_global_iters
        self.local_iters = args.local_iters
        self.batch_size = args.batch_size
        self.learning_rate = args.alpha
        self.eta = args.eta
        self.kappa = args.kappa
        self.delta = args.delta
        self.country = args.country
        if args.country == "japan":
            self.user_ids = args.user_ids[0]
        elif args.country == "uk":
            self.user_ids = args.user_ids[1]
        elif args.country == "both":
            self.user_ids = args.user_ids[3]
        else:
            self.user_ids = args.user_ids[2]
        

        # print(f"user ids : {self.user_ids}")
        self.total_users = len(self.user_ids)
        # print(f"total users : {self.total_users}")
        self.total_samples = 0
        self.total_selected_samples = 0
        self.exp_no = exp_no
        self.current_directory = current_directory
        self.algorithm = args.algorithm

        if args.model_name == "openai_ViT-L/14@336px":
            self.model_name = "ViT-L_14_336px"
        else:
            self.model_name = args.model_name
        self.global_metric = []


        self.users = []
        self.selected_users = []

        """
        Global model metrics
        """

        self.global_test_metric = []
        self.global_test_loss = []
        self.global_test_distance = []
        self.global_test_mae = []

        self.global_train_metric = []
        self.global_train_loss = []
        self.global_train_distance = []
        self.global_train_mae = []

        """
        Local model evaluation
        """

        self.local_test_metric = []
        self.local_test_loss = []
        self.local_test_distance = []
        self.local_test_mae = []

        self.local_train_metric = []
        self.local_train_loss = []
        self.local_train_distance = []
        self.local_train_mae = []
        self.minimum_test_loss = 0.0

        self.data_frac = []

        self.data_in_cluster = [0.0,0.0]

        date_and_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.wandb = wandb.init(project="DIPA2", name="FedDCPrivacy_%s_%d" % (date_and_time, self.total_users), mode=None if args.wandb else "disabled")
                
        
        for i in trange(self.total_users, desc="Data distribution to clients"):
            # print(f"client id : {self.user_ids[i]}")
            user = User(device, args, self.user_ids[i], exp_no, current_directory, wandb)
            if user.valid:
                self.users.append(user)
                self.total_samples += user.train_samples
                
        
        print("Finished creating FedDC server.")

        self.global_model = copy.deepcopy(self.users[0].local_model)

            
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

            
    def select_users(self, round, num_subset_users):
        np.random.seed(round)
        return np.random.choice(self.users, num_subset_users, replace=False)     
    
    def initialize_or_add(self, dest, src):
    
        for key, value in src.items():
            if key in dest:
                dest[key] = [x + y for x, y in zip(dest[key], value)]
            else:
                dest[key] = value.copy()  # Initialize with a copy of the first list


    def eval_train(self, t):
        avg_loss = 0.0
        avg_distance = 0.0
        avg_mae = 0.0
        accumulator = {}
        for c in self.selected_users:
            loss, distance, c_dict, mae = c.train_evaluation(self.global_model.parameters(), t)
            avg_loss += (1/len(self.selected_users))*loss
            avg_distance += (1/len(self.selected_users))*distance
            avg_mae += (1/len(self.selected_users))*mae
            if c_dict:  # Check if test_dict is not None or empty
                self.initialize_or_add(accumulator, c_dict)
        average_dict = {key: [x / len(self.selected_users) for x in value] for key, value in accumulator.items()}

        self.wandb.log(data={ "global_train_loss" : avg_loss})

        self.global_train_metric.append(average_dict)
        self.global_train_loss.append(avg_loss)
        self.global_train_distance.append(avg_distance)
        self.global_train_mae.append(avg_mae)

                    
        print(f"Global round {t} Global Train loss {avg_loss} avg distance {avg_distance}") 
        print(f"Train Performance metric : {average_dict}")
        print(f"Train global mae : {avg_mae}")



    
  
    def eval_test(self, t):
        avg_loss = 0.0
        avg_distance = 0.0
        accumulator = {}
        avg_mae = 0.0
        for c in self.selected_users:
            loss, distance, c_dict, mae = c.test(self.global_model.parameters(), t)
            avg_loss += (1/len(self.selected_users))*loss
            avg_distance += (1/len(self.selected_users))*distance
            avg_mae += (1/len(self.selected_users))*mae
            if c_dict:  # Check if test_dict is not None or empty
                self.initialize_or_add(accumulator, c_dict)
        average_dict = {key: [x / len(self.selected_users) for x in value] for key, value in accumulator.items()}

        
        self.wandb.log(data={ "global_val_loss" : avg_loss})
        self.wandb.log(data={ "global_mae" : avg_mae})
        self.wandb.log(data={ "global_Accuracy"  : average_dict['Accuracy'][0]})
        self.wandb.log(data={ "global_precision"  : average_dict['Precision'][0]})
        self.wandb.log(data={ "global_Recall" : average_dict['Recall'][0]})
        self.wandb.log(data={ "global_f1"  : average_dict['f1'][0]})
        
        self.global_test_metric.append(average_dict)
        self.global_test_loss.append(avg_loss)
        self.global_test_distance.append(avg_distance)
        self.global_test_mae.append(avg_mae)
            
                    
        print(f"Global round {t} Global Test loss {avg_loss} avg distance {avg_distance}") 
        print(f"Test Performance metric : {average_dict}")
        print(f"Test global mae : {avg_mae}")



    

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
        avg_f1 = 0.0
        test_avg_mae = 0.0
        test_avg_cmae = 0.0
        test_avg_f1 = 0.0
        for c in self.users:
            info_prec, info_rec, info_f1, info_cmae, info_mae, results = c.test_global_model_val(self.global_model)
            test_info_prec, test_info_rec, test_info_f1, test_info_cmae, test_info_mae, test_results = c.test_global_model_test(self.global_model)

            # print(f"info_prec {info_prec}, info_rec {info_rec}, info_f1 {info_f1}, info_cmae {info_cmae}, info_mae {info_mae}")
            
            avg_mae += (1/len(self.selected_users))*info_mae
            avg_cmae += (1/len(self.selected_users))*info_cmae
            # avg_loss += (1/len(self.select_users))*loss
            avg_f1 += (1/len(self.selected_users))*info_f1
            
            test_avg_mae += (1/len(self.selected_users))*test_info_mae
            test_avg_cmae += (1/len(self.selected_users))*test_info_cmae
            # test_avg_loss += (1/len(self.select_users))*test_loss
            test_avg_f1 += (1/len(self.selected_users))*test_info_f1

            
        
        print(f"\n Global round {t} : Global val f1: {avg_f1} Global val cmae {avg_cmae} global val mae : {avg_mae} \n")
        print(f"\n Global round {t} : Global test f1: {test_avg_f1} Global test cmae {test_avg_cmae} global test mae : {test_avg_mae} \n")

        self.save_global_model(t, avg_cmae)
    
    def evaluate_local(self, t):
        val_avg_mae = 0.0
        val_avg_cmae = 0.0
        val_avg_f1 = 0.0
        test_avg_mae = 0.0
        test_avg_cmae = 0.0
        test_avg_f1 = 0.0
        for c in self.users:
            info_prec, info_rec, info_f1, info_cmae, info_mae, _ = c.test_local_model_val()
            test_info_prec, test_info_rec, test_info_f1, test_info_cmae, test_info_mae, _ = c.test_local_model_test()
            
            # print(f"info_prec {info_prec}, info_rec {info_rec}, info_f1 {info_f1}, info_cmae {info_cmae}, info_mae {info_mae}")
            
            val_avg_mae += (1/len(self.selected_users))*info_mae
            val_avg_cmae += (1/len(self.selected_users))*info_cmae
            val_avg_f1 += (1/len(self.selected_users))*info_f1

            test_avg_mae += (1/len(self.selected_users))*test_info_mae
            test_avg_cmae += (1/len(self.selected_users))*test_info_cmae
            test_avg_f1 += (1/len(self.selected_users))*test_info_f1
        
        print(f"\033[92m\n Global round {t} : Local val cmae {val_avg_cmae} Local val mae : {val_avg_mae} \n\033[0m")   # Green
        print(f"\033[93m\n Global round {t} : Local Test cmae {test_avg_cmae} Local Test mae : {test_avg_mae} \n\033[0m")  # Yellow



    
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

    def convert_numpy(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def save_results(self):
        for user in self.users:
            val_dict = user.val_round_result_dict
            test_dict = user.test_round_result_dict
            global_val_dict = user.val_global_round_result_dict
            global_test_dict = user.test_global_round_result_dict
        

            user_id = str(user.id)
            val_json_path = f"results/client_level/exp_{self.exp_no}_model_name_{self.model_name}_FedDC/local_val/user_{user_id}_val_round_results.json"
            test_json_path = f"results/client_level/exp_{self.exp_no}_model_name_{self.model_name}_FedDC/local_test/user_{user_id}_test_round_results.json"
            val_global_json_path = f"results/client_level/exp_{self.exp_no}_model_name_{self.model_name}_FedDC/global_val/user_{user_id}_val_round_results.json"
            test_global_json_path = f"results/client_level/exp_{self.exp_no}_model_name_{self.model_name}_FedDC/global_test/user_{user_id}_test_round_results.json"

        
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





    def train(self):
        loss = []
        d=2
        b=3
        for t in trange(self.num_glob_iters, desc=f" exp no : {self.exp_no} number of clients: {len(self.users)} / Global Rounds :"):
            self.selected_users = self.select_users(t, len(self.users)).tolist()
            if t%b == b-1:
                self.send_parameters()
            for user in tqdm(self.selected_users, desc=f"selected users {len(self.selected_users)}"):
                 user.train(t)
            
        
            if t%d == d-1:
                # (self.selected_users)
                # assume self.selected_users is a list of User objects
                user_list = self.selected_users.copy()
                random.shuffle(user_list)

                exchange_dict = {}

                # pair consecutive users
                for i in range(0, len(user_list) - 1, 2):
                    exchange_dict[user_list[i]] = user_list[i+1]

                # if odd user, last one sits out (no entry in dict)
                if len(user_list) % 2 != 0:
                    print(f"User {user_list[-1].id} is sitting out this daisy-chain round.")
                # print(f"Exchange dict : {exchange_dict}")
                for user, exchange_user in exchange_dict.items():
                    # print(f"User {user.id} is exchanging parameters with User {exchange_user.id}")
                    # print(f"exchange user : {exchange_user}")
                    user.exchange_parameters(exchange_user)
                
            elif t%b == b-1:
                self.aggregate_parameters()
            
            self.evaluate_local(t)
            self.evaluate_global(t)
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
            #print("User ID", user.id)
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
        
