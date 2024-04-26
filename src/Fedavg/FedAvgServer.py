import torch
import os
import h5py
from src.Fedavg.UserFedAvg import UserAvg
#from src.utils.data_process import read_data, read_user_data
import numpy as np
import copy
from datetime import date
from tqdm import trange
from tqdm import tqdm
import numpy as np
from sklearn.cluster import SpectralClustering
import time
# Implementation for FedAvg Server
import matplotlib.pyplot as plt
import statistics
class FedAvg():
    def __init__(self,device, args, exp_no, current_directory):
                
        self.device = device
        self.num_glob_iters = args.num_global_iters
        self.local_iters = args.local_iters
        self.batch_size = args.batch_size
        self.learning_rate = args.alpha
        
        self.user_ids = args.user_ids
        print(f"user ids : {self.user_ids}")
        self.total_users = len(self.user_ids)
        print(f"total users : {self.total_users}")
        self.num_users = self.total_users * args.users_frac    #selected users
        self.total_train_samples = 0
        self.exp_no = exp_no
        self.algorithm = args.algorithm
        
        self.current_directory = current_directory

        self.country = args.country
        if args.country == "japan":
            self.user_ids = args.user_ids[2]
        else:
            self.user_ids = args.user_ids[1]

  
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
        
        self.minimum_test_loss = 0.0
        
        for i in trange(self.total_users, desc="Data distribution to clients"):
            user = UserAvg(device, args, int(self.user_ids[i]), exp_no, current_directory)
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
        if glob_iter == 0:
            self.minimum_test_loss = self.global_test_loss[glob_iter]
        else:
            print(self.global_test_loss[glob_iter])
            print(self.minimum_test_loss)
            if self.global_test_loss[glob_iter] < self.minimum_test_loss:
                self.minimum_test_loss = self.global_test_loss[glob_iter]
                model_path = self.current_directory + "/models/" + self.global_model_name + "/" + self.algorithm + "/global_model/"
                print(model_path)
                # input("press")
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                print(f"saving global model at round {glob_iter}")
                torch.save(self.sv_global_model, os.path.join(model_path, "server_" + ".pt"))

    def select_users(self, round, subset_users):

        if subset_users == len(self.users):
            return self.users
        elif  subset_users < len(self.users):
         
            np.random.seed(round)
            return np.random.choice(self.users, subset_users, replace=False)  # , p=pk)

        else: 
            assert (self.subset_users > len(self.users))
            
    def test_error_and_loss(self):
        
        accs = []
        losses = []
        precisions = []
        recalls = []
        f1s = []
        
        for c in self.users:
            accuracy, loss, precision, recall, f1 = c.test(self.global_model.parameters())
            # tot_correct.append(ct * 1.0)
            # num_samples.append(ns)
            accs.append(accuracy)
            losses.append(loss)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            
        return accs, losses, precisions, recalls, f1s

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

        self.global_train_metric.append(average_dict)
        self.global_train_loss.append(avg_loss)
        self.global_train_distance.append(avg_distance)
        self.global_train_mae.append(mae)

                    
        print(f"Global round {t} avg loss {avg_loss} avg distance {avg_distance}") 
        

    
  
    def eval_test(self, t):
        avg_loss = 0.0
        avg_distance = 0.0
        accumulator = {}
        for c in self.selected_users:
            loss, distance, c_dict, mae= c.test(self.global_model.parameters(), t)
            avg_loss += (1/len(self.selected_users))*loss
            avg_distance += (1/len(self.selected_users))*distance
            if c_dict:  # Check if test_dict is not None or empty
                self.initialize_or_add(accumulator, c_dict)
        average_dict = {key: [x / len(self.selected_users) for x in value] for key, value in accumulator.items()}

        self.global_test_metric.append(average_dict)
        self.global_test_loss.append(avg_loss)
        self.global_test_distance.append(avg_distance)
        self.global_test_mae.append(mae)
            
                    
        print(f"Global round {t} avg loss {avg_loss} avg distance {avg_distance}") 
        


    def evaluate(self, t):
        self.eval_test(t)
        self.eval_train(t)

    def save_results(self):
       
        file = "exp_no_" + str(self.exp_no) + "_GR_" + str(self.num_glob_iters) + "_BS_" + str(self.batch_size)
        
        print(file)
       
        directory_name = str(self.algorithm) + "/" +"h5" + "/global_model/"
        # Check if the directory already exists
        if not os.path.exists(self.current_directory + "/results/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/results/" + directory_name)



        with h5py.File(self.current_directory + "/results/" + directory_name + "/" + '{}.h5'.format(file), 'w') as hf:
            hf.create_dataset('Global rounds', data=self.num_glob_iters)
            hf.create_dataset('Local iters', data=self.local_iters)
            hf.create_dataset('Learning rate', data=self.learning_rate)
            hf.create_dataset('Batch size', data=self.batch_size)
           
            hf.create_dataset('global_test_metric',data=self.global_test_metric)
            hf.create_dataset('global_test_loss', data=self.global_test_loss)
            hf.create_dataset('global_test_distance', data=self.global_test_distance)
            hf.create_dataset('global_test_mae', data=self.global_test_mae)

            hf.create_dataset('global_train_metric',data=self.global_train_metric)
            hf.create_dataset('global_train_loss', data=self.global_train_loss)
            hf.create_dataset('global_train_distance', data=self.global_train_distance)
            hf.create_dataset('global_train_mae', data=self.global_train_mae)

            hf.close()

    def save_global_model(self, t):
        file = "_exp_no_" + str(self.exp_no) + "_GR_" + str(t) 
        
        print(file)
       
        directory_name = str(self.algorithm) + "/" +"global_model"
        # Check if the directory already exists
        if not os.path.exists(self.current_directory + "/models/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/models/"+ directory_name)
        
        torch.save(self.global_model,self.current_directory + "/models/"+ directory_name + "/" + file + ".pt")




    def train(self):
        loss = []
        
        for glob_iter in trange(self.num_glob_iters, desc="Global Rounds"):
            self.send_parameters()
            self.selected_users = self.select_users(glob_iter, self.num_users)
            list_user_id = []
            for user in self.selected_users:
                list_user_id.append(user.id)
            print(f"Exp no{self.exp_no} : users selected for global iteration {glob_iter} are : {list_user_id}")

            for user in tqdm(self.selected_users, desc="running clients"):
                user.train()  # * user.train_samples

            self.aggregate_parameters()
            self.evaluate(glob_iter)
            self.save_global_model(glob_iter)
        self.save_results()
