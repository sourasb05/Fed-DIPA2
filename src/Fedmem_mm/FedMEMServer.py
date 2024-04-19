import torch
import os
import h5py
from src.Fedmem_mm.FedMEMUser import Fedmem_user
import numpy as np
import copy
from datetime import date
from tqdm import trange
from tqdm import tqdm
import numpy as np
import pandas as pd
# from sklearn.cluster import SpectralClustering
import time
from sklearn.cluster import KMeans
# Implementation for FedAvg Server
import matplotlib.pyplot as plt
import statistics


class Fedmem_mm():
    def __init__(self,device, args, exp_no, current_directory):
                
        self.device = device
        self.local_iters = args.local_iters
        self.batch_size = args.batch_size
        self.learning_rate = args.alpha
        self.user_ids = args.user_ids
        self.total_users = len(self.user_ids)
        self.num_users = self.total_users * args.users_frac    #selected users
        self.total_train_samples = 0
        self.exp_no = exp_no
        self.current_directory = current_directory
        self.algorithm = args.algorithm
        self.cluster = args.cluster
        self.users = []
        self.selected_users = []

        """
        Local model evaluation
        """
        self.local_test_acc  = []
        self.local_test_loss  = []
        self.local_f1score = []
 
        for i in trange(self.total_users, desc="Data distribution to clients"):
            # id, train, test = read_user_data(i, data)
            user = Fedmem_user(device, args, self.user_ids[i], exp_no, current_directory)
            self.users.append(user)
            self.total_train_samples += user.train_samples

    def send_global_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.global_model.parameters())
    
    
    def save_results(self):
        file = "_exp_no_" + str(self.exp_no) + "_LR_" + str(self.local_iters) + "_BS_" + str(self.batch_size)  + "_num_user_" + str(self.num_users)
        
        print(file)
       
        # directory_name = str(self.global_model_name) + "/" + str(self.algorithm) + "/data_silo_" + str(self.data_silo) + "/" + "target_" + str(self.target) + "/" + str(self.cluster_type) + "/" + str(self.num_users) + "/" "h5"
        
        directory_name = "Fedmem_MM_" + self.cluster

        # Check if the directory already exists
        if not os.path.exists(self.current_directory + "/results/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/results/" + directory_name)
        avg_highest_acc = 0
        accuracy_array = np.array([])
        for user in self.users:
            
            accuracy_array = np.append(accuracy_array, user.maximum_per_accuracy)
        print(f"len(self.users) : {len(self.users)}")
        
        
        avg_highest_acc = np.mean(accuracy_array)
        std_dev = np.std(accuracy_array, ddof=1) # ddof=1 for sample standard deviation, 0 for population


        with h5py.File(self.current_directory + "/results/" + directory_name + "/" + '{}.h5'.format(file), 'w') as hf:
            hf.create_dataset('exp_no', data=self.exp_no)
            hf.create_dataset('Local iters', data=self.local_iters)
            hf.create_dataset('Learning rate', data=self.learning_rate)
            hf.create_dataset('Batch size', data=self.batch_size)
            hf.create_dataset('num users', data=self.num_users)
            
            

            hf.create_dataset('maximum_per_test_accuracy', data=avg_highest_acc)
            hf.create_dataset('maximum_per_test_accuracy_list', data=accuracy_array)
            hf.create_dataset('std_dev', data=std_dev)

            for user in self.users:
                
                hf.create_dataset(f'client_{user.id}_accuracy_array', data=np.array(user.list_accuracy))
                hf.create_dataset(f'client_{user.id}_f1_array', data=np.array(user.list_f1))
                hf.create_dataset(f'client_{user.id}_val_loss_array', data=np.array(user.list_val_loss))
            # each_client_accuracy_array.append(user.list_accuracy)
            # each_client_f1_array.append(user.list_f1)
            # each_client_val_loss_array.append(user.list_val_loss)
            hf.close()
        
    def evaluate_localmodel(self):
        test_accs, f1s = self.test_error_and_loss()
        print(f"test_accs : {test_accs} f1s : {f1s}")
        self.local_test_acc.append(statistics.mean(test_accs))
        self.local_f1score.append(statistics.mean(f1s))

        print(f"Local test accurancy: {self.local_test_acc}")
        print(f"Local f1score: {self.local_f1score}")


    def test_error_and_loss(self):
        # num_samples = []
        # tot_correct = []
        accs = []
        f1s = []
        for c in self.users:
            accuracy, f1 = c.test_local()
            print(f"accuracy {accuracy} , f1 {f1}")
            accs.append(accuracy)
            f1s.append(f1)
           
        return accs, f1s

    
    def train(self):
        # self.selected_users = self.select_users(1, int(self.num_users)).tolist()
        list_user_id = []
        for user in self.users:   #self.selected_users:
            list_user_id.append(user.id)
        # print(f"selected users : {list_user_id}")
        for user in tqdm(self.users, desc=f"total selected users {len(self.users)}"):
            user.train()
        
        # self.evaluate_localmodel()
        # self.save_results()
        