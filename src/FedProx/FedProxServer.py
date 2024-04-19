import torch
import os
import h5py
from src.FedProx.UserProxAvg import UserProx
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


class FedProx():
    def __init__(self,device, model, args, exp_no, current_directory):
                
        self.device = device
        self.num_glob_iters = args.num_global_iters
        self.local_iters = args.local_iters
        self.batch_size = args.batch_size
        self.learning_rate = args.alpha
        self.lamda = args.lambda_1
        self.user_ids = args.user_ids
        print(f"user ids : {self.user_ids}")
        self.total_users = len(self.user_ids)
        print(f"total users : {self.total_users}")
        self.num_users = self.total_users * args.users_frac    #selected users
        self.num_teams = args.num_teams
        self.total_train_samples = 0
        self.exp_no = exp_no
        self.n_clusters = args.num_teams
        self.algorithm = args.algorithm
        self.current_directory = current_directory
        self.target = args.target
        """
        Global model
        
        """

        self.global_model = copy.deepcopy(model)
        self.sv_global_model = copy.deepcopy(model)
        self.global_model_name = args.model_name
  

        self.users = []
        self.selected_users = []
        self.global_train_acc = []
        self.global_train_loss = [] 
        self.global_test_acc = [] 
        self.global_test_loss = []
        self.global_precision = []
        self.global_recall = []
        self.global_f1score = []
        
        self.minimum_test_loss = 0.0
        # data = read_data(args, current_directory)
        # self.tot_users = len(data[0])
        # print(self.tot_users)

        for i in trange(self.total_users, desc="Data distribution to clients"):
            # id, train, test = read_user_data(i, data)
            user = UserProx(device, model, args, int(self.user_ids[i]), exp_no)
            self.users.append(user)
            self.total_train_samples += user.train_samples

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
        # if(self.num_users = self.to)
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

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))

    def select_users(self, round, subset_users):
        # selects num_clients clients weighted by number of samples from possible_clients
        # self.selected_users = []
        # print("num_users :",num_users)
        # print(" size of user per group :",len(self.users[grp]))
        if subset_users == len(self.users):
            # print("All users are selected")
            # print(self.users[grp])
            return self.users
        elif  subset_users < len(self.users):
         
            np.random.seed(round)
            return np.random.choice(self.users, subset_users, replace=False)  # , p=pk)

        else: 
            assert (self.subset_users > len(self.users))
            # print("number of selected users are greater than total users")
    


    
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

    def train_error_and_loss(self):
        accs = []
        losses = []
        for c in self.users:
            accuracy, loss = c.train_error_and_loss(self.global_model.parameters())
            accs.append(accuracy)
            losses.append(loss)

        
        return accs, losses


    def evaluate(self, t):

        test_accs, test_losses, precisions, recalls, f1s = self.test_error_and_loss()
        train_accs, train_losses  = self.train_error_and_loss()
        

        self.global_train_acc.append(statistics.mean(train_accs))
        self.global_test_acc.append(statistics.mean(test_accs))
        self.global_train_loss.append(statistics.mean(train_losses))
        self.global_test_loss.append(statistics.mean(test_losses))
        self.global_precision.append(statistics.mean(precisions))
        self.global_recall.append(statistics.mean(recalls))
        self.global_f1score.append(statistics.mean(f1s))
        

        print(f"Global Trainning Accurancy: {self.global_train_acc[t]}" )
        print(f"Global Trainning Loss: {self.global_train_loss[t]}")
        print(f"Global test accurancy: {self.global_test_acc[t]}")
        print(f"Global test_loss: {self.global_test_loss[t]}")
        print(f"Global Precision: {self.global_precision[t]}")
        print(f"Global Recall: {self.global_recall[t]}")
        print(f"Global f1score: {self.global_f1score[t]}")
    

    def plot_result(self):
        
        print(self.global_train_acc)

        fig, ax = plt.subplots(1,2, figsize=(12,6))

        ax[0].plot(self.global_train_acc, label= "Train_accuracy")
        ax[0].plot(self.global_test_acc, label= "Test_accuracy")
        ax[0].set_xlabel("Global Iteration")
        ax[0].set_ylabel("accuracy")
        ax[0].set_xticks(range(0, self.num_glob_iters, int(self.num_glob_iters/5)))#
        ax[0].legend(prop={"size":12})
        ax[1].plot(self.global_train_loss, label= "Train_loss")
        ax[1].plot(self.global_test_loss, label= "Test_loss")
        ax[1].set_xlabel("Global Iteration")
        #ax[1].set_xscale('log')
        ax[1].set_ylabel("Loss")
        #ax[1].set_yscale('log')
        ax[1].set_xticks(range(0, self.num_glob_iters, int(self.num_glob_iters/5)))
        ax[1].legend(prop={"size":12})
        
        directory_name = str(self.global_model_name) + "/" + str(self.algorithm) + "/" +"plot"
        # Check if the directory already exists
        if not os.path.exists(self.current_directory + "/results/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/results/" + directory_name)

        plt.draw()
       
        plt.savefig(self.current_directory + "/results/" + directory_name  + "/_global_iters_" + str(self.num_glob_iters) + '.png')

        # Show the graph
        plt.show()

        # Save loss, accurancy to h5 fiel
    def save_results(self):
       
        file = "_exp_no_" + str(self.exp_no) + "_GR_" + str(self.num_glob_iters) + "_BS_" + str(self.batch_size)
        
        print(file)
       
        directory_name = str(self.global_model_name) + "/" + str(self.algorithm) + "/" +"h5" + "/" + str(self.target)
        # Check if the directory already exists
        if not os.path.exists(self.current_directory + "/results/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/results/" + directory_name)

        # print("Global Trainning Accurancy: ", self.global_train_acc)
        # print("Global Trainning Loss: ", self.global_train_loss)
        # print("Global test accurancy: ", self.global_test_acc)
        # print("Global test_loss:",self.global_test_loss)


        with h5py.File(self.current_directory + "/results/" + directory_name + "/" + '{}.h5'.format(file), 'w') as hf:
            hf.create_dataset('Global rounds', data=self.num_glob_iters)
            hf.create_dataset('Local iters', data=self.local_iters)
            hf.create_dataset('Learning rate', data=self.learning_rate)
            hf.create_dataset('Lambda', data=self.lamda)
            hf.create_dataset('Batch size', data=self.batch_size)
            hf.create_dataset('global_test_loss', data=self.global_test_loss)
            hf.create_dataset('global_train_loss', data=self.global_train_loss)
            hf.create_dataset('global_test_accuracy', data=self.global_test_acc)
            hf.create_dataset('global_train_accuracy', data=self.global_train_acc)
            hf.create_dataset('global_precision', data=self.global_precision)
            hf.create_dataset('global_recall', data=self.global_recall)
            hf.create_dataset('global_f1score', data=self.global_f1score)
            
            hf.close()

    def save_global_model(self, t):
        file = "_exp_no_" + str(self.exp_no) + "_GR_" + str(t) 
        
        print(file)
       
        directory_name = str(self.global_model_name) + "/" + str(self.algorithm) + "/" +"global_model"
        # Check if the directory already exists
        if not os.path.exists(self.current_directory + "/models/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/models/"+ directory_name)
        
        torch.save(self.global_model,self.current_directory + "/models/"+ directory_name + "/" + file + ".pt")


    def train(self):
        loss = []
        
        for glob_iter in trange(self.num_glob_iters, desc=f" Exp no {self.exp_no} : Global Rounds"):
            self.send_parameters()
            self.selected_users = self.select_users(glob_iter, self.num_users)
            list_user_id = []
            for user in self.selected_users:
                list_user_id.append(user.id)
            print(f"Exp no{self.exp_no} : users selected for global iteration {glob_iter} are : {list_user_id}")

            for user in tqdm(self.selected_users, desc="running clients"):
                user.train(self.global_model)  # * user.train_samples

            self.aggregate_parameters()
            # self.save_global_model(glob_iter)
            self.evaluate(glob_iter)
            # self.save_model(glob_iter)
        self.save_results()
        
        #self.plot_result()

