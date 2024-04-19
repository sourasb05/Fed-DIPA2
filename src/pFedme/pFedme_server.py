import torch
import os
import h5py
from src.pFedme.pFedme_user import pFedme_user
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

class pFedme():
    def __init__(self,device, model, args, exp_no, current_directory):
                
        self.device = device
        self.num_glob_iters = args.num_global_iters
        self.local_iters = args.local_iters
        self.batch_size = args.batch_size
        self.learning_rate = args.alpha
        self.eta = args.eta
        self.beta = args.beta
        self.user_ids = args.user_ids
        print(f"user ids : {self.user_ids}")
        self.total_users = len(self.user_ids)
        print(f"total users : {self.total_users}")
        self.num_users = self.total_users * args.users_frac    #selected users
        self.total_train_samples = 0
        self.exp_no = exp_no
        self.current_directory = current_directory
        self.algorithm = args.algorithm
        self.target = args.target
        #self.c = [[] for _ in range(args.num_teams)]


        """
        Global model
        
        """

        self.global_model = copy.deepcopy(model)
        # print(self.global_model)
        self.global_model.to(self.device)
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

        """
        Local model evaluation
        """

        self.local_train_acc = []
        self.local_test_acc  = []
        self.local_train_loss  = []
        self.local_test_loss  = []
        self.local_precision = []
        self.local_recall = []
        self.local_f1score = []

        self.minimum_clust_loss = 0.0
        self.minimum_global_loss = 0.0
        
        # data = read_data(args, current_directory)
        # self.tot_users = len(data[0])
        # print(self.tot_users)

        for i in trange(self.total_users, desc="Data distribution to clients"):
            # id, train, test = read_user_data(i, data)
            user = pFedme_user(device, self.global_model, args, self.user_ids[i], exp_no, current_directory)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        print("Finished creating Fedmem server.")

        
    def send_global_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.global_model)
   
    def add_parameters(self, user, ratio):
        for server_param, user_param in zip(self.global_model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def global_update(self):
        assert (self.users is not None and len(self.users) > 0)

        previous_param = copy.deepcopy(list(self.global_model.parameters()))
        
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.global_model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data
        
   
    def select_users(self, round, subset_users):
        np.random.seed(round)
        return np.random.choice(self.users, subset_users, replace=False)

        
   
    # Save loss, accurancy to h5 fiel
    def save_results(self):
        file = "_exp_no_" + str(self.exp_no) + "_GR_" + str(self.num_glob_iters) + "_BS_" + str(self.batch_size)
        
        print(file)
       
        directory_name = str(self.global_model_name) + "/" + str(self.algorithm) + "/" + str(self.target) + "/" + str(self.num_users) + "/" +"h5"
        # Check if the directory already exists
        if not os.path.exists(self.current_directory + "/results/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/results/" + directory_name)



        with h5py.File(self.current_directory + "/results/" + directory_name + "/" + '{}.h5'.format(file), 'w') as hf:
            hf.create_dataset('exp_no', data=self.exp_no)
            hf.create_dataset('Global rounds', data=self.num_glob_iters)
            hf.create_dataset('Local iters', data=self.local_iters)
            hf.create_dataset('Learning rate', data=self.learning_rate)
            hf.create_dataset('eta', data=self.eta)
            hf.create_dataset('beta', data=self.beta)
            hf.create_dataset('Batch size', data=self.batch_size)
            # hf.create_dataset('clusters', data=self.clusters_list)
            hf.create_dataset('global_test_loss', data=self.global_test_loss)
            hf.create_dataset('global_train_loss', data=self.global_train_loss)
            hf.create_dataset('global_test_accuracy', data=self.global_test_acc)
            hf.create_dataset('global_train_accuracy', data=self.global_train_acc)
            hf.create_dataset('global_precision', data=self.global_precision)
            hf.create_dataset('global_recall', data=self.global_recall)
            hf.create_dataset('global_f1score', data=self.global_f1score)
        
            hf.create_dataset('per_test_loss', data=self.local_test_loss)
            hf.create_dataset('per_train_loss', data=self.local_train_loss)
            hf.create_dataset('per_test_accuracy', data=self.local_test_acc)
            hf.create_dataset('per_train_accuracy', data=self.local_train_acc)
            hf.create_dataset('per_precision', data=self.local_precision)
            hf.create_dataset('per_recall', data=self.local_recall)
            hf.create_dataset('per_f1score', data=self.local_f1score)

            hf.close()
        
    def save_global_model(self, t): #, cm):
        # cm_df = pd.DataFrame(cm)
        file_cm = "_exp_no_" + str(self.exp_no) + "_confusion_matrix" 
        file = "_exp_no_" + str(self.exp_no) + "_model" 
        
        print(file)
       
        directory_name = str(self.global_model_name) + "/" + str(self.algorithm) + "/" + str(self.target) + "/" + str(self.num_users) + "/" +"global_model"
        # Check if the directory already exists
        if not os.path.exists(self.current_directory + "/models/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/models/"+ directory_name)
        
        if not os.path.exists(self.current_directory + "/models/confusion_matrix/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/models/confusion_matrix/"+ directory_name)
        torch.save(self.global_model,self.current_directory + "/models/"+ directory_name + "/" + file + ".pt")
        # cm_df.to_csv(self.current_directory + "/models/confusion_matrix/"+ directory_name + "/" + file_cm + ".csv", index=False)
    
    
    
    def test_error_and_loss(self, evaluate_model):
        # num_samples = []
        # tot_correct = []
        accs = []
        losses = []
        precisions = []
        recalls = []
        f1s = []
        cms = []
        if evaluate_model == 'global':
            for c in self.selected_users:
                accuracy, loss, precision, recall, f1, cm = c.test(self.global_model.parameters())
               
                accs.append(accuracy)
                losses.append(loss)
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
                cms.append(cm)
            
        elif evaluate_model == 'local':
            for c in self.selected_users:
                accuracy, loss, precision, recall, f1, cm = c.test_local()
                accs.append(accuracy)
                losses.append(loss)
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
                cms.append(cm)
            
        return accs, losses, precisions, recalls, f1s, cms

    def train_error_and_loss(self, evaluate_model):
        accs = []
        losses = []
        
        if evaluate_model == 'global':
            for c in self.selected_users:
                accuracy, loss = c.train_error_and_loss(self.global_model.parameters())
                
                accs.append(accuracy)
                losses.append(loss)
        elif evaluate_model == 'local':
            for c in self.selected_users:
                accuracy, loss = c.train_error_and_loss_local()
                accs.append(accuracy)
                losses.append(loss)
    
        return accs, losses


    def evaluate(self, t):
        
        evaluate_model = "global"
        test_accs, test_losses, precisions, recalls, f1s, cms = self.test_error_and_loss(evaluate_model)
        train_accs, train_losses  = self.train_error_and_loss(evaluate_model)
        
        self.global_train_acc.append(statistics.mean(train_accs))
        self.global_test_acc.append(statistics.mean(test_accs))
        self.global_train_loss.append(statistics.mean(train_losses))
        self.global_test_loss.append(statistics.mean(test_losses))
        self.global_precision.append(statistics.mean(precisions))
        self.global_recall.append(statistics.mean(recalls))
        self.global_f1score.append(statistics.mean(f1s))
        """try:
            cm_sum
        except NameError:
            cm_sum = np.zeros(cms[0].shape)
        for cm in cms:
            cm_sum += 1/len(cms)*cm
"""

        print(f"Global Trainning Accurancy: {self.global_train_acc[t]}" )
        print(f"Global Trainning Loss: {self.global_train_loss[t]}")
        print(f"Global test accurancy: {self.global_test_acc[t]}")
        print(f"Global test_loss: {self.global_test_loss[t]}")
        print(f"Global Precision: {self.global_precision[t]}")
        print(f"Global Recall: {self.global_recall[t]}")
        print(f"Global f1score: {self.global_f1score[t]}")


        if t == 0 and self.minimum_global_loss == 0.0:
            self.minimum_global_loss = self.global_test_loss[0]
        else:
            if self.global_test_loss[t] < self.minimum_global_loss:
                self.minimum_global_loss = self.global_test_loss[t]
                # print(f"new minimum loss of local model at client {self.id} found at global round {t} local epoch {epoch}")
                self.save_global_model(t) #, cm_sum)
                

    def evaluate_localmodel(self, t):
        evaluate_model = "local"
        test_accs, test_losses, precisions, recalls, f1s, cms = self.test_error_and_loss(evaluate_model)
        train_accs, train_losses  = self.train_error_and_loss(evaluate_model)
        
        self.local_train_acc.append(statistics.mean(train_accs))
        self.local_test_acc.append(statistics.mean(test_accs))
        self.local_train_loss.append(statistics.mean(train_losses))
        self.local_test_loss.append(statistics.mean(test_losses))
        self.local_precision.append(statistics.mean(precisions))
        self.local_recall.append(statistics.mean(recalls))
        self.local_f1score.append(statistics.mean(f1s))
        """try:
            cm_sum
        except NameError:
            cm_sum = np.zeros(cms[0].shape)
        for cm in cms:
            cm_sum += 1/len(cms)*cm
        """

        print(f"Local Trainning Accurancy: {self.local_train_acc[t]}" )
        print(f"Local Trainning Loss: {self.local_train_loss[t]}")
        print(f"Local test accurancy: {self.local_test_acc[t]}")
        print(f"Local test_loss: {self.local_test_loss[t]}")
        print(f"Local Precision: {self.local_precision[t]}")
        print(f"Local Recall: {self.local_recall[t]}")
        print(f"Local f1score: {self.local_f1score[t]}")

    
    
    def plot_per_result(self):
        
        fig, ax = plt.subplots(1,2, figsize=(12,6))

        ax[0].plot(self.local_train_acc, label= "Train_accuracy")
        ax[0].plot(self.local_test_acc, label= "Test_accuracy")
        ax[0].set_xlabel("Global Iteration")
        ax[0].set_ylabel("accuracy")
        ax[0].set_xticks(range(0, self.num_glob_iters, int(self.num_glob_iters/5)))#
        ax[0].legend(prop={"size":12})
        ax[1].plot(self.local_train_loss, label= "Train_loss")
        ax[1].plot(self.local_test_loss, label= "Test_loss")
        ax[1].set_xlabel("Global Iteration")
        #ax[1].set_xscale('log')
        ax[1].set_ylabel("Loss")
        #ax[1].set_yscale('log')
        ax[1].set_xticks(range(0, self.num_glob_iters, int(self.num_glob_iters/5)))
        ax[1].legend(prop={"size":12})
        
        directory_name = str(self.global_model_name) + "/" + str(self.algorithm) + "/" + str(self.target) + "/" + self.cluster_type  + "/" + str(self.num_users) + "/plot/personalized"
        # Check if the directory already exists
        if not os.path.exists(self.current_directory + "/results/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/results/" + directory_name)

        plt.draw()
       
        plt.savefig(self.current_directory + "/results/" + directory_name  + "/exp_no_" + str(self.exp_no) + "_global_iters_" + str(self.num_glob_iters) + '.png')

        # Show the graph
        plt.show()



    def plot_global_result(self):
        
        # print(self.global_train_acc)

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
        
        directory_name = str(self.global_model_name) + "/" + str(self.algorithm) +  "/" + str(self.target) + "/" + self.cluster_type  + "/" + str(self.num_users) + "/" +"plot/global"
        # Check if the directory already exists
        if not os.path.exists(self.current_directory + "/results/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/results/" + directory_name)

        plt.draw()
       
        plt.savefig(self.current_directory + "/results/" + directory_name  + "/exp_no_" + str(self.exp_no) + "_global_iters_" + str(self.num_glob_iters) + '.png')

        # Show the graph
        plt.show()
        
    
    def train(self):
        loss = []
        
        for t in trange(self.num_glob_iters, desc=f" exp no : {self.exp_no}  number of clients: {self.num_users} Global Rounds :"):
            
            self.send_global_parameters()
            
            self.selected_users = self.select_users(t, int(self.num_users)).tolist()
            list_user_id = []
            for user in self.selected_users:
                list_user_id.append(user.id)

            for user in tqdm(self.selected_users, desc="running selected clients"):
                user.train()  # * user.train_samples
        
            self.global_update()

        
            self.evaluate_localmodel(t)
            self.evaluate(t)

        self.save_results()
        # self.plot_per_result()
        # self.plot_global_result()