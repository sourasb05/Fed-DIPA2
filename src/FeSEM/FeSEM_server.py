import torch
import os
import h5py
from src.FeSEM.FeSEM_user import User
import numpy as np
import copy
from datetime import date
from tqdm import trange
from tqdm import tqdm
import numpy as np
# from sklearn.cluster import SpectralClustering
import time
from sklearn.cluster import KMeans
from src.TrainModels.trainmodels import *
import torch.nn.init as init
import matplotlib.pyplot as plt
import statistics
import pprint
# Implementation for FeSEM Server
"""
Class FeSEM() is the server class.

"""

class FeSEM():
    def __init__(self,device, model, args, exp_no, current_directory):
                
        self.device = device
        self.num_glob_iters = args.num_global_iters
        self.local_iters = args.local_iters
        self.batch_size = args.batch_size
        self.learning_rate = args.alpha
        self.algorithm = args.algorithm
        self.user_ids = args.user_ids
        # print(f"user ids : {self.user_ids}")
        self.total_users = len(self.user_ids)
        # print(f"total users : {self.total_users}")
        self.num_users = self.total_users * args.users_frac
        self.n_clusters = args.num_teams
        self.target = args.target
        self.total_train_samples = 0
        self.exp_no = exp_no
        
        self.gamma = args.gamma # scale parameter for RBF kernel 
        self.current_directory = current_directory
        self.clusterhead_models = []
        self.cluster_dict = {}
        self.c = []


        """
        Global model
        
        """

        self.global_model = copy.deepcopy(model)     
        self.global_model.to(device)   
        self.global_model_name = args.model_name



        """
        Clusterhead models
        """
        for _ in range(self.n_clusters):
            self.c.append(copy.deepcopy(list(self.global_model.parameters())))
        
  

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
        Cluster head evaluation
        """

        self.cluster_train_acc = []
        self.cluster_test_acc = []
        self.cluster_train_loss = []
        self.cluster_test_loss = []
        self.cluster_precision = []
        self.cluster_recall = []
        self.cluster_f1score = []

       
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


        for i in range(self.total_users):
            user = User(device, self.global_model, args, self.user_ids[i], exp_no, current_directory)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        print("Finished creating FedSEM server.")

    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        
    def send_global_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.global_model)
    
    def send_cluster_parameters(self):
        for clust_id in range(self.n_clusters):
            users = np.array(self.cluster_dict[clust_id])
            for user in users:
                user.set_parameters(self.clusterhead_models[clust_id])

    def add_parameters(self, cluster_model, ratio):
        for server_param, cluster_param in zip(self.global_model.parameters(), cluster_model.parameters()):
            server_param.data = server_param.data + cluster_param.data.clone() * ratio


    def add_parameters_clusters(self, user, ratio, cluster_id):
        
        # model = self.clusterhead_models.parameters()

        for cluster_param, user_param in zip(self.c[cluster_id], user.get_parameters()):
            cluster_param.data = cluster_param.data + user_param.data.clone() * ratio


    def aggregate_clusterhead(self, r_ik_dict):

        for clust_id in range(self.n_clusters):
            for param in self.c[clust_id]:
                param.data = torch.zeros_like(param.data)
            users = np.array(self.cluster_dict[clust_id])
            # print(f"len(users) : {len(users)}")
            if len(users != 0):
                print("clust_id : {clust_id}")
                for user in users:
                    self.add_parameters_clusters(user, 1/len(users), clust_id)
            

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.global_model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))

    def select_users(self, round, subset_users):
        np.random.seed(round)
        return np.random.choice(self.users, subset_users, replace=False)

    def flatten_params(self, model_parameters):
        params = []
        for param in model_parameters:
            params.append(param.view(-1))
        return torch.cat(params)
    
    def find_similarity(self, params1, params2, ratio):
        similarity =  ratio * torch.norm(params1 - params2, p=2)                    

        # (similarity.item())
        return similarity


    def similarity_check(self):
        similarity_matrix = {}
        for k in range(self.n_clusters):
            if k not in similarity_matrix:
                similarity_matrix[k] = []
            # print(f"similarity checking cluster{k}")
            params1 = self.flatten_params(self.c[k])
            for user in self.selected_users:
                # if user != comp_user:
                    
                params2 = self.flatten_params(user.local_model.parameters())

                #print("user sample",user.train_samples)
                # print("total sample",self.total_train_samples)
                ratio = float(user.train_samples/self.total_train_samples)
                # print(ratio)
                similarity = self.find_similarity(params1, params2, ratio)


               #print("user_id["+ str(user.id)+"] and user_id["+str(comp_user.id)+"] = ",similarity.item())
                    

                similarity_matrix[k].extend([similarity.item()])
        # print(f"similarity matrix {similarity_matrix}")
                
        
        return similarity_matrix



    def find_rik_dict(self, similarity_dict):


        # # Step 1: Find the minimum in each column
        # min_in_clusters = [min(similarity_dict[row][i] for row in similarity_dict) for i in range(len(self.selected_users))]
        # print(f"min_in_clusters: ", min_in_clusters)
        # # Step 2: Create new dictionary with 1s and 0s
        # r_ik_dict = {}
        # for row in similarity_dict:
        #     r_ik_dict[row] = [1 if similarity_dict[row][i] == min_in_clusters[i] else 0 for i in range(len(similarity_dict[row]))]

        # print(f"r_ik_dict : {r_ik_dict}")
        # return r_ik_dict


        # Step 1: Find the index of the first minimum in each column
        min_indices = []
        for i in range(len(self.selected_users)):
            column_values = [similarity_dict[row][i] for row in similarity_dict]
            min_value = min(column_values)
            min_index = column_values.index(min_value)  # Get the index of the first occurrence of the min value
            min_indices.append(min_index)

        # Step 2: Create new dictionary with 1s for the first min in each column and 0s otherwise
        r_ik_dict = {}
        row_keys = list(similarity_dict.keys())  # Extracting keys to ensure consistent ordering
        for i, row_key in enumerate(row_keys):
            r_ik_dict[row_key] = []
            for j in range(len(self.selected_users)):
                column_values = [similarity_dict[row_keys[k]][j] for k in range(len(row_keys))]
                # Assign 1 if current row has the min value for this column, and it's the first occurrence
                if i == min_indices[j] and similarity_dict[row_key][j] == min(column_values):
                    r_ik_dict[row_key].append(1)
                else:
                    r_ik_dict[row_key].append(0)
        # print(f"r_ik_dict : {r_ik_dict}")
        return r_ik_dict
    def create_cluster(self, r_ik_dict):
        
        for row_key, row_values in r_ik_dict.items():
            self.cluster_dict[row_key] = [self.selected_users[i] for i, value in enumerate(row_values) if value == 1]

        # print(self.cluster_dict)
        


        
    
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
        else:
            for clust_id in range(self.n_clusters):
                users = np.array(self.cluster_dict[clust_id])
                if len(users)!= 0:
                    for c in users:
                        accuracy, loss, precision, recall, f1, cm = c.test(self.c[clust_id])
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
        else:
            for clust_id in range(self.n_clusters):
                users = np.array(self.cluster_dict[clust_id])
                if len(users != 0):
                    for c in users:
                        accuracy, loss = c.train_error_and_loss(self.c[clust_id])
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
        # print(f"Global Precision: {self.global_precision[t]}")
        # print(f"Global Recall: {self.global_recall[t]}")
        print(f"Global f1score: {self.global_f1score[t]}")


        if t == 0 and self.minimum_global_loss == 0.0:
            self.minimum_global_loss = self.global_test_loss[0]
        else:
            if self.global_test_loss[t] < self.minimum_global_loss:
                self.minimum_global_loss = self.global_test_loss[t]
                # print(f"new minimum loss of local model at client {self.id} found at global round {t} local epoch {epoch}")
                self.save_global_model(t) #, cm_sum)
                

    def evaluate_clusterhead(self, t):
        evaluate_model = "cluster"
        test_accs, test_losses, precisions, recalls, f1s, cms = self.test_error_and_loss(evaluate_model)
        train_accs, train_losses  = self.train_error_and_loss( evaluate_model)
        
        self.cluster_train_acc.append(statistics.mean(train_accs))
        self.cluster_test_acc.append(statistics.mean(test_accs))
        self.cluster_train_loss.append(statistics.mean(train_losses))
        self.cluster_test_loss.append(statistics.mean(test_losses))
        self.cluster_precision.append(statistics.mean(precisions))
        self.cluster_recall.append(statistics.mean(recalls))
        self.cluster_f1score.append(statistics.mean(f1s))
        
        print(f"Cluster Trainning Accurancy: {self.cluster_train_acc[t]}" )
        print(f"Cluster Trainning Loss: {self.cluster_train_loss[t]}")
        print(f"Cluster test accurancy: {self.cluster_test_acc[t]}")
        print(f"Cluster test_loss: {self.cluster_test_loss[t]}")
        # print(f"Cluster Precision: {self.cluster_precision[t]}")
        # print(f"Cluster Recall: {self.cluster_recall[t]}")
        print(f"Cluster f1score: {self.cluster_f1score[t]}")

        """
        if t == 0 and self.minimum_clust_loss == 0.0:
            self.minimum_clust_loss = self.cluster_test_loss[0]
        else:
            if self.cluster_test_loss[t] < self.minimum_clust_loss:
                self.minimum_clust_loss = self.cluster_test_loss[t]
                # print(f"new minimum loss of local model at client {self.id} found at global round {t} local epoch {epoch}")
                self.save_cluster_model(t)"""

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
            # hf.create_dataset('Lambda_1', data=self.lambda_1)
            # hf.create_dataset('Lambda_2', data=self.lambda_2)
            hf.create_dataset('Batch size', data=self.batch_size)
            # hf.create_dataset('clusters', data=self.clusters_list)
            hf.create_dataset('global_test_loss', data=self.global_test_loss)
            hf.create_dataset('global_train_loss', data=self.global_train_loss)
            hf.create_dataset('global_test_accuracy', data=self.global_test_acc)
            hf.create_dataset('global_train_accuracy', data=self.global_train_acc)
            hf.create_dataset('global_precision', data=self.global_precision)
            hf.create_dataset('global_recall', data=self.global_recall)
            hf.create_dataset('global_f1score', data=self.global_f1score)
            
            
            # hf.create_dataset('per_test_loss', data=self.local_test_loss)
            # hf.create_dataset('per_train_loss', data=self.local_train_loss)
            # hf.create_dataset('per_test_accuracy', data=self.local_test_acc)
            # hf.create_dataset('per_train_accuracy', data=self.local_train_acc)
            # hf.create_dataset('per_precision', data=self.local_precision)
            # hf.create_dataset('per_recall', data=self.local_recall)
            # hf.create_dataset('per_f1score', data=self.local_f1score)

            hf.close()


    def apriori_clusters(self):
        if self.target == 10:
            self.cluster_dict_user_id = { 0 : ['50','25','55','28','30'],
                                     1 : ['18','52','38','34','60','17','16'],
                                     2 : ['44','53','45','47','57','41','48'],
                                     3 : ['56','22','37','35'],
                                     4 : ['19','32','33','23','26','54','61','43','46','49','31','27','39','29','62','42']
                                    }
        elif self.target == 3:
            self.cluster_dict_user_id = { 0 : ['47', '45', '48', '55', '16', '31', '62', '61', '57', '39', '41', '53', '17', '18'],
                                     1 : ['27','46', '42', '60', '29', '34', '36','23', '43', '30', '25', '28', '44'],
                                     2 : ['37', '56', '19', '54', '33', '32', '38', '22', '49', '51', '52', '26', '35']
                                    }
            
        self.cluster_dict = {cluster : [] for cluster in self.cluster_dict_user_id}


        for cluster, user_ids in self.cluster_dict_user_id.items():
            for user in self.users:
                if user.id in user_ids:
                    self.cluster_dict[cluster].append(user)

        clustered_users_ids = {cluster: [user.id for user in users] for cluster, users in self.cluster_dict.items()}
        print(f" cluster is created : {clustered_users_ids}")

    def aggregare_clusters_at_round_0(self):
        for clust_id in range(self.n_clusters):
            users = np.array(self.cluster_dict[clust_id])
            for user in users:
                self.add_parameters_clusters(user, 1/len(users), clust_id)

    

    def train(self):
        loss = []
        # self.apriori_clusters()
        for t in trange(self.num_glob_iters, desc=f" exp no : {self.exp_no} number of clients: {self.num_users} Global Rounds :"):
            
            self.selected_users = self.select_users(t, int(self.num_users)).tolist()
            list_user_id = []
            for user in self.selected_users:
                list_user_id.append(user.id)
            
            similarity_dict = self.similarity_check()
            # print(similarity_dict)
            r_ik_dict = self.find_rik_dict(similarity_dict)
            # print(r_ik_dict)
            self.create_cluster(r_ik_dict)
            # print(self.cluster_dict)
            # input("pause")
            self.aggregate_clusterhead(r_ik_dict)
            
            for clust_id in range(self.n_clusters):
                users = np.array(self.cluster_dict[clust_id])
                print(f"number of users {len(users)} in cluster : {clust_id}")
                if len(users) != 0:
                    for user in users:
                        user.train(self.c[clust_id])
            # if t == 0:  
            #    self.aggregare_clusters_at_round_0()

                    
            
        
            # self.evaluate_localmodel()
            self.evaluate_clusterhead(t)
        # self.evaluate()

        self.save_results()
        #self.save_model()


    