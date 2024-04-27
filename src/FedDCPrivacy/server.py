import torch
import os
import h5py
from src.FedDCPrivacy.user import user
import numpy as np
import copy
from datetime import date
from tqdm import trange
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import statistics
import sys

class Fedmem():
    def __init__(self,device, args, exp_no, current_directory):
                
        self.device = device
        self.num_glob_iters = args.num_global_iters
        self.local_iters = args.local_iters
        self.batch_size = args.batch_size
        self.learning_rate = args.alpha
        self.eta = args.eta
        self.country = args.country
        if args.country == "japan":
            self.user_ids = args.user_ids[2]
        else:
            self.user_ids = args.user_ids[1]

        # print(f"user ids : {self.user_ids}")
        self.total_users = len(self.user_ids)
        print(f"total users : {self.total_users}")
        elf.total_train_samples = 0
        self.exp_no = exp_no
        self.gamma = args.gamma # scale parameter for RBF kernel 
        self.lambda_1 = args.lambda_1 # similarity tradeoff
        self.lambda_2 = args.lambda_2
        self.current_directory = current_directory
        self.algorithm = args.algorithm
        self.data_silo = args.data_silo
        self.fix_client_every_GR = args.fix_client_every_GR
        self.fixed_user_id = args.fixed_user_id

        print(f"self.fixed_user_id : {self.fixed_user_id}")
        self.cluster_dict = {}
        self.clusters_list = []
        self.c = []
        self.cluster_dict_user_id = {}
        #self.c = [[] for _ in range(args.num_teams)]
        
        self.global_metric = []

        """
        Clusterhead models
        """

        # for _ in range(self.n_clusters):
        #    self.c.append(copy.deepcopy(list(self.global_model.parameters())))
            

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


        self.minimum_clust_loss = 0.0
        self.minimum_global_loss = 0.0

        self.data_frac = []

        self.data_in_cluster = [0.0,0.0]
        
        for i in trange(self.total_users, desc="Data distribution to clients"):
            print(f"client id : {self.user_ids[i]}")
            user = Fedmem_user(device, args, self.user_ids[i], exp_no, current_directory)
            self.users.append(user)
            self.total_samples += user.samples
            
        
            if self.user_ids[i] == str(self.fixed_user_id):
                self.fixed_user = user
                print(f'id found : {self.fixed_user.id}')
        # print("Finished creating Fedmem server.")

        #Create Global_model
        for user in self.users:
            self.data_frac.append([user.id, user.samples])
        print(f"data available {self.data_frac}")
        self.global_model = copy.deepcopy(self.users[0].local_model)
        
        # Step 1: Determine the threshold (let's use median for simplicity)
        threshold = sorted([item[1] for item in self.data_frac])[len(self.data_frac)//2]

        # Step 2: Divide into two clusters
        resourceful = [item[0] for item in data if item[1] >= threshold]
        resourceless = [item[0] for item in data if item[1] < threshold]
        
        self.participated_rf_clients = self.kappa*len(resourceful)  #selected resourceful users
        self.participated_rl_clients = (1-self.kappa)*len(resourceless) #selected resourceful users
        self.num_users = self.participated_rf_clients + self.participated_rl_clients
        s
        # cluster formation
        self.clusters = [resourceful, resourceless] 

        for user in self.cluster[0]:
            self.data_in_cluster[0] += user.samples
            
        for user in self.cluster[1]:
            self.data_in_cluster[1] += user.samples
            
       """
        Clusterhead models
        """

        for _ in range(len(self.clusters)):
            self.c.append(copy.deepcopy(list(self.global_model.parameters())))
            

    def send_global_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.global_model.parameters())
    
    def send_cluster_parameters(self):
        for clust_id in range(self.num_teams):
            users = np.array(self.cluster_dict[clust_id])
            users_id = np.array(self.cluster_dict_user_id[clust_id])
            # print(f"cluster {clust_id} model has been sent to {len(users_id)} users {users_id}")
            if len(users) != 0:
                #for param in self.c[clust_id]:
                    # print(f" cluster {clust_id} parameters :{param.data}")
                    # input("press")
                for user in users:
                    user.set_parameters(self.c[clust_id])

    def add_parameters(self, cluster_model, ratio):
        for server_param, cluster_param in zip(self.global_model.parameters(), cluster_model):
            server_param.data += cluster_param.data.clone() * ratio 

    def global_update(self):
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)
        
        for user in self.selected_users:
            for server_param, local_param in zip(self.global_model.parameters(), user.local_model.parameters()):
                server_param.data += local_param.data.clone() * (1/len(self.selected_users))


        
        """for cluster_model in self.c:
            self.add_parameters(cluster_model, 1/len(self.c))
        """
    def add_parameters_clusters(self, user, cluster_id):
        
        for cluster_param, user_param in zip(self.c[cluster_id], user.get_parameters()):
            # cluster_param.data = cluster_param.data + user.ratio * user_param.data.clone()
            cluster_param.data = cluster_param.data + user.samples/self.data_in_cluster[cluster_id]* user_param.data.clone()
        for cluster_param , global_param in zip(self.c[cluster_id], self.global_model.parameters()):
            cluster_param.data += self.eta*(cluster_param.data - global_param.data)

            # print(f"cluster {cluster_id} model after adding user {user.id}'s local model : {cluster_param.data} ")
        # self.c[cluster_id] = copy.deepcopy(list(self.c[cluster_id].parameters()))
    def aggregate_clusterhead(self):

        for clust_id in range(self.num_teams):
            for param in self.c[clust_id]:
                param.data = torch.zeros_like(param.data)
        
            users = np.array(self.cluster_dict[clust_id])
            
            if len(users) != 0:
                for user in users:
                    self.add_parameters_clusters(user, clust_id)


    def select_users(self, clust_id, round):
        np.random.seed(round)
        return np.random.choice(self.clusters[clust_id], self.participated_rf_clients, replace=False)


    def select_n_1_users(self, round, subset_users):
        # Filter out the permanent user from the list of users
        print(f"fixed client : {self.fixed_user.id}")
        filtered_users = [user for user in self.users if user != self.fixed_user]
    
        # Set the random seed for reproducibility
        np.random.seed(round)
    
        # Perform random selection from the filtered list of users
        selected_users = np.random.choice(filtered_users, subset_users, replace=False).tolist()
        selected_users.append(self.fixed_user)
        return selected_users



    def flatten_params(self, parameters):
        params = []
        for param in parameters:
            params.append(param.view(-1))
        return torch.cat(params)
    
    def find_similarity(self, similarity_metric, params1, params2, params_g, params_c):
                            
        if similarity_metric == "cosign similarity":
            similarity = torch.nn.functional.cosine_similarity(params1.unsqueeze(0), params2.unsqueeze(0))
        elif similarity_metric == "euclidian":
            similarity_u =  torch.exp(-self.gamma * torch.sqrt(torch.sum((params1 - params2) ** 2)))
            similarity_g =  torch.exp(-self.gamma * torch.sqrt(torch.sum((params1 + params2 - 2* params_g) ** 2)))
            similarity_c = torch.exp(-self.gamma * torch.sqrt(torch.sum((params1 + params2 - 2*params_c) ** 2)))
            
            similarity = (1-self.lambda_1 - self.lambda_2)*similarity_u + self.lambda_1*similarity_g + self.lambda_2*similarity_c
            # print(f"similarity_u : {similarity_u}, similarity_g : {similarity_g}, similarity : {similarity}")
            # print("RBF: ",similarity.item())

            #simi = torch.sqrt(torch.sum((params1 - params2) ** 2))

            
        elif similarity_metric == "manhattan":
            similarity = torch.sum(torch.abs(params1 - params2))
        elif similarity_metric == "pearson_correlation":
            similarity = self.pearson_correlation(params1, params2)

        return similarity


    def similarity_check(self):
        clust_id = 0
        similarity_matrix = {}
        # similarity_metric = "manhattan"
        similarity_metric = "euclidian"
        #print("computing cosign similarity")
        params_g = self.flatten_params(self.global_model.parameters())
        
        for user in tqdm(self.selected_users, desc="participating clients"):
            
            for key, values in self.cluster_dict.items():
                if user in values:
                    # print(f"user {user.id} is in cluster {key}")
                    clust_id = key
                    break
            
            if user.id not in similarity_matrix:
                similarity_matrix[user.id] = []
            #print(similarity_matrix)
            


            params1 = self.flatten_params(user.local_model.parameters())
            params_c = self.flatten_params(self.c[clust_id])
            
            for comp_user in self.selected_users:
                # if user != comp_user:
                    
                params2 = self.flatten_params(comp_user.local_model.parameters())

                similarity = self.find_similarity(similarity_metric, params1, params2, params_g, params_c)


               #print("user_id["+ str(user.id)+"] and user_id["+str(comp_user.id)+"] = ",similarity.item())
                    

                similarity_matrix[user.id].extend([similarity.item()])
                
        
        return similarity_matrix

    def eigen_decomposition(self, laplacian_matrix, n_components):
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
        # Sort eigenvectors by eigenvalues
        idx = np.argsort(eigenvalues)
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
        return eigenvectors[:, :n_components]
    
    def compute_laplacian(self, similarity_matrix):
        degree_matrix = np.diag(similarity_matrix.sum(axis=1))
        
        return degree_matrix - similarity_matrix



    def spectral(self, similarity_dict, n_clusters):

        size = len(similarity_dict)
        # print(size)
        matrix = np.zeros((size, size))
        # print(matrix)
        i = 0
        for key, values in similarity_dict.items():
           # print(key)
           # print(values)
            matrix[i] = values
            i+=1
        laplacian_matrix = self.compute_laplacian(matrix)
        # input("laplacian")
        # print(laplacian_matrix)

        eigenvectors = self.eigen_decomposition(laplacian_matrix, n_clusters)
        # input("eigenvectors")
        # print(eigenvectors)

        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(eigenvectors)
        return kmeans.labels_
    

    def reassign_to_new_cluster(self, key, value):
        found_key = None
        # print(f"client_id : {value.id}")

        # Now, add the value to the new key
        if key not in self.cluster_dict:
            self.cluster_dict[key] = []
            self.cluster_dict_user_id[key] = []
    
        # Search for the value in the dictionary
        for k, values in self.cluster_dict.items():
            if value in values:
                found_key = k
                # print(f"found key {found_key} for value {value.id}")
                # input("press")
            
                self.cluster_dict[found_key].remove(value)
                self.cluster_dict_user_id[found_key].remove(value.id)
                # print(self.cluster_dict_user_id)
                # input("press")
                break

        self.cluster_dict[key].append(value)
        self.cluster_dict_user_id[key].append(value.id)
        return found_key  # return the original key if the value was reassigned


              
    def combine_cluster_user(self,clusters):
        
        user_ids = []
        for user in self.selected_users:
            user_ids.append(user.id)
        for key, value, in zip(clusters, self.selected_users):
            original_key = self.reassign_to_new_cluster(key, value)
        #    if original_key is not None:
                # print(f"Value {value.id} reassigned from {original_key} to {key}.")

        # print(f"Clusters: {self.cluster_dict_user_id}")
        
        self.clusters_list.append(list(self.cluster_dict_user_id.values()))


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
        # print(f" cluster is created : {clustered_users_ids}")
        

    def save_clusters(self, t):
        file = "exp_no" + str(self.exp_no) + "_clusters_at_GR_" + str(t)
        print(file)
        # directory_name =  "/data_silo_" + str(self.data_silo) + "/" + "target_" + str(self.target) + "/" + str(self.cluster_type) + "/" + str(self.num_users) + "/" + str(self.exp_no) + "/" + "h5"
        
        directory_name =  "/data_silo_" + str(self.data_silo) + "/" + "fixed_client_" + str(self.fixed_user.id)  + "/" + "target_" + str(self.target) + "/" + str(self.cluster_type) + "/" + str(self.num_users) + "/" + "h5"

        
        if not os.path.exists(self.current_directory + "/results/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/results/" + directory_name)
        
        with h5py.File(self.current_directory + "/results/" + directory_name + "/" + '{}.h5'.format(file), 'w') as hf:
            for key, value in self.cluster_dict_user_id.items():
                hf.create_dataset(str(key), data=value)

    def save_results(self):
        file = "_exp_no_" + str(self.exp_no) + "_GR_" + str(self.num_glob_iters) + "_BS_" + str(self.batch_size) + "_data_silo_" + str(self.data_silo) + "_num_user_" + str(self.num_users)
        
        print(file)
       
        # directory_name = str(self.global_model_name) + "/" + str(self.algorithm) + "/data_silo_" + str(self.data_silo) + "/" + "target_" + str(self.target) + "/" + str(self.cluster_type) + "/" + str(self.num_users) + "/" "h5"
        
        directory_name = "fixed_client_" + str(self.fixed_user.id) + "/target_" + str(self.target)+  "/" + self.cluster_type

        # Check if the directory already exists
        if not os.path.exists(self.current_directory + "/results/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/results/" + directory_name)
        avg_highest_acc = 0
        accuracy_array = np.array([])
        each_client_accuracy_array = []
        each_client_f1_array = []
        each_client_val_loss_array = []
        for user in self.users:
            
            accuracy_array = np.append(accuracy_array, user.maximum_per_accuracy)
            # each_client_accuracy_array.append(user.list_accuracy)
            # each_client_f1_array.append(user.list_f1)
            # each_client_val_loss_array.append(user.list_val_loss)
        print(f"len(self.users) : {len(self.users)}")
        #print(f"each_client_accuracy_array : {each_client_accuracy_array}")
        #print(f"each_client_f1_array : {each_client_f1_array}")
        #print(f"each_client_val_loss_array : {each_client_val_loss_array}")
        
        
        avg_highest_acc = np.mean(accuracy_array)
        std_dev = np.std(accuracy_array, ddof=1) # ddof=1 for sample standard deviation, 0 for population


        with h5py.File(self.current_directory + "/results/" + directory_name + "/" + '{}.h5'.format(file), 'w') as hf:
            hf.create_dataset('exp_no', data=self.exp_no)
            hf.create_dataset('Global rounds', data=self.num_glob_iters)
            hf.create_dataset('Local iters', data=self.local_iters)
            hf.create_dataset('Learning rate', data=self.learning_rate)
            hf.create_dataset('Lambda_1', data=self.lambda_1)
            hf.create_dataset('Lambda_2', data=self.lambda_2)
            hf.create_dataset('Batch size', data=self.batch_size)
            hf.create_dataset('data silo', data=self.data_silo)
            hf.create_dataset('num users', data=self.num_users)
            
            # hf.create_dataset('clusters', data=self.clusters_list)
            hf.create_dataset('global_test_loss', data=self.global_test_loss)
            hf.create_dataset('global_train_loss', data=self.global_train_loss)
            hf.create_dataset('global_test_accuracy', data=self.global_test_acc)
            hf.create_dataset('global_train_accuracy', data=self.global_train_acc)
            hf.create_dataset('global_precision', data=self.global_precision)
            hf.create_dataset('global_recall', data=self.global_recall)
            hf.create_dataset('global_f1score', data=self.global_f1score)
            
            hf.create_dataset('cluster_test_loss', data=self.cluster_test_loss)
            hf.create_dataset('cluster_train_loss', data=self.cluster_train_loss)
            hf.create_dataset('cluster_test_accuracy', data=self.cluster_test_acc)
            hf.create_dataset('cluster_train_accuracy', data=self.cluster_train_acc)
            hf.create_dataset('cluster_precision', data=self.cluster_precision)
            hf.create_dataset('cluster_recall', data=self.cluster_recall)
            hf.create_dataset('cluster_f1score', data=self.cluster_f1score)

            hf.create_dataset('per_test_loss', data=self.local_test_loss)
            hf.create_dataset('per_train_loss', data=self.local_train_loss)
            hf.create_dataset('per_test_accuracy', data=self.local_test_acc)
            hf.create_dataset('per_train_accuracy', data=self.local_train_acc)
            hf.create_dataset('per_precision', data=self.local_precision)
            hf.create_dataset('per_recall', data=self.local_recall)
            hf.create_dataset('per_f1score', data=self.local_f1score)

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
        
    def save_global_model(self, t): #, cm):
        # cm_df = pd.DataFrame(cm)
        file_cm = "_exp_no_" + str(self.exp_no) + "_confusion_matrix" 
        file = "_exp_no_" + str(self.exp_no) + "_model" 
        
        print(file)
       
        directory_name = str(self.global_model_name) + "/" + str(self.algorithm) + "/data_silo_" + str(self.data_silo) + "/" + "target_" + str(self.target) + "/" + str(self.cluster_type) + "/" + str(self.num_users) + "/" + "h5"

        # Check if the directory already exists
        if not os.path.exists(self.current_directory + "/models/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/models/"+ directory_name)
        
        if not os.path.exists(self.current_directory + "/models/confusion_matrix/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/models/confusion_matrix/"+ directory_name)
        torch.save(self.global_model,self.current_directory + "/models/"+ directory_name + "/" + file + ".pt")
        # cm_df.to_csv(self.current_directory + "/models/confusion_matrix/"+ directory_name + "/" + file_cm + ".csv", index=False)
    
    
    def save_cluster_model(self, t):
        

        for cluster in range(self.num_teams):
            file = "_exp_no_" + str(self.exp_no) + "_cluster_model_" + str(cluster)
        
            print(file)
            directory_name = str(self.global_model_name) + "/" + str(self.algorithm) + "/" + str(self.target) + "/" + str(self.num_users)  + "/" +"cluster_model_" + str(cluster) 
            # Check if the directory already exists
            if not os.path.exists(self.current_directory + "/models/"+ directory_name):
            # If the directory does not exist, create it
                os.makedirs(self.current_directory + "/models/"+ directory_name)
        
            torch.save(self.c[cluster],self.current_directory + "/models/"+ directory_name + "/" + file + ".pt")

    def initialize_or_add(self, dest, src):
    
        for key, value in src.items():
            if key in dest:
                dest[key] = [x + y for x, y in zip(dest[key], value)]
            else:
                dest[key] = value.copy()  # Initialize with a copy of the first list


    def test_error_and_loss(self, t):
        # num_samples = []
        # tot_correct = []
        accs = []
        avg_loss = 0.0
        avg_distance = 0.0
        precisions = []
        recalls = []
        f1s = []
        cms = []
        accumulator = {}
        for c in self.selected_users:
            loss, distance, c_dict = c.test(self.global_model.parameters(), t)
            avg_loss += (1/len(self.selected_users))*loss
            avg_distance += (1/len(self.selected_users))*distance
            if c_dict:  # Check if test_dict is not None or empty
                self.initialize_or_add(accumulator, c_dict)
        average_dict = {key: [x / len(self.selected_users) for x in value] for key, value in accumulator.items()}

        self.global_metric.append(average_dict)
   
            
                    
        print(f"Global round {t} avg loss {avg_loss} avg distance {avg_distance}") 
        print(average_dict)           

    
    def evaluate(self, t):
        
        self.test_error_and_loss(t)
        
    def evaluate_clusterhead(self, t):
        evaluate_model = "cluster"
        test_accs, test_losses, precisions, recalls, f1s, cms = self.test_error_and_loss(evaluate_model, t)
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
        print(f"Cluster Precision: {self.cluster_precision[t]}")
        print(f"Cluster Recall: {self.cluster_recall[t]}")
        print(f"Cluster f1score: {self.cluster_f1score[t]}")

        if t == 0 and self.minimum_clust_loss == 0.0:
            self.minimum_clust_loss = self.cluster_test_loss[0]
        else:
            if self.cluster_test_loss[t] < self.minimum_clust_loss:
                self.minimum_clust_loss = self.cluster_test_loss[t]
                # print(f"new minimum loss of local model at client {self.id} found at global round {t} local epoch {epoch}")
                self.save_cluster_model(t)

    


    def train(self):
        loss = []

        for t in trange(self.num_glob_iters, desc=f" exp no : {self.exp_no} cluster type : {self.cluster_type} number of clients: {self.num_users} Global Rounds :"):
            
            self.selected_users = self.select_users(t, int(self.participated_clients)).tolist()
            
            list_user_id = []
            for user in self.selected_users:
                list_user_id.append(user.id)
            print(f"selected users : {list_user_id}")
            
            for user in tqdm(self.selected_users[0], desc=f"total selected users  from cluster {len(self.selected_users)}"):
                    user.train()
               
            if self.cluster_type == "dynamic":
                similarity_matrix = self.similarity_check()
                # print(similarity_matrix)
                clusters = self.spectral(similarity_matrix, self.n_clusters).tolist()
                print(clusters)
                self.combine_cluster_user(clusters)
                self.save_clusters(t)
            
            self.aggregate_clusterhead()
            self.global_update()

        
            self.evaluate_localmodel(t)
            self.evaluate_clusterhead(t)
            self.evaluate(t)
        
        #print(self.global_metric)
        # informative_accuracy = [d['Accuracy'][0] for d in self.global_metric if 'Accuracy' in d and len(d['Accuracy']) > 0]
        # informative_precision = [d['Precision'][0] for d in self.global_metric if 'Precision' in d and len(d['Precision']) > 0]
        # informative_recall = [d['Recall'][0] for d in self.global_metric if 'Recall' in d and len(d['Recall']) > 0]
        # informative_f1 = [d['f1'][0] for d in self.global_metric if 'f1' in d and len(d['f1']) > 0]

        # print(f"-----Informativeness-----")
        # print(f"Accuracy : {informative_accuracy}")
        # print(f"Precision : {informative_precision}")
        # print(f"Recall : {informative_recall}")
        # print(f"f1 : {informative_f1}")
        # self.save_results()
        # self.plot_per_result()
        # self.plot_cluster_result()
        # self.plot_global_result()
