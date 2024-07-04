import torch
import os
import h5py
from src.apriori_FedDCPrivacy.user import User
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
from sklearn.cluster import KMeans
import pickle

from torchmetrics import Precision, Recall, F1Score
from src.utils.results_utils import CalculateMetrics, InformativenessMetrics
import pprint
from sklearn.cluster import KMeans

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
        self.gamma=args.gamma
        self.lambda_1=args.lambda_1
        self.lambda_2=args.lambda_2

        self.country = args.country
        if args.country == "japan":
            self.user_ids = args.user_ids[0]
        elif args.country == "uk":
            self.user_ids = args.user_ids[1]
        elif args.country == "both":
            self.user_ids = args.user_ids[3]
        else:
            self.user_ids = args.user_ids[2]
        
        self.cluster_save_path =  "delta_" + str(self.delta) + "_kappa_" + str(self.kappa) + "_best_clusters.pickle"

        self.total_users = len(self.user_ids)
        self.total_samples = 0
        self.total_selected_samples = 0
        self.exp_no = exp_no
        self.current_directory = current_directory
        self.algorithm = args.algorithm
        self.fix_client_every_GR = args.fix_client_every_GR
        self.fixed_user_id = args.fixed_user_id
        self.n_clusters = args.num_teams
        self.cluster_dict = {}
        self.clusters_list = []
        self.c = []
        self.cluster_dict_user_id = {}

        # print(f"self.fixed_user_id : {self.fixed_user_id}")

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
        self.wandb = wandb.init(project="DIPA2", name="%s_%s_%d" % (self.algorithm, date_and_time, self.total_users), mode=None if args.wandb else "disabled")
                
        #self.read_all_cluster_information()

        if args.test: self.read_cluster_information()
        
        count = 0

        for i in trange(self.total_users, desc="Data distribution to clients"):
            # print(f"client id : {self.user_ids[i]}")
            user = User(device, args, self.user_ids[i], exp_no, current_directory, wandb)
            if user.valid:
                self.users.append(user)
                self.total_samples += user.train_samples
                
                if self.user_ids[i] == str(self.fixed_user_id):
                    self.fixed_user = user
                    # print(f'id found : {self.fixed_user.id}')
                
                if args.test and user.model_status == False:
                    count+=1

        # print("Finished creating Fedmem server.")

        self.total_users = len(self.users) 
        self.num_users = self.total_users * args.users_frac    #selected users
        
        print("Total Users Present :", self.total_users)

        self.global_model = copy.deepcopy(self.users[0].local_model)

        for _ in range(self.n_clusters):
            self.c.append(copy.deepcopy(list(self.global_model.parameters())))
       
        for user in self.users:
                
            self.data_frac.append([user, user.id, user.train_samples/self.total_samples])
            # print(f"data available {self.data_frac}")

            # Step 2: Sort the list in descending order
            self.data_frac = sorted(self.data_frac, key=lambda x: x[2], reverse=True)
            # print(self.data_frac)
            
            resourceful = []
            cum_sum = 0.0
            for value in self.data_frac:
                # print(value[2])
                cum_sum += value[2]
                resourceful.append(value[0])

                if cum_sum >= 0.5:
                    break
            
            resourceless = [x for x in self.users if x not in resourceful]
            

            #print(len(resourceful))
            #print(len(resourceless))

       # print(resourceful)
       # print(resourceless)
        self.participated_rf_clients = self.kappa*len(resourceful)  #selected resourceful users
        self.participated_rl_clients = self.delta*len(resourceless) #selected resourceful users
        self.num_users = self.participated_rf_clients + self.participated_rl_clients

        # cluster formation
        self.clusters = [resourceful, resourceless] 
       # print(self.clusters)

        for user in self.clusters[0]:
            self.data_in_cluster[0] += user.train_samples
            
        for user in self.clusters[1]:
            self.data_in_cluster[1] += user.train_samples

        #self.apriori_clusters = self.get_apriori_clusters(resourceful)
        self.cluster_dict = self.get_apriori_clusters(resourceful)

        self.resourceless = resourceless
        self.resourceful = resourceful
        
        
    def get_apriori_clusters(self, users):

        print("Total Users", len(users))

        bigfives = []
        for user in users:
            bigfives.append(user.train_df[users[0].bigfives].iloc[0].tolist())

        clusterer = KMeans(n_clusters=self.n_clusters, random_state=10)
        clusters_ids = clusterer.fit_predict(bigfives)

        clusters = {}
        for i, cid in enumerate(clusters_ids):
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append(users[i])

        return clusters

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
            model_path = self.current_directory + "/models/" + self.algorithm + "/global_model/" + "delta_" + str(self.delta) + "_kappa_" + str(self.kappa) + "/"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            checkpoint = {'GR': glob_iter,
                        'model_state_dict': self.global_model.state_dict(),
                        'loss': self.minimum_test_loss
                        }
            torch.save(checkpoint, os.path.join(model_path, "server_checkpoint_GR" + str(glob_iter) + ".pt"))
            
        if self.global_test_loss[glob_iter] < self.minimum_test_loss:
            self.minimum_test_loss = self.global_test_loss[glob_iter]
            model_path = self.current_directory + "/models/" + self.algorithm + "/global_model/" + "delta_" + str(self.delta) + "_kappa_" + str(self.kappa) + "/"
            print(f"model path :", model_path)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            checkpoint = {'GR': glob_iter,
                        'model_state_dict': self.global_model.state_dict(),
                        'loss': self.minimum_test_loss
                        }
            torch.save(checkpoint, os.path.join(model_path, "best_server_checkpoint" + ".pt"))
        cluster_path = self.current_directory + "/models/" + self.algorithm + "/global_model/"
        print(f"cluster path :", cluster_path)
        if not os.path.exists(cluster_path):
            os.makedirs(cluster_path)
        with open(os.path.join(cluster_path, self.cluster_save_path), 'wb') as handle:
            pickle.dump(self.cluster_dict_user_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def select_users(self, round, switch, num_subset_users):
        if switch == 0:
            np.random.seed(round)
            return np.random.choice(self.clusters[0], num_subset_users, replace=False) 
        elif switch == 1:
            np.random.seed(round)
            return np.random.choice(self.clusters[1], num_subset_users, replace=False)
        else: 
            assert (switch > 1)
                
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


    def evaluate(self, t):
        self.eval_test(t)
        self.eval_train(t)

    def evaluate_cluster(self, t):
        self.eval_test(t)
        self.eval_train(t)

    def save_results(self):
       
        file = "exp_no_" + str(self.exp_no) + self.algorithm + "_GR_" + str(self.num_glob_iters) + "_BS_" + str(self.batch_size) + "_delta_" + str(self.delta) + "_kappa_" + str(self.kappa)
        
        print(file)
       
        directory_name = str(self.algorithm) + "/" +"h5" + "/global_model/" + "delta_" + str(self.delta) + "_kappa_" + str(self.kappa) + "/" 
        # Check if the directory already exists
        if not os.path.exists(self.current_directory + "/results/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/results/" + directory_name)

        json_test_metric = json.dumps(self.global_test_metric)
        json_train_metric = json.dumps(self.global_train_metric)


        with h5py.File(self.current_directory + "/results/" + directory_name + "/" + '{}.h5'.format(file), 'w') as hf:
            hf.create_dataset('Global rounds', data=self.num_glob_iters)
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
        #print(f"eigenvalues : {eigenvalues}, eigenvectors : {eigenvectors}")
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


    def add_parameters(self, cluster_model, ratio):
        for server_param, cluster_param in zip(self.global_model.parameters(), cluster_model):
            server_param.data += cluster_param.data.clone() * ratio 

    def global_update(self):
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)
        
        for user in self.selected_users:
            for server_param, local_param in zip(self.global_model.parameters(), user.local_model.parameters()):
                server_param.data += local_param.data.clone() * (user.train_samples/self.samples)

    def add_parameters_clusters(self, user, cluster_id):
        
        for cluster_param, user_param in zip(self.c[cluster_id], user.get_parameters()):
            cluster_param.data = cluster_param.data + (user.train_samples/self.samples)* user_param.data.clone()
      
    def aggregate_clusterhead(self):

        for clust_id in range(self.n_clusters):
            for param in self.c[clust_id]:
                param.data = torch.zeros_like(param.data)
        
            users = np.array(self.cluster_dict[clust_id])

            if len(users) != 0:
                for user in users:
                    self.add_parameters_clusters(user, clust_id)

    def aggregate_clusterhead_rl(self):

        for clust_id in range(self.n_clusters):
            if clust_id not in self.cluster_dict_rl:
                continue

            users = np.array(self.cluster_dict_rl[clust_id])

            if len(users) != 0:
                for user in users:
                    self.add_parameters_clusters(user, clust_id)

    def find_cluster_id(self, user_id):
        for key, values in self.cluster_dict_user_id.items():
            if user_id in values:
                return key
        return None
    
    def set_parameters(self, model, params):
        for param, glob_param in zip(model.parameters(), params):
            param.data = glob_param.data.clone()
    
    def train(self):

        for t in trange(self.num_glob_iters, desc=f"Global Rounds"):

            # Training ResourceFul in Clustered Setting
            self.samples = 0.0

            for cluster_id in range(self.n_clusters):
                for user in self.cluster_dict[cluster_id]:
                    self.samples += user.train_samples

            for cluster_id in tqdm(range(self.n_clusters), desc=f"Training RF"):
                for user in self.cluster_dict[cluster_id]:
                    user.train(self.c[cluster_id],t)

            self.aggregate_clusterhead()

            # Identifying Best Clusters for ResourceLess
            self.cluster_dict_rl = {}

            for user in tqdm(self.resourceless, desc=f"Evaluating Clusters for RL"):
                min_cmae = 1000
                min_cluster_id = -1
                for cluster_id in range(self.n_clusters):   
                    info_cmae = user.test_model(self.c[cluster_id])
                    if min_cmae > info_cmae:
                        min_cmae = info_cmae
                        min_cluster_id = cluster_id
                
                user.cluster_id = min_cluster_id

                if min_cluster_id not in self.cluster_dict_rl:
                    self.cluster_dict_rl[min_cluster_id] = []
                self.cluster_dict_rl[min_cluster_id].append(user)

            # Daisy Chaining
            for cluster_id in tqdm(range(self.n_clusters), desc=f"Daisy Chaining"):
                if cluster_id not in self.cluster_dict_rl:
                    continue

                self.cluster_dict_rl[cluster_id][0].train(self.c[cluster_id],t)

                if len(self.cluster_dict_rl[cluster_id]) == 1:
                    continue

                for ui in range(len(self.cluster_dict_rl[cluster_id])):

                    src_user_id = ui
                    dst_user_id = ui + 1
                    if dst_user_id == len(self.cluster_dict_rl[cluster_id]):
                        dst_user_id = 0

                    src_user = self.cluster_dict_rl[cluster_id][src_user_id]
                    dst_user = self.cluster_dict_rl[cluster_id][dst_user_id]
                    dst_user.train(src_user.get_parameters(), t)

            # Aggregation with RL
            for cluster_id in range(self.n_clusters):
                if cluster_id in self.cluster_dict_rl:
                    for user in self.cluster_dict_rl[cluster_id]:
                        self.samples += user.train_samples

            self.selected_users = []
            for cluster_id in range(self.n_clusters):
                for user in self.cluster_dict[cluster_id]:
                    self.selected_users.append(user)
                
                if cluster_id in self.cluster_dict_rl:
                    for user in self.cluster_dict_rl[cluster_id]:
                        self.selected_users.append(user)       

            self.aggregate_clusterhead_rl()   
            self.global_update()

            self.evaluate(t)
            self.save_model(t)

        self.save_results()

    def train_org(self):
        loss = []

        for t in trange(self.num_glob_iters, desc=f" exp no : {self.exp_no} number of clients: {self.num_users} / Global Rounds :"):
            self.samples = 0.0
            subset_rf = int(len(self.clusters[0])*self.kappa)
            subset_rl = int(len(self.clusters[1])*self.delta)
            
            self.selected_rf_users = self.select_users(t,0, subset_rf).tolist()
            self.selected_rl_users = self.select_users(t,1, subset_rl).tolist()
            self.selected_users = self.selected_rf_users + self.selected_rl_users

            for user in self.selected_users:
                self.samples += user.train_samples

            exchange_dict = {key: random.sample(self.selected_rl_users, 2) for key in self.selected_rf_users} 

            list_user_id = [[],[]]
            for user in self.selected_rf_users:
                #print(f"rf : {user.id}")
                list_user_id[0].append(user.id)
            for user in self.selected_rl_users:
                #print(f"rl : {user.id}")
                list_user_id[1].append(user.id)
            
            # print(f"selected users : {list_user_id}")
            
            for user in tqdm(self.selected_rf_users, desc=f"selected users from resourceful {len(self.selected_rf_users)}"):
                clust_id = self.find_cluster_id(user.id)
                # print(f"clust_id : {clust_id}")

                if clust_id is not None:
                    user.train(self.c[clust_id],t)
                else:
                    user.train(self.global_model.parameters(),t)
            for user in tqdm(self.selected_rl_users, desc=f"total selected users  from resourceless {len(self.selected_rl_users)}"):
                clust_id = self.find_cluster_id(user.id)
                # print(f"clust_id : {clust_id}")
                if clust_id is not None:
                    user.train(self.c[clust_id],t)
                else:
                    user.train(self.global_model.parameters(),t)
            for user in tqdm(self.selected_rf_users, desc=f"model exchange training"):
                user.exchange_train(exchange_dict[user], t)
            
            similarity_matrix = self.similarity_check()
            clusters = self.spectral(similarity_matrix, self.n_clusters).tolist()
            print(clusters)
            self.combine_cluster_user(clusters)

            self.aggregate_clusterhead()
            self.global_update()
            self.evaluate(t)
            self.save_model(t)
        self.save_results()

    def read_cluster_information(self):
        cluster_path = self.current_directory + "/models/" + self.algorithm + "/cluster_model/"
        print(f"cluster path :", cluster_path)
        with open(os.path.join(cluster_path, self.cluster_save_path), 'rb') as handle:
            self.cluster_dict_user_id = pickle.load(handle)

        new_user_ids = []
        for user_id in self.user_ids:
            user_id_present = False
            for cluster_user_ids in self.cluster_dict_user_id.values():
                if user_id in cluster_user_ids:
                    user_id_present = True
            if user_id_present:
                new_user_ids.append(user_id)

        self.user_ids = new_user_ids
        self.total_users = len(self.user_ids)

    def read_all_cluster_information(self):

        for delta in [0.1, 0.5, 0.8, 1.0]:
            for kappa in [0.1, 0.5, 0.8, 1.0]:

                if (delta != 1.0 and kappa == 1.0) or \
                   (delta == 1.0 and kappa != 1.0):
                    continue

                cluster_save_path =  "delta_" + str(delta) + "_kappa_" + str(kappa) + "_best_clusters.pickle"

                cluster_path = self.current_directory + "/models/" + self.algorithm + "/cluster_model/"
                print(f"cluster path :", cluster_path)
                with open(os.path.join(cluster_path, cluster_save_path), 'rb') as handle:
                    self.cluster_dict_user_id = pickle.load(handle)
                
                # pprint.pprint(self.cluster_dict_user_id)

                count = 0
                for values in self.cluster_dict_user_id.values():
                    count += len(values)

                print(delta, kappa, count)

    def test_all(self):
        output_channel = {'informationType': 6, 'sharingOwner': 7, 'sharingOthers': 7}
        all_results = []
        for user in self.users:
            #print("User ID", user.id)
            results = user.test_eval()
            all_results.extend(results)
        
        results_data = CalculateMetrics(all_results)
        for i, k in enumerate(output_channel.keys()):
            for metric, values in results_data.items():
                if metric != "info_metrics":
                    print("%.02f " % values[i], end="")

        info_prec, info_rec, info_f1, info_cmae, info_mae = results_data["info_metrics"]
        print("%.02f %.02f %.02f %.02f %.02f" % (info_prec, info_rec, info_f1, info_cmae, info_mae))

    def test_client_level(self):
        output_channel = {'informationType': 6, 'sharingOwner': 7, 'sharingOthers': 7}

        all_results_data = []
        for user in self.users:
            results = user.test_eval()
            all_results_data.append(CalculateMetrics(results))

        all_results = []
        for results_data in all_results_data:
            results = []
            for i, k in enumerate(output_channel.keys()):
                for metric, values in results_data.items():
                    if metric != "info_metrics":
                        results.append(values[i])        

            results.extend(results_data["info_metrics"])
            all_results.append(results)
        
        results_mean = np.mean(all_results, axis=0)
        for result in results_mean:
            print("%.02f" % result, end=" ")
        print()

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
        
