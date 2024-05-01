import torch
import os
import h5py
from src.dynamic_FedDCPrivacy.dynamic_user import User
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
            self.user_ids = args.user_ids[3][:20]
        else:
            self.user_ids = args.user_ids[2]
        

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
        self.wandb = wandb.init(project="DIPA2", name="dynamic_FedDCPrivacy_%s_%d" % (date_and_time, self.total_users), mode=None if args.wandb else "disabled")
                
        
        for i in trange(self.total_users, desc="Data distribution to clients"):
            # print(f"client id : {self.user_ids[i]}")
            user = User(device, args, self.user_ids[i], exp_no, current_directory, wandb)
            if user.valid:
                self.users.append(user)
                self.total_samples += user.samples
                
                if self.user_ids[i] == str(self.fixed_user_id):
                    self.fixed_user = user
                    # print(f'id found : {self.fixed_user.id}')
        # print("Finished creating Fedmem server.")

        self.total_users = len(self.users) 
        self.num_users = self.total_users * args.users_frac    #selected users
        

        self.global_model = copy.deepcopy(self.users[0].local_model)

        for _ in range(self.n_clusters):
            self.c.append(copy.deepcopy(list(self.global_model.parameters())))
          


        for user in self.users:
                
            self.data_frac.append([user, user.id, user.samples/self.total_samples])
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
            
            print(len(resourceful))
            print(len(resourceless))

       # print(resourceful)
       # print(resourceless)
        self.participated_rf_clients = self.kappa*len(resourceful)  #selected resourceful users
        self.participated_rl_clients = self.delta*len(resourceless) #selected resourceful users
        self.num_users = self.participated_rf_clients + self.participated_rl_clients

        # cluster formation
        self.clusters = [resourceful, resourceless] 
       # print(self.clusters)
        

        for user in self.clusters[0]:
            self.data_in_cluster[0] += user.samples
            
        for user in self.clusters[1]:
            self.data_in_cluster[1] += user.samples
        
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
            model_path = self.current_directory + "/models/" + self.algorithm + "/global_model/"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            checkpoint = {'GR': glob_iter,
                        'model_state_dict': self.global_model.state_dict(),
                        'loss': self.minimum_test_loss
                        }
            torch.save(checkpoint, os.path.join(model_path, "best_server_checkpoint" + ".pt"))

        with open(os.path.join(model_path, "best_clusters.pickle"), 'wb') as handle:
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
                server_param.data += local_param.data.clone() * (user.samples/self.samples)


    def add_parameters_clusters(self, user, cluster_id):
        
        for cluster_param, user_param in zip(self.c[cluster_id], user.get_parameters()):
            cluster_param.data = cluster_param.data + (user.samples/self.samples)* user_param.data.clone()
      
    def aggregate_clusterhead(self):

        for clust_id in range(self.n_clusters):
            for param in self.c[clust_id]:
                param.data = torch.zeros_like(param.data)
        
            users = np.array(self.cluster_dict[clust_id])
            # print(users)
            # input("press")
            # print(f"number of users are {len(users)} in cluster {clust_id} ")
            if len(users) != 0:
                for user in users:
                    self.add_parameters_clusters(user, clust_id)

    def find_cluster_id(self, user_id):
        for key, values in self.cluster_dict_user_id.items():
            if user_id in values:
                return key
        return None




    def train(self):
        loss = []

        for t in trange(self.num_glob_iters, desc=f" exp no : {self.exp_no} number of clients: {self.num_users} / Global Rounds :"):
            self.samples = 0.0
            subset_rf = len(self.clusters[0])
            subset_rl = len(self.clusters[1])
            
            self.selected_rf_users = self.select_users(t,0, subset_rf).tolist()
            self.selected_rl_users = self.select_users(t,1, subset_rl).tolist()
            self.selected_users = self.selected_rf_users + self.selected_rl_users

            for user in self.selected_users:
                self.samples += user.samples

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


    def test(self):
        for user in self.users:
            
            #if user.train_samples > 30:
            print("User ID", user.id)
            user.test()
            #sys.exit()