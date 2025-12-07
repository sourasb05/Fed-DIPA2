import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch
import os
import h5py
from src.Fedmem.FedMEMUser import Fedmem_user
import numpy as np
import copy
from tqdm import trange
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
import statistics
import sys
import wandb
import datetime
import json
from torchmetrics import Precision, Recall, F1Score
from src.utils.results_utils import InformativenessMetrics
import pickle

class Fedmem():
    def __init__(self,device, args, exp_no, current_directory):
                
        self.device = device
        self.num_glob_iters = args.num_global_iters
        self.local_iters = args.local_iters
        self.batch_size = args.batch_size
        self.learning_rate = args.alpha
        self.eta = args.eta
        self.country = args.country
        if args.country == "both":
            self.user_ids = args.user_ids[3]
        else:
            self.user_ids = args.user_ids[2]

        self.cluster_save_path =  "_best_clusters.pickle"

        if args.model_name == "openai_ViT-L/14@336px":
            self.model_name = "ViT-L_14_336px"
        else:
            self.model_name = args.model_name

        self.total_users = len(self.user_ids)
        self.num_teams = args.num_teams
        self.total_train_samples = 0
        self.exp_no = exp_no
        self.n_clusters = args.num_teams
        self.gamma = args.gamma # scale parameter for RBF kernel 
        self.lambda_1 = args.lambda_1 # similarity tradeoff
        self.lambda_2 = args.lambda_2
        self.current_directory = current_directory
        self.algorithm = args.algorithm
        self.cluster_type = args.cluster
        self.cluster_dict = {}
        self.clusters_list = []
        self.c = []
        self.cluster_dict_user_id = {}
        #self.c = [[] for _ in range(args.num_teams)]
        self.top_accuracies = []
        self.global_metric = []
 

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

        self.minimum_clust_loss = 0.0
        self.minimum_global_loss = 0.0

        self.data_frac = []

        date_and_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.wandb = wandb.init(project="DIPA2", name="Fedmem%s_%d" % (date_and_time, self.total_users), mode=None if args.wandb else "disabled")
        

        for i in trange(self.total_users, desc="Data distribution to clients"):
            # id, train, test = read_user_data(i, data)
           # print(f"client id : {self.user_ids[i]}")
            user = Fedmem_user(device, args, self.user_ids[i], exp_no, current_directory, wandb)
            if user.valid: # Copy for all algorithms
                self.users.append(user)
                self.total_train_samples += user.train_samples
            self.total_users = len(self.users)
            self.num_users = int(self.total_users)*args.p  #selected users
        print(self.num_users)
        

        #Create Global_model
        for user in self.users:
            self.data_frac.append(user.train_samples/self.total_train_samples)
        print(f"data available {self.data_frac}")
        self.global_model = copy.deepcopy(self.users[0].local_model)
        
        """
        Clusterhead models
        """

        for _ in range(self.n_clusters):
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
            """if len(users) != 0:
                for param in self.c[clust_id]:
                    print(f" cluster {clust_id} parameters :{param.data}")
                """
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
                server_param.data += local_param.data.clone() * (user.train_samples/self.samples)


    def add_parameters_clusters(self, user, cluster_id):
        
        for cluster_param, user_param in zip(self.c[cluster_id], user.get_parameters()):
            cluster_param.data = cluster_param.data + (user.train_samples/self.samples)* user_param.data.clone()
      
    def aggregate_clusterhead(self):

        for clust_id in range(self.num_teams):
            for param in self.c[clust_id]:
                param.data = torch.zeros_like(param.data)
        
            users = np.array(self.cluster_dict[clust_id])
            # print(users)
            # input("press")
            # print(f"number of users are {len(users)} in cluster {clust_id} ")
            if len(users) != 0:
                for user in users:
                    self.add_parameters_clusters(user, clust_id)


    def select_users(self, round, subset_users):
        np.random.seed(round)
        return np.random.choice(self.users, subset_users, replace=False)


    def select_n_1_users(self, round, subset_users):
        # Filter out the permanent user from the list of users
        # print(f"fixed client : {self.fixed_user.id}")
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

    def save_global_model(self, glob_iter):
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
            torch.save(checkpoint, os.path.join(model_path, "best_server_checkpoint" + ".pt"))
        cluster_path = self.current_directory + "/models/" + self.algorithm + "/global_model/"
        # print(f"cluster path :", cluster_path)
        if not os.path.exists(cluster_path):
            os.makedirs(cluster_path)
        with open(os.path.join(cluster_path, self.cluster_save_path), 'wb') as handle:
            pickle.dump(self.cluster_dict_user_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
    def initialize_or_add(self, dest, src):
    
        for key, value in src.items():
            if key in dest:
                dest[key] = [x + y for x, y in zip(dest[key], value)]
            else:
                dest[key] = value.copy()  # Initialize with a copy of the first list


    def eval_train_local(self, t):
        avg_loss = 0.0
        avg_distance = 0.0
        avg_mae = 0.0
        accumulator = {}
        for c in self.selected_users:
            loss, distance, c_dict, mae = c.train_evaluation_local(t)
            avg_loss += (1/len(self.selected_users))*loss
            avg_distance += (1/len(self.selected_users))*distance
            avg_mae += (1/len(self.selected_users))*mae
            if c_dict:  # Check if test_dict is not None or empty
                self.initialize_or_add(accumulator, c_dict)
        average_dict = {key: [x / len(self.selected_users) for x in value] for key, value in accumulator.items()}

        self.wandb.log(data={ "global_train_loss" : avg_loss})

        self.local_train_metric.append(average_dict)
        self.local_train_loss.append(avg_loss)
        self.local_train_distance.append(avg_distance)
        self.local_train_mae.append(avg_mae)

        # print(f"Global round {t} Local :: Train ::")           
        # print(f"Train loss {avg_loss} avg distance {avg_distance}") 
        # print(f"Train Performance metric : {average_dict}")
        # print(f"Train global mae : {avg_mae}")



    
  
    def eval_test_local(self, t):
        avg_loss = 0.0
        avg_distance = 0.0
        accumulator = {}
        avg_mae = 0.0
        for c in self.selected_users:
            loss, distance, c_dict, mae = c.test_local(t)
            avg_loss += (1/len(self.selected_users))*loss
            avg_distance += (1/len(self.selected_users))*distance
            avg_mae += (1/len(self.selected_users))*mae
            if c_dict:  # Check if test_dict is not None or empty
                self.initialize_or_add(accumulator, c_dict)
        average_dict = {key: [x / len(self.selected_users) for x in value] for key, value in accumulator.items()}

        
        self.wandb.log(data={ "local_val_loss" : avg_loss})
        self.wandb.log(data={ "local_mae" : avg_mae})
        self.wandb.log(data={ "local_Accuracy"  : average_dict['Accuracy'][0]})
        self.wandb.log(data={ "local_precision"  : average_dict['Precision'][0]})
        self.wandb.log(data={ "local_Recall" : average_dict['Recall'][0]})
        self.wandb.log(data={ "local_f1"  : average_dict['f1'][0]})

        self.local_test_metric.append(average_dict)
        self.local_test_loss.append(avg_loss)
        self.local_test_distance.append(avg_distance)
        self.local_test_mae.append(avg_mae)

        # print(f"Global round {t} Local :: Test ::")           
        # print(f"test loss {avg_loss} avg distance {avg_distance}") 
        # print(f"test Performance metric : {average_dict}")
        # print(f"test global mae : {avg_mae}")

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

                    
        # print(f"Global round {t} Global Train loss {avg_loss} avg distance {avg_distance}") 
        # print(f"Train Performance metric : {average_dict}")
        # print(f"Train global mae : {avg_mae}")



    
  
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
            
                    
        # print(f"Global round {t} Global Test loss {avg_loss} avg distance {avg_distance}") 
        # print(f"Test Performance metric : {average_dict}")
        # print(f"Test global mae : {avg_mae}")


    def evaluate(self, t):
        self.eval_test(t)
        self.eval_train(t)

    """def evaluate_local(self, t):
        self.eval_test_local(t)
        self.eval_train_local(t)"""
    
    def evaluate_local(self, t):
        val_avg_mae = 0.0
        val_avg_cmae = 0.0
        test_avg_mae = 0.0
        test_avg_cmae = 0.0
        for c in self.users:
            info_prec, info_rec, info_f1, info_cmae, info_mae, result_dict = c.test_local_model_val()
            test_info_prec, test_info_rec, test_info_f1, test_info_cmae, test_info_mae, test_result_dict = c.test_local_model_test()
            
            # print(f"info_prec {info_prec}, info_rec {info_rec}, info_f1 {info_f1}, info_cmae {info_cmae}, info_mae {info_mae}")
            
            val_avg_mae += (1/len(self.selected_users))*info_mae
            val_avg_cmae += (1/len(self.selected_users))*info_cmae
            test_avg_mae += (1/len(self.selected_users))*test_info_mae
            test_avg_cmae += (1/len(self.selected_users))*test_info_cmae
        
        print(f"\033[92m\n Global round {t} : Local val cmae {val_avg_cmae} Local val mae : {val_avg_mae} \n\033[0m")   # Green
        print(f"\033[93m\n Global round {t} : Local Test cmae {test_avg_cmae} Local Test mae : {test_avg_mae} \n\033[0m")  # Yellow



    
    def find_cluster_id(self, user_id):
        for key, values in self.cluster_dict_user_id.items():
            if user_id in values:
                return key
        return None
    
    def convert_numpy(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def save_local_results(self):
        for user in self.users:
            val_dict = user.val_round_result_dict
            test_dict = user.test_round_result_dict
            """if user in self.selected_rf_users:
                user_cat = "Resourceful user"
            else:
                user_cat = "Resourceless user"
            """
            user_id = str(user.id)
            ####### Cluster ablation 
            # val_json_path = f"results/client_level/CFedDC_rl1_C{self.n_clusters}/local_val/user_{user_id}_val_round_results.json"
            # test_json_path = f"results/client_level/CFedDC_rl1_C{self.n_clusters}/local_test/user_{user_id}_test_round_results.json"
            ####### kappa and delta ablation
            val_json_path = f"results/client_level/exp_{self.exp_no}_model_name_{self.model_name}_FedMEM/local_val/user_{user_id}_val_round_results.json"
            test_json_path = f"results/client_level/exp_{self.exp_no}_model_name_{self.model_name}_FedMEM/local_test/user_{user_id}_test_round_results.json"
            os.makedirs(os.path.dirname(val_json_path), exist_ok=True)
            os.makedirs(os.path.dirname(test_json_path), exist_ok=True)
            # print(f"Saving to {val_json_path} (Category: {user_cat})")
            # print(f"Saving to {test_json_path} (Category: {user_cat})")


            # Combine resource category and val_dict into one JSON object
            val_full_output = {
              #  "resource_category": user_cat,
                "validation_results": val_dict
            }

            test_full_output = {
              #  "resource_category": user_cat,
                "validation_results": test_dict
            }

            # Ensure the parent folder exists
            os.makedirs(os.path.dirname(val_json_path), exist_ok=True)
            os.makedirs(os.path.dirname(test_json_path), exist_ok=True)


            # Save to JSON file (overwrite if it exists)
            with open(val_json_path, 'w') as f:
                json.dump(val_full_output, f, indent=2, default=self.convert_numpy)
            # Save to JSON file (overwrite if it exists)
            with open(test_json_path, 'w') as f:
                json.dump(test_full_output, f, indent=2, default=self.convert_numpy)




    def train(self):
        loss = []
        
        for t in trange(self.num_glob_iters, desc=f" exp no : {self.exp_no} cluster type : {self.cluster_type} number of clients: {self.num_users} Global Rounds :"):

            
            self.samples = 0.0
            self.selected_users = self.select_users(t, int(self.num_users)).tolist()
            list_user_id = []
            for user in self.selected_users:
                list_user_id.append(user.id)
                self.samples += user.train_samples
            # print(f"selected users : {list_user_id}")
            
            
            for user in tqdm(self.selected_users, desc=f"total selected users {len(self.selected_users)}"):
                clust_id = self.find_cluster_id(user.id)
                # print(f"clust_id : {clust_id}")
                if clust_id is not None:
                    user.train(self.c[clust_id],t)
                else:
                    user.train(self.global_model.parameters(),t)

            similarity_matrix = self.similarity_check()
            clusters = self.spectral(similarity_matrix, self.n_clusters).tolist()
            print(f"Global round {t} clusters : {clusters}")
            self.combine_cluster_user(clusters)
            # self.save_clusters(t)
            
            self.aggregate_clusterhead()
            self.global_update()

            self.evaluate_local(t)
            # self.evaluate(t)
        self.save_local_results()
        
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
        
