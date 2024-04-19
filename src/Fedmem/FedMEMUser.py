import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split,  TensorDataset
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import os
import glob
import copy
import pandas as pd
import numpy as np
from src.Optimizer.Optimizer import Fedmem
from src.utils.data_process import FeatureDataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

class Fedmem_user():

    def __init__(self,device, model, args, id, exp_no, current_directory):

        self.device = device
        
        self.id = id  # integer
        self.batch_size = args.batch_size
        self.exp_no = exp_no
        self.current_directory = current_directory
        """
        Hyperparameters
        """
        self.learning_rate = args.alpha
        self.local_iters = args.local_iters
        self.eta = args.eta
        self.global_model_name = args.model_name
        self.algorithm = args.algorithm
        self.cluster_type = args.cluster
        self.data_silo = args.data_silo
        self.target = args.target
        self.num_users = args.total_users * args.users_frac 
        self.fixed_user_id = args.fixed_user_id
        """
        DataLoaders
        """

        # Load dataset
        features_folder = '/proj/sourasb-220503/FedMEM/dataset/r3_mem_ResNet50_features'
        # List all files in the directory
        files = glob.glob(os.path.join(features_folder, '*'))

        # Count the files
        total_samples = len(files)

        

        if self.data_silo == 100:
            if args.target == 10:
                annotations_file_train = '/proj/sourasb-220503/FedMEM/dataset/clients/data_silo/mem_s/' + str(self.data_silo) + '/' + 'Client_ID_' + str(self.id) +'_training.csv'
                annotations_file_test = '/proj/sourasb-220503/FedMEM/dataset/clients/data_silo/mem_s/' + str(self.data_silo) + '/' + 'Client_ID_' + str(self.id) +'_validation.csv'

            else:
                annotations_file_train = '/proj/sourasb-220503/FedMEM/dataset/clients/data_silo/A1/' + str(self.data_silo) + '/'+ 'Client_ID_' + str(self.id) +'_training.csv'
                annotations_file_test = '/proj/sourasb-220503/FedMEM/dataset/clients/data_silo/A1/' + str(self.data_silo) + '/'+ 'Client_ID_' + str(self.id) +'_validation.csv'
        
        else:
            if args.target == 10:
                annotations_file_train = '/proj/sourasb-220503/FedMEM/dataset/clients/data_silo/mem_s/' + str(self.data_silo) + '/' + 'Client_ID_' + str(self.id) + '/' +  'Client_ID_training_' + str(self.exp_no) + '.csv'
                annotations_file_test = '/proj/sourasb-220503/FedMEM/dataset/clients/data_silo/mem_s/' + str(self.data_silo) + '/' + 'Client_ID_' + str(self.id) + '/' + 'Client_ID_' + str(self.id) +'_validation.csv'

            else:
                annotations_file_train = '/proj/sourasb-220503/FedMEM/dataset/clients/data_silo/A1/' + str(self.data_silo) + '/' + 'Client_ID_' + str(self.id) + '/' +  'Client_ID_training_' + str(self.exp_no) + '.csv'
                annotations_file_test = '/proj/sourasb-220503/FedMEM/dataset/clients/data_silo/A1/' + str(self.data_silo) + '/' + 'Client_ID_' + str(self.id) + '/' + 'Client_ID_' + str(self.id) +'_validation.csv'
        
        
        print(annotations_file_train)
        print(annotations_file_test)
        
        
        self.train_dataset = FeatureDataset(features_folder, annotations_file_train)
        self.test_dataset = FeatureDataset(features_folder, annotations_file_test)

        # Split dataset into training and validation

        # train_set, val_set = train_test_split(self.train_dataset, test_size=0.2, random_state=self.exp_no)


        self.train_samples = len(self.train_dataset)
        self.test_samples = len(self.test_dataset)

        self.ratio = (self.train_samples + self.test_samples)/total_samples
        
        # DataLoaders
        torch.manual_seed(self.exp_no)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False)

        
        self.trainloaderfull = DataLoader(self.train_dataset, self.train_samples)
        
        

        # those parameters are for persionalized federated learing.
        
        self.local_model = copy.deepcopy(model)
        self.eval_model = copy.deepcopy(model)
        self.best_local_model = copy.deepcopy(model)
        self.local_model.to(self.device)
        self.eval_model.to(self.device)
        self.best_local_model.to(self.device)
        # print(self.local_model)
        # print(self.eval_model)

        # Freeze all layers
        for param in self.local_model.parameters():
            param.requires_grad = False

        # Unfreeze the second last layer
        for param in self.local_model.fc1.parameters():
            param.requires_grad = True

        # Unfreeze the last layer
        for param in self.local_model.fc2.parameters():
            param.requires_grad = True

        """
        Loss
        """

        self.loss = nn.CrossEntropyLoss()

        self.minimum_loss = 0.0
        self.maximum_per_accuracy = 0.0
        self.maximum_per_f1 = 0.0

        self.list_accuracy = []
        self.list_f1 = []
        self.list_val_loss = []

        """
        Optimizer
        """
        # self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=bb_step)
        self.optimizer = torch.optim.SGD([{'params': self.local_model.fc1.parameters()},{'params': self.local_model.fc2.parameters()}
                                        ], lr=self.learning_rate, weight_decay=0.001)  # Only optimize the last layer

        #self.optimizer = Fedmem(self.local_model.parameters(), self.learning_rate, self.eta)


    #def bb_step(optimizer, grad, step_size):

    def set_parameters(self, cluster_model):
        for param, glob_param in zip(self.local_model.parameters(), cluster_model):
            param.data = glob_param.data.clone()
            # print(f"user {self.id} parameters : {param.data}")
        # input("press")
            
    def get_parameters(self):
        return self.local_model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param

    def get_updated_parameters(self):
        return self.local_weight_updated

    def update_parameters(self, new_params):
        for param, new_param in zip(self.eval_model.parameters(), new_params):
            param.data = new_param.data.clone()


    def test(self, global_model, t):
        # Set the model to evaluation mode
        self.eval_model.eval()
        self.update_parameters(global_model)
        y_true = []
        y_pred = []
        
        total_loss = 0.0
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.eval_model(inputs)
                loss = self.loss(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())  # Collect true labels
                y_pred.extend(predicted.cpu().numpy())  # Collect predicted labels

        # Convert collected labels to numpy arrays for metric calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        validation_loss = total_loss / len(self.test_loader)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        cm = confusion_matrix(y_true,y_pred)  
        
        file_cm = "cm_user_" + str(self.id) + "_GR_" + str(t)
        #print(file)
       
        directory_name = "global" 
        cm_df = pd.DataFrame(cm)

        if not os.path.exists(self.current_directory + "/results/confusion_matrix/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/results/confusion_matrix/"+ directory_name)
        
        cm_df.to_csv(self.current_directory + "/results/confusion_matrix/"+ directory_name + "/" + file_cm + ".csv", index=False)

        # print(f"local model saved at global round :{t} local round :{iter}")       
        return accuracy, validation_loss, precision, recall, f1, cm

    def test_local(self, t):
        self.best_local_model.eval()
        loss = 0
    
        y_true = []
        y_pred = []
        
        total_loss = 0.0
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.best_local_model(inputs)
                loss = self.loss(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())  # Collect true labels
                y_pred.extend(predicted.cpu().numpy())  # Collect predicted labels

        # Convert collected labels to numpy arrays for metric calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        validation_loss = total_loss / len(self.test_loader)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        cm = confusion_matrix(y_true,y_pred)
        self.list_accuracy.append([accuracy, t])
        self.list_f1.append([f1, t]) 
        self.list_val_loss.append([validation_loss,t])       
        return accuracy, validation_loss, precision, recall, f1, cm


    def train_error_and_loss(self, global_model):
      
        # Set the model to evaluation mode
        self.eval_model.eval()
        self.update_parameters(global_model)
        y_true = []
        y_pred = []
        
        total_loss = 0.0
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, labels in self.trainloaderfull:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.eval_model(inputs)
                loss = self.loss(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())  # Collect true labels
                y_pred.extend(predicted.cpu().numpy())  # Collect predicted labels

        # Convert collected labels to numpy arrays for metric calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        train_loss = total_loss / len(self.trainloaderfull)
        # precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        # recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        # f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
                
        return accuracy, train_loss

    def train_error_and_loss_local(self):
        self.best_local_model.eval()
        loss = 0
    
        y_true = []
        y_pred = []
        
        total_loss = 0.0
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, labels in self.trainloaderfull:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs)
                loss = self.loss(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())  # Collect true labels
                y_pred.extend(predicted.cpu().numpy())  # Collect predicted labels

        # Convert collected labels to numpy arrays for metric calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        train_loss = total_loss / len(self.trainloaderfull)
        # precision = precision_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted
        # recall = recall_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted
        # f1 = f1_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted
                
        return accuracy, train_loss

    
    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.local_model = torch.load(os.path.join(model_path, "server" + ".pt"))

    
#    @staticmethod
#    def model_exists():
#        return os.path.exists(os.path.join("models", "server" + ".pt"))
    def evaluate_model(self, epoch, t):
        self.local_model.eval()  # Set the model to evaluation mode
        y_true = []
        y_pred = []
        total_loss = 0.0
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs)
                loss = self.loss(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())  # Collect true labels
                y_pred.extend(predicted.cpu().numpy())  # Collect predicted labels

        # Convert collected labels to numpy arrays for metric calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        validation_loss = total_loss / len(self.test_loader)
        # precision = precision_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted
        # recall = recall_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted
        f1 = f1_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted
        
        if epoch == 0:
            self.maximum_per_accuracy = accuracy
            self.maximum_per_f1 = f1
            for best_param, param in zip(self.best_local_model.parameters(), self.local_model.parameters()):
                best_param.data = param.data.clone()
            cm = confusion_matrix(y_true, y_pred)
            # print(f"new maximum personalized accuracy of client {self.id} found at global round {t} local epoch {epoch}")
            self.save_local_model(epoch, cm, t)
        else:
            if accuracy > self.maximum_per_accuracy:
                self.maximum_per_accuracy = accuracy
                self.maximum_per_f1 = f1
                # print(f"new maximum personalized accuracy of client {self.id} found at global round {t} local epoch {epoch}")
                cm = confusion_matrix(y_true, y_pred)
                for best_param, param in zip(self.best_local_model.parameters(), self.local_model.parameters()):
                    best_param.data = param.data.clone()
                self.save_local_model(epoch, cm, t)
                
                
        
    
    def save_local_model(self, iter, cm, t):
        cm_df = pd.DataFrame(cm)
        
        # file = "per_model_user"+ str(self.id) +"_exp_no_" + str(self.exp_no) + "_LR_" + str(iter) + "_GR_" + str(t) 
        file = "per_model_user_" + str(self.id)
        file_cm = "cm_user_" + str(self.id)
        #print(file)
       
        # directory_name =  "fixed_client_" + str(self.fixed_user_id) + "/" + str(self.global_model_name) + "/" + str(self.algorithm) + "/data_silo_" + str(self.data_silo) + "/" + "target_" + str(self.target) + "/" + str(self.cluster_type) + "/" + str(self.num_users) + "/" + str(self.exp_no)
        directory_name =  "Cemnet" + str(self.fixed_user_id) + "/" + str(self.global_model_name) + "/" + str(self.algorithm) + "/data_silo_" + str(self.data_silo) + "/" + "target_" + str(self.target) + "/" + str(self.cluster_type) + "/" + str(self.num_users) + "/" + str(self.exp_no)

        
        
        # Check if the directory already exists
        if not os.path.exists(self.current_directory + "/models/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/models/"+ directory_name)
        if not os.path.exists(self.current_directory + "/results/confusion_matrix/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/results/confusion_matrix/"+ directory_name)
        
        torch.save(self.local_model.state_dict(),self.current_directory + "/models/"+ directory_name + "/" + file + ".pt")
        cm_df.to_csv(self.current_directory + "/results/confusion_matrix/"+ directory_name + "/" + file_cm + ".csv", index=False)

        # print(f"local model saved at global round :{t} local round :{iter}")

    def train(self, cluster_model, t):

        for epoch in range(self.local_iters):  # Loop over the dataset multiple times
            self.local_model.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                # Forward + backward + optimize
                outputs = self.local_model(inputs)
                loss = self.loss(outputs, labels)              
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            self.evaluate_model(epoch, t)
            
