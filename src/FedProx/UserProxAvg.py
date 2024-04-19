import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import json
import numpy as np
import copy
from src.utils.data_process import FeatureDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class UserProx():

    def __init__(self,device, model, args, id, exp_no):

        self.device = device
        
        self.id = id  # integer
        
        self.batch_size = args.batch_size

        """
        Hyperparameters
        """
        self.learning_rate = args.alpha
        self.lamda = args.lambda_1
        self.local_iters = args.local_iters
        
        # those parameters are for persionalized federated learing.
        self.local_model = copy.deepcopy(model)

        
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

        """
        Optimizer
        """
        # self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=0.01)
        
        self.optimizer = torch.optim.SGD([{'params': self.local_model.fc1.parameters()},
                                            {'params': self.local_model.fc2.parameters()}
                                        ], lr=self.learning_rate, weight_decay=0.01)  # Only optimize the last layer




        # Load dataset
        # Load dataset
        features_folder = '/proj/sourasb-220503/FedMEM/dataset/r3_mem_ResNet50_features'
        if args.target == 10:
            annotations_file = '/proj/sourasb-220503/FedMEM/dataset/clients/'+ 'Client_ID_' + str(self.id) +'.csv'
        else:
            annotations_file = '/proj/sourasb-220503/FedMEM/dataset/clients/A1/'+ 'Client_ID_' + str(self.id) +'.csv'
        print(annotations_file)
        
        dataset = FeatureDataset(features_folder, annotations_file)

        # Split dataset into training and validation
        train_dataset, val_dataset = train_test_split(dataset, test_size=0.25, random_state=exp_no)
        self.train_samples = len(train_dataset)
        self.test_samples = len(val_dataset)

        # DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=124, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

        
        self.trainloaderfull = DataLoader(train_dataset, self.train_samples)
        self.testloaderfull = DataLoader(val_dataset, self.test_samples)
        
        self.iter_trainloader = iter(self.train_loader)
        self.iter_testloader = iter(self.val_loader)

        
    def set_parameters(self, model):
        for param, glob_param in zip(self.local_model.parameters(), model.parameters()):
            param.data = glob_param.data.clone()
            
    def get_parameters(self):
        for param in self.local_model.parameters():
            param.detach()
        return self.local_model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param

    def get_updated_parameters(self):
        return self.local_weight_updated

    def update_parameters(self, new_params):
        for param, new_param in zip(self.local_model.parameters(), new_params):
            param.data = new_param.data.clone()


    def test(self, global_model):
        self.local_model.eval()
        self.update_parameters(global_model)
        y_true = []
        y_pred = []
        
        total_loss = 0.0
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, labels in self.testloaderfull:
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
        validation_loss = total_loss / len(self.testloaderfull)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
                
        return accuracy, validation_loss, precision, recall, f1

    def train_error_and_loss(self, global_model):
        self.local_model.eval()
        y_true = []
        y_pred = []
        
        total_loss = 0.0
        self.update_parameters(global_model)
        
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
        train_loss = total_loss / len(self.testloaderfull)
                
        return accuracy, train_loss

    
    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.train_loader)
            (X, y) = next(self.iter_trainloader)
        return (X.to(self.device), y.to(self.device))

    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X.to(self.device), y.to(self.device))

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.local_model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.local_model = torch.load(os.path.join(model_path, "server" + ".pt"))

    
    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))



    def train(self, global_model):
 
        self.local_model.train()
        for epoch in range(self.local_iters):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.local_model(inputs)
                loss = self.loss(outputs, labels)
                proximal_term = 0.0
                for param, g_param in zip(self.local_model.parameters(), global_model.parameters()):
                    proximal_term += (self.lamda / 2) * torch.norm(param - g_param) ** 2
                loss += proximal_term
                loss.backward()
                self.optimizer.step()

