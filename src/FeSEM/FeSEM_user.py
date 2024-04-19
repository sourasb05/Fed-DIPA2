import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import copy
from src.TrainModels.trainmodels import *
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from src.utils.data_process import FeatureDataset
from sklearn.model_selection import train_test_split
from tqdm import trange

class User():

    def __init__(self, device, global_model, args, id, exp_no, current_directory):

        self.device = device
        
        self.id = id  # integer
        self.exp_no = exp_no
        self.batch_size = args.batch_size
        self.target = args.target
        
        """
        Hyperparameters
        """
        self.learning_rate = args.alpha
        self.local_iters = args.local_iters

        """
        DataLoaders
        """

        # Load dataset
        features_folder = '/proj/sourasb-220503/FedMEM/dataset/r3_mem_ResNet50_features'
        if args.target == 10:
            annotations_file_train = '/proj/sourasb-220503/FedMEM/dataset/clients/data_silo/mem_s/' + str(100) + '/' + 'Client_ID_' + str(self.id) +'_training.csv'
            annotations_file_test = '/proj/sourasb-220503/FedMEM/dataset/clients/data_silo/mem_s/' + str(100) + '/' + 'Client_ID_' + str(self.id) +'_validation.csv'
        else:
            annotations_file_train = '/proj/sourasb-220503/FedMEM/dataset/clients/data_silo/A1/' + str(100) + '/'+ 'Client_ID_' + str(self.id) +'_training.csv'
            annotations_file_test = '/proj/sourasb-220503/FedMEM/dataset/clients/data_silo/A1/' + str(100) + '/'+ 'Client_ID_' + str(self.id) +'_validation.csv'
        
        print(annotations_file_train)
        print(annotations_file_test)
        
        
        train_dataset = FeatureDataset(features_folder, annotations_file_train)
        val_dataset = FeatureDataset(features_folder, annotations_file_test)


        # dataset = FeatureDataset(features_folder, annotations_file)

        # Split dataset into training and validation
        # train_dataset, val_dataset = train_test_split(dataset, test_size=0.25, random_state=self.exp_no)
        
        self.train_samples = len(train_dataset)
        self.test_samples = len(val_dataset)

        # DataLoaders
        self.train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

        
        self.trainloaderfull = DataLoader(train_dataset, self.train_samples)
        self.testloaderfull = DataLoader(val_dataset, self.test_samples)
        
        self.iter_trainloader = iter(self.train_loader)
        self.iter_testloader = iter(self.val_loader)


        # those parameters are for persionalized federated learing.
        self.local_model = global_model

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
        
        self.optimizer = torch.optim.SGD([{'params': self.local_model.fc1.parameters()},
                                            {'params': self.local_model.fc2.parameters()}
                                        ], lr=self.learning_rate, weight_decay=0.01)  # Only optimize the last layer

    def set_parameters(self, cluster_model):
        for param, glob_param in zip(self.local_model.parameters(), cluster_model.parameters()):
            param.data = glob_param.data.clone()
            
    def get_parameters(self):
        for param in self.local_model.parameters():
            param.detach()
        return self.local_model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param

    def update_parameters(self, new_params):
        for param, new_param in zip(self.local_model.parameters(), new_params):
            param.data = new_param.data.clone()


    def test(self, global_model):
        # Set the model to evaluation mode
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
        cm = confusion_matrix(y_true,y_pred)         
        return accuracy, validation_loss, precision, recall, f1, cm

    def test_local(self):
        self.local_model.eval()
        loss = 0
    
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
        cm = confusion_matrix(y_true,y_pred)
                
        return accuracy, validation_loss, precision, recall, f1, cm


    def train_error_and_loss(self, global_model):
      
        # Set the model to evaluation mode
        self.local_model.eval()
        self.update_parameters(global_model)
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
        train_loss = total_loss / len(self.testloaderfull)
        # precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        # recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        # f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
                
        return accuracy, train_loss

    def train_error_and_loss_local(self):
        self.local_model.eval()
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
        train_loss = total_loss / len(self.testloaderfull)
        # precision = precision_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted
        # recall = recall_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted
        # f1 = f1_score(y_true, y_pred, average='weighted')  # Use 'macro' for unweighted
                
        return accuracy, train_loss


    
    
   


    def train(self, global_model):
        for param, new_param in zip(self.local_model.parameters(), global_model):
            param.data = new_param.data.clone()
        self.local_model.train()
        for epoch in range(self.local_iters):  # local update
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

    