import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split,  TensorDataset
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50
import torchvision.models as models
import torchvision.transforms as transforms
from src.TrainModels.trainmodels import CEMNet
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import os
import glob
import copy
import pandas as pd
import numpy as np
from src.utils.data_process import FeatureDataset_mm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import h5py
class Fedmem_user():

    def __init__(self,device, args, id, exp_no, current_directory):

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
        self.target = args.target
        self.num_users = args.total_users * args.users_frac 
        self.cluster = args.cluster
        self.algorithm = args.algorithm
        """
        DataLoaders
        """

        # Load dataset
        features_folder = current_directory + '/dataset/r3_mem_ResNet50_features'
        print(f"featires_folder: {features_folder}")
        # List all files in the directory
        files = glob.glob(os.path.join(features_folder, '*'))

        # Count the files
        total_samples = len(files)

        annotations_file_train = current_directory + '/dataset/extrinsic_features/'  + 'Client_ID_' + str(self.id) + '/' +  'Client_ID_' + str(self.id) + '_training.csv'
        annotations_file_test = current_directory + '/dataset/extrinsic_features/' + 'Client_ID_' + str(self.id) + '/' + 'Client_ID_' + str(self.id) +'_validation.csv'

        print(annotations_file_train)
        print(annotations_file_test)
        
        context_columns = ['Gender','Age','Handness','Race','Education','Language','Scan','Interval','Training','Face','People','Place','Activity','B75_Span','B75_L','B75_R','B90_Span','B90_L','B90_R','B95_Span','B95_L','B95_R','Distinc_Test','Distinc_Encode40','Distinc_Encode60','Distinc_Encode70','Mem_s']

        self.train_dataset = FeatureDataset_mm(features_folder, annotations_file_train, context_columns)
        self.test_dataset = FeatureDataset_mm(features_folder, annotations_file_test, context_columns)

        
        self.train_samples = len(self.train_dataset)
        self.test_samples = len(self.test_dataset)

        self.ratio = (self.train_samples + self.test_samples)/total_samples
        
        # DataLoaders
        torch.manual_seed(self.exp_no)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False)

        
        self.trainloaderfull = DataLoader(self.train_dataset, self.train_samples, shuffle=False)
        
        # checkpoint_path = self.current_directory + '/models/Cemnet16/ResNet50TL/Fedmem/data_silo_100/target_10/dynamic/40.0/0/per_model_user_' + str(self.id) + '.pt'
        
       # checkpoint_path = self.current_directory + '/models/ResNet50TL/Fedmem/data_silo_100/target_10/' + self.cluster + '/40.0/0/' + 'per_model_user_' + str(self.id) + '.pt'
        # checkpoint_path = self.current_directory + '/models/ResNet50TL/Fedmem/apriori/local_models/' + 'per_model_user_' + str(self.id) + '.pt'
      #  checkpoint_path = self.current_directory + '/models/ResNet50TL/FedProx/global_model/' + 'server_' + '.pt'
        
        # checkpoint_path = self.current_directory + '/models/ResNet50TL/FedAvg/global_model/exp_0/_exp_no_0_GR_28.pt'
        # checkpoint_path = self.current_directory + '/models/ResNet50TL/pFedme/10/40.0/global_model/exp_0/_exp_no_0_model.pt'
        checkpoint_path = self.current_directory + '/models/ResNet50TL/Fedmem/data_silo_100/target_10/apriori_hsgd/40.0/h5/exp_0/_exp_no_0_model.pt'
      

        """ self.local_model =  CEMNet(n_class=args.target,
                            mlp_input_size=args.mlp_input_size, 
                            mlp_hidden_size=args.mlp_hidden_size, 
                            mlp_output_size=args.mlp_output_size,
                            checkpoint_path=checkpoint_path).to(device)
        """
        self.local_model = torch.load(checkpoint_path)

        print(self.local_model)
        input("press")
        self.best_local_model = copy.deepcopy(self.local_model)
        self.best_local_model.to(self.device)
        # print(self.local_model)
        # print(self.eval_model)

        # Freeze all layers
        # print(self.local_model)
        """for param in self.local_model.parameters():
            param.requires_grad = False

        # Unfreeze the second last layer
        for param in self.local_model.fc1.parameters():
            param.requires_grad = True

        # Unfreeze the last layer
        for param in self.local_model.fc2.parameters():
            param.requires_grad = True
        
        # Unfreeze the last layer of cemnet
        for param in self.local_model.fc.parameters():
            param.requires_grad = True"""



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
        # self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=0.001)
   
    def test_local(self):
        self.best_local_model.eval()
        loss = 0
    
        y_true = []
        y_pred = []
        
        total_loss = 0.0
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, contexts, labels in self.test_loader:
                inputs, contexts, labels = inputs.to(self.device), contexts.to(self.device), labels.to(self.device)
                outputs = self.best_local_model(inputs, contexts)
                loss = self.loss(outputs.float(), labels.long())
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
        self.list_accuracy.append(accuracy)
        self.list_f1.append(f1) 
        self.list_val_loss.append(validation_loss)       
        return accuracy, f1

    def train_error_and_loss_local(self):
        self.best_local_model.eval()
        loss = 0
    
        y_true = []
        y_pred = []
        
        total_loss = 0.0
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, contexts, labels in self.test_loader:
                inputs, contexts, labels = inputs.to(self.device), contexts.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs, contexts)
                loss = self.loss(outputs.float(), labels.long())
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

    def evaluate_model(self):
        self.local_model.eval()  # Set the model to evaluation mode
        y_true = []
        y_pred = []
        total_loss = 0.0
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, contexts, labels in self.test_loader:
                inputs, contexts, labels = inputs.to(self.device), contexts.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence_scores, predicted_classes = torch.max(probabilities, dim=1)
                print(f"confidence_score of lifelogger {self.id} of {len(confidence_scores)}/{self.test_samples}: {confidence_scores}" )
                print(f"predicted_class of lifelogger {self.id} of {len(predicted_classes)}/{self.test_samples}: {predicted_classes}")
                
                test_confidence_scores_np = confidence_scores.cpu().numpy()
                test_predicted_classes_np = predicted_classes.cpu().numpy()
                extrinsic_test = self.current_directory + '/dataset/extrinsic_features/' + 'Client_ID_' + str(self.id) + '/' + 'Client_ID_' + str(self.id) +'_validation.csv'
                df_test = pd.read_csv(extrinsic_test)
                df_test['Condidence'] = test_confidence_scores_np
                df_test['predicted_mem_s'] = test_predicted_classes_np
                file_name = "Client_id_" + self.id + ".csv"
                directory_path_test = self.current_directory + "/dataset/contexual/test/h-sgd" # + self.algorithm +  "/" +  self.cluster
                file_path = os.path.join(directory_path_test, file_name )

                if not os.path.exists(directory_path_test):
                    os.makedirs(directory_path_test)
                print(df_test)
                df_test.to_csv(file_path, index=False)




            for inputs, contexts, labels in self.trainloaderfull:
                inputs, contexts, labels = inputs.to(self.device), contexts.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence_scores, predicted_classes = torch.max(probabilities, dim=1)
                print(f"confidence_score of lifelogger {self.id} of {len(confidence_scores)}/{self.train_samples}: {confidence_scores}" )
                print(f"predicted_class of lifelogger {self.id} of {len(predicted_classes)}/{self.train_samples}: {predicted_classes}")
                
                train_confidence_scores_np = confidence_scores.cpu().numpy()
                train_predicted_classes_np = predicted_classes.cpu().numpy()
                extriensic_train = self.current_directory + '/dataset/extrinsic_features/'  + 'Client_ID_' + str(self.id) + '/' +  'Client_ID_' + str(self.id) + '_training.csv'
                df_train = pd.read_csv(extriensic_train)
                df_train['Condidence'] = train_confidence_scores_np
                df_train['predicted_mem_s'] = train_predicted_classes_np

                directory_path_train = self.current_directory + "/dataset/contexual/train/h-sgd" # + self.algorithm + "/" + self.cluster
                file_name = "Client_id_" + self.id + ".csv"
                file_path = os.path.join(directory_path_train, file_name)

                if not os.path.exists(directory_path_train):
                    os.makedirs(directory_path_train)
                
                df_train.to_csv(file_path, index=False)



                """loss = self.loss(outputs.float(), labels.long())
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
        print(f" Client {self.id} [Epoch {epoch}] Validation Accuracy : {accuracy} Validation Loss : {validation_loss}")

        self.list_accuracy.append(accuracy)
        self.list_val_loss.append(validation_loss)
        self.list_f1.append(f1)
        if epoch == 0:
            self.maximum_per_accuracy = accuracy
            self.maximum_per_f1 = f1
            for best_param, param in zip(self.best_local_model.parameters(), self.local_model.parameters()):
                best_param.data = param.data.clone()
            cm = confusion_matrix(y_true, y_pred)
            # print(f"new maximum personalized accuracy of client {self.id} found at global round {t} local epoch {epoch}")
            self.save_local_model(cm)
        else:
            if accuracy > self.maximum_per_accuracy:
                self.maximum_per_accuracy = accuracy
                self.maximum_per_f1 = f1
                # print(f"new maximum personalized accuracy of client {self.id} found at global round {t} local epoch {epoch}")
                cm = confusion_matrix(y_true, y_pred)
                for best_param, param in zip(self.best_local_model.parameters(), self.local_model.parameters()):
                    best_param.data = param.data.clone()
                self.save_local_model(cm)
        # self.save_local_result(accuracy, validation_loss, f1)
        """
    
        
    
    def save_local_model(self, cm):
        cm_df = pd.DataFrame(cm)
        
        # file = "per_model_user"+ str(self.id) +"_exp_no_" + str(self.exp_no) + "_LR_" + str(iter) + "_GR_" + str(t) 
        file = "per_model_user_" + str(self.id)
        file_cm = "cm_user_" + str(self.id)
        #print(file)
       
        directory_name =  "Cemnet/" + self.cluster 

        
        
        # Check if the directory already exists
        if not os.path.exists(self.current_directory + "/models/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/models/"+ directory_name)
        if not os.path.exists(self.current_directory + "/results/confusion_matrix/"+ directory_name):
        # If the directory does not exist, create it
            os.makedirs(self.current_directory + "/results/confusion_matrix/"+ directory_name)
        
        torch.save(self.local_model,self.current_directory + "/models/"+ directory_name + "/" + file + ".pt")
        cm_df.to_csv(self.current_directory + "/results/confusion_matrix/"+ directory_name + "/" + file_cm + ".csv", index=False)

        # print(f"local model saved at global round :{t} local round :{iter}")

    def train(self):

        """for epoch in range(self.local_iters):  # Loop over the dataset multiple times
            self.local_model.train()
            running_loss = 0.0
            # print(f"self.local_model : {self.local_model}")
            for images, contexts, labels in self.train_loader:
                images, contexts, labels = images.to(self.device), contexts.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()  # Clear gradients
                outputs = self.local_model(images, contexts)  # Forward pass
                outputs = outputs.float()
                labels = labels.long()
                # print(outputs)
                # print(labels)
                loss = self.loss(outputs, labels)  # Compute loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update weights
                running_loss += loss.item()
                """
        self.evaluate_model()
        # self.save_local_result()


    def save_local_result(self):
    
            file = "per_user_" + str(self.id)
            print(file)
        
            directory_name =  "Cemnet/" + self.cluster + "/"

            
            
            # Check if the directory already exists
            if not os.path.exists(self.current_directory + "/results/Fedmem_MM/"+ directory_name):
            # If the directory does not exist, create it
                os.makedirs(self.current_directory + "/results/Fedmem_MM/"+ directory_name)
            
            with h5py.File(self.current_directory + "/results/Fedmem_MM/" + directory_name + "/" + '{}.h5'.format(file), 'w') as hf:
                
                hf.create_dataset('per_test_accuracy', data=self.list_accuracy)
                hf.create_dataset('per_val_loss', data=self.list_val_loss)
                hf.create_dataset('per_f1', data=self.list_f1)

                hf.close()